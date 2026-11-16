# SPDX-License-Identifier: Apache-2.0
"""Image-generation engine for oMLX.

The engine wraps `mflux <https://github.com/mflux-community/mflux>`_ pipelines
and exposes them to the server the same way the audio engines are exposed:
one ``BaseNonStreamingEngine`` per model, loaded and evicted by the engine
pool, with generation running on the global MLX executor.

Two properties shape the implementation:

* **One pipeline resident at a time.** mflux's text-to-image and edit
  pipelines are separate Python classes and each holds a full copy of the
  weights, so keeping both loaded would double resident memory. The engine
  loads the pipeline the current request needs and swaps when the request kind
  changes. A swap means a reload; that is logged so the cost is visible.
* **Optional runtime.** mflux is an optional extra (``omlx[image]``). Nothing
  in this module imports it at module scope, so oMLX starts and lists models
  without it; requests fail with an actionable 503 instead.
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import io
import logging
import random
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..engine_core import get_mlx_executor
from ..image_models import (
    ImageBackendUnavailable,
    ImageGenerationError,
    ImageModelSpec,
    ImageModelUnsupported,
    ImageRequest,
    ImageRequestError,
    ImageResult,
    detect_image_spec,
    is_mflux_checkpoint,
    resolve_defaults,
    resolve_dimensions,
)
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

#: Formats ``/v1/images/*`` can return, mapped to Pillow format names.
OUTPUT_FORMATS: dict[str, tuple[str, str]] = {
    "png": ("PNG", "image/png"),
    "jpeg": ("JPEG", "image/jpeg"),
    "jpg": ("JPEG", "image/jpeg"),
    "webp": ("WEBP", "image/webp"),
}

#: Upper bound on images per request. Each image is a full denoising run, so
#: this is small by design; OpenAI's default clients ask for one.
MAX_IMAGES_PER_REQUEST = 4


class MfluxBackend:
    """Thin indirection over mflux's constructors.

    Keeping every mflux import inside this class means the engine can be unit
    tested with a stub backend, and an mflux version that drops a variant
    produces a clear error instead of an ImportError at module load.
    """

    name = "mflux"

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("mflux") is not None

    def model_config(self, spec: ImageModelSpec) -> Any:
        try:
            from mflux.models.common.config import ModelConfig
        except ImportError as error:  # pragma: no cover - guarded by available()
            raise ImageBackendUnavailable(_MISSING_MFLUX_MESSAGE) from error
        factory = getattr(ModelConfig, spec.config_method, None)
        if not callable(factory):
            raise ImageBackendUnavailable(
                f"mflux does not know the '{spec.config_method}' model config "
                f"needed for {spec.display_name}. Update mflux with: "
                "pip install -U 'omlx[image]'"
            )
        return factory()

    def pipeline(
        self,
        spec: ImageModelSpec,
        kind: str,
        *,
        model_path: str | Path | None = None,
        quantize: int | None = None,
    ) -> Any:
        """Build the mflux pipeline for ``spec``.

        ``kind`` is ``"generate"`` or ``"edit"``. Krea 2 serves both through
        one class (img2img is a flag), the other families have dedicated edit
        pipelines.
        """
        try:
            if spec.family == "flux3":
                from mflux.models.flux3.variants import Flux3Klein, Flux3KleinEdit

                pipeline_class = Flux3KleinEdit if kind == "edit" else Flux3Klein
            elif spec.family == "krea2":
                from mflux.models.krea2.variants.txt2img.krea2 import Krea2

                pipeline_class = Krea2
            elif spec.family == "qwen21":
                if kind == "edit":
                    from mflux.models.qwen21.variants.edit.qwen_image_21_edit import (
                        QwenImage21Edit,
                    )

                    pipeline_class = QwenImage21Edit
                else:
                    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import (
                        QwenImage21,
                    )

                    pipeline_class = QwenImage21
            else:  # pragma: no cover - registry only contains known families
                raise ImageModelUnsupported(
                    f"Unsupported image model family '{spec.family}'."
                )
        except ImportError as error:
            raise ImageBackendUnavailable(_MISSING_MFLUX_MESSAGE) from error

        kwargs: dict[str, Any] = {"model_config": self.model_config(spec)}
        if model_path is not None:
            kwargs["model_path"] = str(model_path)
        if quantize is not None:
            kwargs["quantize"] = int(quantize)
        return pipeline_class(**kwargs)


_MISSING_MFLUX_MESSAGE = (
    "Image generation needs the optional mflux runtime. Install it with: "
    'pip install "omlx[image]" (Apple Silicon, macOS 15+).'
)


class ImageGenerationEngine(BaseNonStreamingEngine):
    """Serve one image model through the engine pool."""

    #: Fast, allocation-free check for request handlers and endpoint hints.
    is_image_engine = True

    def __init__(
        self,
        model_name: str,
        *,
        spec: ImageModelSpec | None = None,
        quantize: int | None = None,
        backend: Any | None = None,
    ) -> None:
        """
        Args:
            model_name: Local model directory (as discovered by the pool).
            spec: Registry entry; detected from ``model_name`` when omitted.
            quantize: Optional weight quantization passed to mflux. ``None``
                keeps the checkpoint's stored precision.
            backend: mflux facade; injectable for tests.
        """
        super().__init__()
        self._model_name = model_name
        self._model_path = Path(model_name)
        self._spec = spec or detect_image_spec(self._model_path)
        if self._spec is None:
            raise ImageModelUnsupported(
                f"'{model_name}' is not a supported image model. oMLX serves "
                "FLUX 3 Image Klein, Krea 2 Turbo and Qwen-Image-2.1 through "
                "mflux; point the model directory at one of those checkpoints."
            )
        self._backend = backend or MfluxBackend()
        #: Only the real backend needs the mflux distribution to be installed.
        self._uses_mflux_backend = backend is None
        self._quantize = quantize
        self._pipeline: Any | None = None
        #: "generate", "edit", or None while nothing is loaded.
        self._pipeline_kind: str | None = None
        #: Serialises pipeline lifetime changes and generation calls.
        self._op_lock = threading.Lock()
        self._images_generated = 0
        self._last_generation_seconds: float | None = None
        self._started_at: float | None = None
        self._load_count = 0

    # ------------------------------------------------------------------
    # Identity / status
    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def spec(self) -> ImageModelSpec:
        return self._spec

    @property
    def model_type(self) -> str:
        """Discovery-facing type (``Entry.model_type`` is "image")."""
        return "image"

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "image",
            "model": self._model_name,
            "family": self._spec.family,
            "display_name": self._spec.display_name,
            "loaded": self._pipeline is not None,
            "pipeline": self._pipeline_kind,
            "quantize": self._quantize,
            "images_generated": self._images_generated,
            "last_generation_seconds": self._last_generation_seconds,
            "load_count": self._load_count,
            "uptime_seconds": (
                time.monotonic() - self._started_at if self._started_at else None
            ),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Load the text-to-image pipeline (the common case)."""
        if self._pipeline is not None:
            return
        if self._uses_mflux_backend and not MfluxBackend.available():
            raise ImageBackendUnavailable(_MISSING_MFLUX_MESSAGE)

        logger.info(
            "Starting image engine: %s (%s, mflux family=%s)",
            self._model_name,
            self._spec.display_name,
            self._spec.family,
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), self._start_sync)
        self._started_at = time.monotonic()
        logger.info("Image engine started: %s", self._model_name)

    async def stop(self) -> None:
        """Drop the pipeline and return its buffers to the system."""
        if self._pipeline is None:
            return
        logger.info("Stopping image engine: %s", self._model_name)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), self._teardown_sync)
        logger.info("Image engine stopped: %s", self._model_name)

    def _start_sync(self) -> None:
        with self._op_lock:
            self._ensure_pipeline_sync("generate")

    def _teardown_sync(self) -> None:
        with self._op_lock:
            self._pipeline = None
            self._pipeline_kind = None
            gc.collect()
            _clear_mlx_cache()
        self._started_at = None

    def _ensure_pipeline_sync(self, kind: str) -> Any:
        """Return the pipeline for ``kind``, loading or swapping as needed.

        Must run on the MLX executor (it builds mflux modules and touches
        Metal). Callers hold ``_op_lock`` or are the startup path.
        """
        if self._pipeline is not None and self._pipeline_kind == kind:
            return self._pipeline
        if self._pipeline is not None:
            # The other pipeline is a full second copy of the weights.
            logger.info(
                "Swapping image pipeline for %s: %s -> %s (weights reload)",
                self._model_name,
                self._pipeline_kind,
                kind,
            )
            self._pipeline = None
            self._pipeline_kind = None
            gc.collect()
            _clear_mlx_cache()

        # Upstream diffusers snapshots are loadable by mflux directly, but an
        # mflux-saved folder is the supported local layout. Only pass the path
        # when it is a real weight directory so a marker-only folder falls
        # back to mflux's canonical repo instead of failing weight loading.
        model_path: Path | None = self._model_path
        if not _has_weights(self._model_path):
            logger.warning(
                "No mflux/diffusers weights found in %s; mflux will download "
                "the canonical checkpoint for %s.",
                self._model_path,
                self._spec.display_name,
            )
            model_path = None
        elif not is_mflux_checkpoint(self._model_path):
            logger.info(
                "%s holds an upstream (diffusers) checkpoint; mflux loads it directly.",
                self._model_name,
            )

        try:
            pipeline = self._backend.pipeline(
                self._spec,
                kind,
                model_path=model_path,
                quantize=self._quantize,
            )
        except ImageGenerationError:
            raise
        except Exception as error:  # noqa: BLE001 - reported as a 500
            raise ImageGenerationError(
                f"Could not load {self._spec.display_name} from "
                f"{model_path or self._spec.repo_id}: {error}"
            ) from error
        self._pipeline = pipeline
        self._pipeline_kind = kind
        self._load_count += 1
        return pipeline

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    async def generate(self, request: ImageRequest) -> ImageResult:
        """Text-to-image. Runs the pipeline on the MLX executor."""
        self._validate_output_format(request.output_format)
        steps, guidance = _resolve_sampling(self._spec, request)
        width, height = _resolve_size(self._spec, request.width, request.height)
        seed = _resolve_seed(request.seed)
        self._validate_family_sampling(guidance, request.negative_prompt)

        activity = self._begin_activity(
            "image_generation",
            detail=request.prompt[:120],
            total_items=request.n,
            metadata={"model": self._model_name, "steps": steps},
        )
        started = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            images, final_width, final_height = await loop.run_in_executor(
                get_mlx_executor(),
                self._generate_sync,
                request,
                steps,
                guidance,
                width,
                height,
                seed,
            )
        finally:
            await self._finish_activity(activity)

        elapsed = time.monotonic() - started
        self._record_generation(elapsed)
        return ImageResult(
            images=images,
            mime_type=OUTPUT_FORMATS[request.output_format][1],
            model_id=self._spec.model_id,
            seed=seed,
            width=final_width,
            height=final_height,
            steps=steps,
            guidance=guidance,
            mode="generate",
            elapsed_seconds=elapsed,
        )

    async def edit(
        self,
        request: ImageRequest,
        image_paths: Sequence[str | Path],
        mask_path: str | Path | None = None,
    ) -> ImageResult:
        """Image editing from one or more reference images.

        Reference editing (FLUX 3, Qwen-Image-2.1) and strength-based img2img
        (all families) are the same endpoint, matching OpenAI's
        ``/v1/images/edits``.
        """
        self._validate_output_format(request.output_format)
        references = [Path(path) for path in image_paths]
        if not references:
            raise ImageRequestError("At least one reference image is required.")
        if len(references) > self._spec.max_references:
            raise ImageRequestError(
                f"{self._spec.display_name} accepts at most "
                f"{self._spec.max_references} reference images; got {len(references)}."
            )
        if len(references) > 1 and not self._spec.supports_edit:
            raise ImageRequestError(
                f"{self._spec.display_name} does not support multi-reference "
                "editing. Send a single image for strength-based img2img."
            )
        if mask_path is not None and self._spec.family != "qwen21":
            raise ImageRequestError(
                f"Masked edits are only supported by Qwen-Image-2.1; "
                f"{self._spec.display_name} ignores masks."
            )
        if request.n > 1 and self._spec.supports_edit:
            raise ImageRequestError(
                "n > 1 is not supported for reference editing; send one "
                "request per image."
            )

        steps, guidance = _resolve_sampling(self._spec, request)
        try:
            width, height = resolve_edit_dimensions(
                self._spec,
                request.width,
                request.height,
                references[0],
            )
        except ValueError as error:
            raise ImageRequestError(str(error)) from error
        seed = _resolve_seed(request.seed)
        self._validate_family_sampling(guidance, request.negative_prompt)

        activity = self._begin_activity(
            "image_edit",
            detail=request.prompt[:120],
            total_items=request.n,
            metadata={
                "model": self._model_name,
                "steps": steps,
                "references": len(references),
            },
        )
        started = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            images, final_width, final_height = await loop.run_in_executor(
                get_mlx_executor(),
                self._edit_sync,
                request,
                references,
                mask_path,
                steps,
                guidance,
                width,
                height,
                seed,
            )
        finally:
            await self._finish_activity(activity)

        elapsed = time.monotonic() - started
        self._record_generation(elapsed)
        return ImageResult(
            images=images,
            mime_type=OUTPUT_FORMATS[request.output_format][1],
            model_id=self._spec.model_id,
            seed=seed,
            width=final_width,
            height=final_height,
            steps=steps,
            guidance=guidance,
            mode="edit",
            elapsed_seconds=elapsed,
            extra={"references": len(references)},
        )

    # ------------------------------------------------------------------
    # Sync implementation (MLX executor only)
    # ------------------------------------------------------------------
    def _generate_sync(
        self,
        request: ImageRequest,
        steps: int,
        guidance: float | None,
        width: int,
        height: int,
        seed: int,
    ) -> tuple[list[bytes], int, int]:
        with self._op_lock:
            pipeline = self._ensure_pipeline_sync("generate")
            encoded: list[bytes] = []
            final_size = (width, height)
            for index in range(request.n):
                kwargs: dict[str, Any] = {
                    "prompt": request.prompt,
                    "seed": seed + index,
                    "num_inference_steps": steps,
                    "width": width,
                    "height": height,
                }
                if self._spec.family == "krea2":
                    kwargs["guidance"] = guidance if guidance is not None else 1.0
                    negative = request.negative_prompt
                    if negative and kwargs["guidance"] > 1.0:
                        kwargs["negative_prompt"] = negative
                elif self._spec.family == "qwen21":
                    kwargs["guidance"] = guidance if guidance is not None else 1.0
                    if request.negative_prompt and kwargs["guidance"] > 1.0:
                        kwargs["negative_prompt"] = request.negative_prompt
                elif guidance is not None:
                    kwargs["guidance"] = guidance

                generated = _call_pipeline(pipeline, kwargs)
                encoded.append(
                    _encode_image(
                        generated.image,
                        request.output_format,
                        request.output_compression,
                    )
                )
                final_size = generated.image.size
            return encoded, final_size[0], final_size[1]

    def _edit_sync(
        self,
        request: ImageRequest,
        references: list[Path],
        mask_path: str | Path | None,
        steps: int,
        guidance: float | None,
        width: int | None,
        height: int | None,
        seed: int,
    ) -> tuple[list[bytes], int, int]:
        with self._op_lock:
            if self._spec.family == "krea2":
                return self._img2img_sync(
                    request, references[0], steps, guidance, width, height, seed
                )

            kind = "edit"
            pipeline = self._ensure_pipeline_sync(kind)
            paths = [str(path) for path in references]
            kwargs: dict[str, Any] = {
                "prompt": request.prompt,
                "seed": seed,
                "num_inference_steps": steps,
                "image_paths": paths,
            }
            if width is not None and height is not None:
                kwargs["width"] = width
                kwargs["height"] = height
            if guidance is not None:
                kwargs["guidance"] = guidance
            if request.negative_prompt and (guidance or 0) > 1.0:
                kwargs["negative_prompt"] = request.negative_prompt

            if self._spec.family == "qwen21":
                if mask_path is not None:
                    kwargs["mask_image"] = str(mask_path)
                if request.strength is not None:
                    kwargs["strength"] = _validate_strength(request.strength)
                if "width" not in kwargs:
                    kwargs["output_resolution"] = max(
                        self._spec.defaults.width, self._spec.defaults.height
                    )

            encoded: list[bytes] = []
            final_size = (
                width or self._spec.defaults.width,
                height or self._spec.defaults.height,
            )
            for index in range(request.n):
                kwargs["seed"] = seed + index
                generated = _call_pipeline(pipeline, kwargs)
                encoded.append(
                    _encode_image(
                        generated.image,
                        request.output_format,
                        request.output_compression,
                    )
                )
                final_size = generated.image.size
            return encoded, final_size[0], final_size[1]

    def _img2img_sync(
        self,
        request: ImageRequest,
        reference: Path,
        steps: int,
        guidance: float | None,
        width: int,
        height: int,
        seed: int,
    ) -> tuple[list[bytes], int, int]:
        """Strength-based img2img (Krea 2's editing mode)."""
        pipeline = self._ensure_pipeline_sync("generate")
        strength = (
            _validate_strength(request.strength)
            if request.strength is not None
            else self._spec.defaults.image_strength
        )
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "image_path": str(reference),
            "num_inference_steps": steps,
            "width": width,
            "height": height,
            "guidance": guidance if guidance is not None else 1.0,
        }
        if strength is not None:
            kwargs["image_strength"] = strength
        encoded: list[bytes] = []
        final_size = (width, height)
        for index in range(request.n):
            kwargs["seed"] = seed + index
            generated = _call_pipeline(pipeline, kwargs)
            encoded.append(
                _encode_image(
                    generated.image,
                    request.output_format,
                    request.output_compression,
                )
            )
            final_size = generated.image.size
        return encoded, final_size[0], final_size[1]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_output_format(self, output_format: str) -> None:
        if output_format not in OUTPUT_FORMATS:
            raise ImageRequestError(
                f"Unsupported output format '{output_format}'; choose one of "
                f"{', '.join(sorted(set(OUTPUT_FORMATS) - {'jpg'}))}."
            )

    def _validate_family_sampling(
        self, guidance: float | None, negative_prompt: str | None
    ) -> None:
        if (
            self._spec.family == "qwen21"
            and self._spec.key.endswith("turbo")
            and guidance is not None
            and guidance != 1.0
        ):
            raise ImageRequestError(
                f"{self._spec.display_name} samples without guidance; "
                "guidance must be 1.0 (pass negative_prompt only with a "
                "guidance-capable checkpoint)."
            )
        if negative_prompt and not self._spec.defaults.negative_prompt:
            raise ImageRequestError(
                f"{self._spec.display_name} does not use negative prompts."
            )

    def _record_generation(self, elapsed: float) -> None:
        self._images_generated += 1
        self._last_generation_seconds = elapsed


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def resolve_edit_dimensions(
    spec: ImageModelSpec,
    width: int | None,
    height: int | None,
    reference: Path,
) -> tuple[int | None, int | None]:
    """Pick the edit canvas.

    An explicit size is validated as usual. With ``size="auto"``:

    * FLUX 3 and Qwen-Image-2.1 derive the canvas from the reference images
      inside mflux (that is what their CLIs do), so ``None`` is passed through;
    * Krea 2 img2img has no such fallback, so the reference's own size is used,
      rounded down to the 16-px grid and clamped to the family range.
    """
    if width is not None or height is not None:
        return resolve_dimensions(spec, width, height)
    if spec.family in ("flux3", "qwen21"):
        return (None, None)

    try:
        from PIL import Image

        with Image.open(reference) as image:
            reference_width, reference_height = image.size
    except Exception:  # noqa: BLE001 - fall back to the family default
        logger.debug("Could not read reference size from %s", reference, exc_info=True)
        return resolve_dimensions(spec, None, None)

    def _snap(value: int) -> int:
        snapped = (value // spec.size_multiple) * spec.size_multiple
        return max(spec.min_size, min(spec.max_size, snapped))

    return resolve_dimensions(spec, _snap(reference_width), _snap(reference_height))


def _clear_mlx_cache() -> None:
    """Release Metal buffers after dropping a pipeline (no-op off-device)."""
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover - engine only runs with MLX
        return
    try:
        mx.synchronize()
        mx.clear_cache()
    except Exception:  # pragma: no cover - best effort during teardown
        logger.debug("MLX cache clear failed during image teardown", exc_info=True)


def _has_weights(model_path: Path) -> bool:
    """True when the directory looks like it holds loadable weights."""
    if not model_path.is_dir():
        return False
    if (model_path / "mflux_model.json").is_file():
        return True
    patterns = (
        "*.safetensors",
        "*/*.safetensors",
        "*.bin",
        "*/*.bin",
        "*.gguf",
    )
    return any(next(model_path.glob(pattern), None) for pattern in patterns)


def _resolve_sampling(
    spec: ImageModelSpec, request: ImageRequest
) -> tuple[int, float | None]:
    """Apply family defaults, reporting bad values as client errors."""
    try:
        return resolve_defaults(spec, request.steps, request.guidance)
    except ValueError as error:
        raise ImageRequestError(str(error)) from error


def _resolve_size(
    spec: ImageModelSpec, width: int | None, height: int | None
) -> tuple[int, int]:
    try:
        return resolve_dimensions(spec, width, height)
    except ValueError as error:
        raise ImageRequestError(str(error)) from error


def _resolve_seed(seed: int | None) -> int:
    if seed is None:
        return random.SystemRandom().randrange(0, 2**32)
    if seed < 0:
        raise ImageRequestError("seed must be non-negative.")
    return seed


def _validate_strength(strength: float) -> float:
    if not 0 < strength <= 1:
        raise ImageRequestError("strength must be in (0, 1].")
    return float(strength)


def _call_pipeline(pipeline: Any, kwargs: dict[str, Any]) -> Any:
    """Run one mflux generation call and classify its failures.

    mflux raises ``ValueError`` for invalid sampling combinations (bad size,
    wrong step count, unsupported scheduler). Those are the caller's fault and
    become 400s; everything else is a runtime failure.
    """
    try:
        return pipeline.generate_image(**kwargs)
    except ImageGenerationError:
        raise
    except (ValueError, AssertionError) as error:
        raise ImageRequestError(str(error)) from error
    except Exception as error:  # noqa: BLE001 - reported as a 500
        raise ImageGenerationError(f"{type(error).__name__}: {error}") from error


def _encode_image(
    image: Any, output_format: str, compression: int | None = None
) -> bytes:
    """Encode a Pillow image as PNG, JPEG, or WebP bytes.

    ``compression`` is OpenAI's ``output_compression`` (0-100): JPEG/WebP
    quality, or a zlib level scaled onto Pillow's 0-9 PNG range. ``None``
    keeps the defaults (quality 95, PNG level 6).
    """
    pil_format, _ = OUTPUT_FORMATS[output_format]
    if pil_format == "JPEG" and image.mode in ("RGBA", "LA", "P"):
        # JPEG has no alpha channel; composite over white like other image APIs.
        rgba = image.convert("RGBA")
        background = _white_background(rgba.size)
        background.alpha_composite(rgba)
        image = background.convert("RGB")
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {}
    if pil_format in ("JPEG", "WEBP"):
        save_kwargs["quality"] = 95 if compression is None else compression
    if pil_format == "WEBP":
        save_kwargs["method"] = 6
    if pil_format == "PNG":
        save_kwargs["compress_level"] = (
            6 if compression is None else round(compression * 9 / 100)
        )
    image.save(buffer, format=pil_format, **save_kwargs)
    return buffer.getvalue()


def _white_background(size: tuple[int, int]):
    from PIL import Image

    return Image.new("RGBA", size, (255, 255, 255, 255))
