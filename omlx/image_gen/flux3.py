# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 Image Klein adapter (text-to-image and reference editing) via mflux.

One loaded ``Flux3KleinEdit`` serves both modes: its shared core runs plain
text-to-image when no reference tokens are passed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ._mflux import (
    MFLUX_INSTALL_HINT,
    apply_load_accel,
    BoundedCache,
    image_digest,
    is_prequantized_dir,
    read_json,
    run_context,
)
from .base import (
    ImageCapabilities,
    ImageDefaults,
    ImageGenParams,
    ImageModelAdapter,
    MemoryPlan,
    StepCallback,
)

logger = logging.getLogger(__name__)

_TRANSFORMER_CLASS = "Flux3Transformer2DModel"


class Flux3KleinAdapter(ImageModelAdapter):
    name = "flux3_klein"
    pipeline_classes = ("Flux3Pipeline", "Flux3KleinPipeline")
    # Step-distilled flow-map checkpoint: 16 steps, no CFG.
    defaults = ImageDefaults(steps=16, guidance=1.0, size_multiple=16, max_side=2048)
    capabilities = ImageCapabilities(
        generate=True,
        edit=True,
        max_references=10,
        mask=False,
        strength=False,
        negative_prompt=False,
    )

    # ---- detection -----------------------------------------------------

    def matches(self, model_dir: Path, model_index: dict) -> bool:
        if super().matches(model_dir, model_index):
            return True
        config = read_json(model_dir / "transformer" / "config.json")
        if config and config.get("_class_name") == _TRANSFORMER_CLASS:
            return (model_dir / "vae").is_dir() and (model_dir / "text_encoder").is_dir()
        return False

    def _flow_map(self, model_dir: Path) -> bool:
        config = read_json(model_dir / "transformer" / "config.json") or {}
        return bool(config.get("flow_map", True))

    # ---- sizing --------------------------------------------------------

    def component_files(self, model_dir: Path) -> dict[str, list[Path]]:
        return {
            "transformer": sorted((model_dir / "transformer").glob("*.safetensors")),
            "text_encoder": sorted((model_dir / "text_encoder").glob("*.safetensors")),
            "vae": sorted((model_dir / "vae").glob("*.safetensors")),
        }

    def quantizable_components(self) -> set[str]:
        # Measured: only the DiT shrinks as expected under load-time
        # quantization; count the text encoder at full size (conservative).
        return {"transformer"}

    def is_prequantized(self, model_dir: Path) -> bool:
        return is_prequantized_dir(model_dir / "transformer")

    def memory_plans(self, width: int, height: int, n_refs: int = 0) -> list[MemoryPlan]:
        # Calibrated on M5 Max (mflux 0.24, 8-bit). Each reference adds up to
        # one canvas worth of tokens (multi-reference: <= 1 MP each).
        tokens = (width // 16) * (height // 16) * (1 + n_refs)
        denoise = int((0.8e9 + 0.30e6 * tokens) * 1.15)

        # FLUX.2 VAE decode; tiled decode bounds it by the tile area.
        def decode_peak(pixels: int) -> int:
            return int((0.4e9 + 3.2e9 * pixels / 1e6) * 1.15)

        pixels = width * height
        plans = [MemoryPlan("full", max(denoise, decode_peak(pixels)), {"tile": None})]
        for tile in (512, 256):
            if tile * tile < pixels:
                plans.append(
                    MemoryPlan(f"vae_tile_{tile}", max(denoise, decode_peak(tile * tile)), {"tile": tile})
                )
        return plans

    def edit_canvas(self, references: list[Any]) -> tuple[int, int]:
        from mflux.models.flux3.latent_creator.flux3_reference_images import Flux3ReferenceImages

        prepared, _ = Flux3ReferenceImages.prepare([r.convert("RGB") for r in references])
        return prepared[0].size

    # ---- runtime -------------------------------------------------------

    def load(
        self, model_dir: Path, quantize: Optional[int], options: Optional[dict] = None
    ) -> Any:
        try:
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.flux3.variants.edit.flux3_klein_edit import Flux3KleinEdit
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise ImportError(MFLUX_INSTALL_HINT) from exc

        config = (
            ModelConfig.flux3_image_klein()
            if self._flow_map(model_dir)
            else ModelConfig.flux3_image_klein_base()
        )
        model = Flux3KleinEdit(quantize=quantize, model_path=str(model_dir), model_config=config)
        model.prompt_cache = BoundedCache(maxsize=16, max_bytes=512 << 20)
        # Cross-request cache of VAE-encoded reference tokens.
        model._omlx_ref_cache = BoundedCache(maxsize=8, max_bytes=1 << 30)
        apply_load_accel(model, options)
        if (options or {}).get("compile", True):
            # mflux cannot compile FLUX.3's whole step (host-side float64
            # RoPE), so fuse each block instead (exact).
            from .accel.compile import compile_blocks

            t = model.transformer
            blocks = [*t.txt_mode_blocks, *t.single_transformer_blocks]
            for mode_blocks in t.mode_blocks.values():
                blocks.extend(mode_blocks)
            model._omlx_accel_status["compiled_blocks"] = compile_blocks(blocks)
        return model

    def _encode_references(self, model: Any, references: list[Any]):
        from mflux.models.flux3.latent_creator.flux3_reference_images import Flux3ReferenceImages

        images = [r.convert("RGB") for r in references]
        prepared, box = Flux3ReferenceImages.prepare(images)
        key = tuple(image_digest(p) for p in prepared)
        cache: BoundedCache = model._omlx_ref_cache
        if key in cache:
            tokens, ids = cache[key]
        else:
            tokens, ids = model.encode_references(prepared)
            cache[key] = (tokens, ids)
        return prepared, box, tokens, ids

    def generate(
        self,
        model: Any,
        params: ImageGenParams,
        on_step: StepCallback,
        plan: Optional[MemoryPlan] = None,
    ):
        guidance = params.guidance if model.model_config.supports_guidance else None
        cond_tokens = cond_ids = crop_box = None
        width, height = params.width, params.height
        if params.references:
            from mflux.models.flux3.latent_creator.flux3_reference_images import (
                Flux3ReferenceImages,
            )

            prepared, box, cond_tokens, cond_ids = self._encode_references(model, params.references)
            explicit = params.extra.get("explicit_size", False)
            width, height, crop_box = Flux3ReferenceImages.resolve_canvas(
                prepared, box, width if explicit else None, height if explicit else None
            )
        with run_context(model, on_step, params.steps, plan):
            result = model._generate(
                seed=params.seed,
                prompt=params.prompt,
                num_inference_steps=params.steps,
                height=height,
                width=width,
                guidance=guidance,
                cond_tokens=cond_tokens,
                cond_ids=cond_ids,
                crop_box=crop_box,
            )
        return result.image

    def release(self, model: Any) -> None:
        for attr in ("prompt_cache", "_omlx_ref_cache"):
            cache = getattr(model, attr, None)
            if cache is not None:
                cache.clear()

    def runtime_stats(self, model: Any) -> dict:
        out = {}
        for attr, label in (("prompt_cache", "prompt_cache"), ("_omlx_ref_cache", "reference_cache")):
            cache = getattr(model, attr, None)
            if isinstance(cache, BoundedCache):
                out[label] = cache.stats()
        out["accel_status"] = getattr(model, "_omlx_accel_status", {})
        return out

