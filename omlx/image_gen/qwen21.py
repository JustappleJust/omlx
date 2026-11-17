# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 (Turbo / base) adapter: text-to-image and editing via mflux.

One loaded ``QwenImage21Edit`` serves both modes (no references = plain
text-to-image). mflux already reuses the text/reference prefix K/V across
denoise steps (``use_kv_cache``); oMLX adds cross-request caches for the
reference-conditioned prompt encoding and the VAE-encoded references.
"""

from __future__ import annotations

import hashlib
import logging
import math
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

_TRANSFORMER_CLASS = "QwenImage21Transformer2DModel"


def _array_digest(array: Any) -> str:
    import numpy as np

    data = np.asarray(array)
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{data.dtype}:{data.shape}".encode())
    h.update(data.tobytes())
    return h.hexdigest()


class _CachedVAEEncode:
    """Instance-level wrapper caching ``vae.encode`` results across requests.

    Reference images are re-encoded on every edit; repeated edits of the same
    picture (the common interactive loop) skip the VAE encoder entirely.
    """

    def __init__(self, encode, cache: BoundedCache):
        self._encode = encode
        self._cache = cache

    def __call__(self, pixels):
        key = _array_digest(pixels)
        if key in self._cache:
            return self._cache[key]
        out = self._encode(pixels)
        import mlx.core as mx

        mx.eval(out)
        self._cache[key] = out
        return out


class QwenImage21Adapter(ImageModelAdapter):
    name = "qwen_image21"
    pipeline_classes = ("QwenImage21Pipeline",)
    # Turbo: 8 fixed sigma nodes, no CFG (mflux guidance 1.0 = off).
    defaults = ImageDefaults(steps=8, guidance=1.0, size_multiple=32, max_side=2048)
    capabilities = ImageCapabilities(
        generate=True,
        edit=True,
        max_references=10,
        mask=True,
        strength=True,
        lossy_accel=("step_cache",),
        negative_prompt=True,
    )

    # ---- detection -----------------------------------------------------

    def matches(self, model_dir: Path, model_index: dict) -> bool:
        if super().matches(model_dir, model_index):
            return True
        config = read_json(model_dir / "transformer" / "config.json")
        return bool(config and config.get("_class_name") == _TRANSFORMER_CLASS)

    def _is_turbo(self, model_dir: Path) -> bool:
        index = read_json(model_dir / "model_index.json") or {}
        if index.get("sample_sigmas"):
            return True
        return "turbo" in str(model_dir).lower()

    # ---- sizing --------------------------------------------------------

    def component_files(self, model_dir: Path) -> dict[str, list[Path]]:
        return {
            "transformer": sorted((model_dir / "transformer").glob("*.safetensors")),
            "text_encoder": sorted((model_dir / "text_encoder").glob("*.safetensors")),
            "vae": sorted((model_dir / "vae").glob("*.safetensors")),
        }

    def is_prequantized(self, model_dir: Path) -> bool:
        return is_prequantized_dir(model_dir / "transformer")

    def memory_plans(self, width: int, height: int, n_refs: int = 0) -> list[MemoryPlan]:
        # Calibrated on M5 Max (mflux 0.24, 8-bit). References are resized to
        # ~1 MP each and join the token sequence.
        tokens = (width // 16) * (height // 16) + n_refs * 4096
        denoise = int((1.0e9 + 0.30e6 * tokens) * 1.15)

        def decode_peak(pixels: int) -> int:
            return int(12.0e9 * math.sqrt(pixels / 1e6) * 1.15)

        pixels = width * height
        plans = [MemoryPlan("full", max(denoise, decode_peak(pixels)), {"tile": None})]
        for tile in (512, 256):
            if tile * tile < pixels:
                plans.append(
                    MemoryPlan(f"vae_tile_{tile}", max(denoise, decode_peak(tile * tile)), {"tile": tile})
                )
        return plans

    def edit_canvas(self, references: list[Any]) -> tuple[int, int]:
        ref = references[-1]
        ratio = ref.width / ref.height
        width = math.sqrt(1024 * 1024 * ratio)
        return (
            max(32, round(width / 32) * 32),
            max(32, round(width / ratio / 32) * 32),
        )

    # ---- runtime -------------------------------------------------------

    def load(
        self, model_dir: Path, quantize: Optional[int], options: Optional[dict] = None
    ) -> Any:
        try:
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise ImportError(MFLUX_INSTALL_HINT) from exc

        config = (
            ModelConfig.qwen_image_21_turbo()
            if self._is_turbo(model_dir)
            else ModelConfig.qwen_image_21()
        )
        model = QwenImage21Edit(quantize=quantize, model_path=str(model_dir), model_config=config)
        model.prompt_cache = BoundedCache(maxsize=16, max_bytes=512 << 20)
        self._install_caches(model)
        apply_load_accel(model, options)
        return model

    def _install_caches(self, model: Any) -> None:
        # Reference-conditioned prompt encodings (mflux only caches text-only).
        model._omlx_cond_prompt_cache = BoundedCache(maxsize=8, max_bytes=1 << 30)
        model._omlx_vae_cache = BoundedCache(maxsize=16, max_bytes=1 << 30)
        original_encode_prompt = model._encode_prompt

        def encode_prompt(prompt, images):
            if not images:
                return original_encode_prompt(prompt, images)
            key = (prompt, tuple(image_digest(i) for i in images))
            cache = model._omlx_cond_prompt_cache
            if key in cache:
                return cache[key]
            out = original_encode_prompt(prompt, images)
            cache[key] = out
            return out

        model._encode_prompt = encode_prompt
        model.vae.encode = _CachedVAEEncode(model.vae.encode, model._omlx_vae_cache)

    def generate(
        self,
        model: Any,
        params: ImageGenParams,
        on_step: StepCallback,
        plan: Optional[MemoryPlan] = None,
    ):
        refs = [r.convert("RGBA") for r in params.references]
        step_cache = bool(params.accel.get("step_cache"))
        kwargs: dict[str, Any] = {
            "seed": params.seed,
            "prompt": params.prompt,
            "num_inference_steps": params.steps,
            "width": params.width,
            "height": params.height,
            "guidance": params.guidance,
            "image_paths": refs,
            "use_kv_cache": True,
            "use_step_cache": step_cache,
        }
        if params.guidance > 1:
            kwargs["negative_prompt"] = params.negative_prompt or ""
        if params.mask is not None:
            kwargs["mask_image"] = params.mask.convert("L")
        if refs and params.strength < 1:
            kwargs["strength"] = params.strength
        with run_context(model, on_step, params.steps, plan):
            result = model.generate_image(**kwargs)
        return result.image

    def release(self, model: Any) -> None:
        for attr in ("prompt_cache", "_omlx_cond_prompt_cache", "_omlx_vae_cache"):
            cache = getattr(model, attr, None)
            if cache is not None:
                cache.clear()

    def runtime_stats(self, model: Any) -> dict:
        out = {}
        for attr, label in (
            ("prompt_cache", "prompt_cache"),
            ("_omlx_cond_prompt_cache", "reference_prompt_cache"),
            ("_omlx_vae_cache", "reference_latent_cache"),
        ):
            cache = getattr(model, attr, None)
            if isinstance(cache, BoundedCache):
                out[label] = cache.stats()
        out["accel_status"] = getattr(model, "_omlx_accel_status", {})
        return out
