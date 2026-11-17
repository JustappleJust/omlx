# SPDX-License-Identifier: Apache-2.0
"""Krea 2 (Turbo / Raw) adapter backed by mflux's native MLX implementation."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

from ._mflux import (
    MFLUX_INSTALL_HINT,
    BoundedCache,
    apply_load_accel,
    run_context,
    safetensors_keys,
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

_NATIVE_TRANSFORMER_FILES = ("turbo.safetensors", "raw.safetensors")
# Top-level weight-name prefixes of the native / mflux-saved Krea 2
# transformer; "txtfusion" + "txtmlp" are unique to Krea 2.
_KREA2_KEY_PREFIXES = frozenset({"blocks", "first", "txtfusion", "txtmlp"})


def _root_weight_prefixes(model_dir: Path) -> set[str]:
    """Top-level key prefixes of the root transformer checkpoint (header only)."""
    keys: list[str] = []
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text()).get("weight_map") or {}
            keys = list(weight_map)
        except (OSError, ValueError, AttributeError):
            keys = []
    if not keys:
        candidates = [model_dir / n for n in _NATIVE_TRANSFORMER_FILES]
        candidates += sorted(model_dir.glob("*.safetensors"))
        for path in candidates:
            if not path.is_file():
                continue
            keys = safetensors_keys(path)
            break
    return {k.split(".", 1)[0] for k in keys}


_BF16_CLASS_CACHE: dict[type, type] = {}


def _run_transformer_in_bf16(transformer: Any) -> None:
    """Run the Krea 2 DiT on bf16 activations (sampler state stays float32).

    mflux feeds the float32 noise latents straight into the DiT; its first
    projection keeps that dtype, so every block (attention, MLPs, all GEMMs)
    then runs in float32 -- about 1.4x slower, and the INT8-activation GEMM
    (bf16/fp16 only) cannot engage. The reference diffusers pipeline and
    vpipe run this transformer in bf16; only the model input/output are cast,
    the scheduler update keeps accumulating in float32.
    """
    import mlx.core as mx

    cls = type(transformer)
    sub = _BF16_CLASS_CACHE.get(cls)
    if sub is None:

        def __call__(self, x, timestep, *args, **kwargs):  # noqa: N807
            out_dtype = x.dtype
            y = cls.__call__(self, x.astype(mx.bfloat16), timestep, *args, **kwargs)
            return y.astype(out_dtype)

        sub = type(f"Bf16{cls.__name__}", (cls,), {"__call__": __call__})
        _BF16_CLASS_CACHE[cls] = sub
    transformer.__class__ = sub


class Krea2Adapter(ImageModelAdapter):
    name = "krea2"
    pipeline_classes = ("Krea2Pipeline",)
    # mflux semantics: guidance == 1.0 disables CFG (Turbo is distilled, no CFG).
    defaults = ImageDefaults(steps=8, guidance=1.0)
    capabilities = ImageCapabilities(generate=True, edit=False)

    def matches(self, model_dir: Path, model_index: dict) -> bool:
        if super().matches(model_dir, model_index):
            return True
        # No model_index.json: mflux's own download (turbo.safetensors +
        # components) or an mflux-save / mflux-community conversion.
        if not ((model_dir / "vae").is_dir() and (model_dir / "text_encoder").is_dir()):
            return False
        return _KREA2_KEY_PREFIXES <= _root_weight_prefixes(model_dir)

    # ---- sizing --------------------------------------------------------

    def _transformer_files(self, model_dir: Path) -> list[Path]:
        # Mirrors Krea2WeightDefinition._select_transformer_variant.
        for name in _NATIVE_TRANSFORMER_FILES:
            p = model_dir / name
            if p.is_file():
                return [p]
        tdir = model_dir / "transformer"
        if tdir.is_dir():
            return sorted(tdir.glob("*.safetensors"))
        return sorted(p for p in model_dir.glob("*.safetensors") if p.is_file())

    def component_files(self, model_dir: Path) -> dict[str, list[Path]]:
        return {
            "transformer": self._transformer_files(model_dir),
            "text_encoder": sorted((model_dir / "text_encoder").glob("*.safetensors")),
            "vae": sorted((model_dir / "vae").glob("*.safetensors")),
        }

    def is_prequantized(self, model_dir: Path) -> bool:
        # mflux-save writes the transformer shards at the root with
        # quantization metadata; diffusers / native checkpoints are bf16.
        files = self._transformer_files(model_dir)
        if not files or files[0].name in _NATIVE_TRANSFORMER_FILES:
            return False
        if files[0].parent.name == "transformer":
            return False
        try:
            from safetensors import safe_open

            with safe_open(str(files[0]), framework="numpy") as f:
                meta = f.metadata() or {}
            level = meta.get("quantization_level")
            return level not in (None, "", "None")
        except Exception:
            return False

    def memory_plans(self, width: int, height: int, n_refs: int = 0) -> list[MemoryPlan]:
        # Calibrated on M5 Max with mflux 0.24 (bf16 activations); peaks are
        # transient bytes above the resident weights, measured with
        # mx.get_peak_memory and padded ~10%.
        #
        # Denoise: ~1.0 GB + ~0.27 MB per latent token ((W/16)*(H/16)).
        tokens = (width // 16) * (height // 16)
        denoise = int((1.0e9 + 0.27e6 * tokens) * 1.1)

        # Qwen-Image VAE decode peak grows with sqrt(pixels): ~15.5 GB per
        # sqrt(megapixel) (0.26 MP -> 7.9 GB, 1 MP -> 15.6 GB, 2.4 MP ->
        # 23.3 GB). Tiled decode bounds it by the tile area.
        def decode_peak(pixels: int) -> int:
            return int(15.5e9 * math.sqrt(pixels / 1e6) * 1.1)

        pixels = width * height
        plans = [
            MemoryPlan("full", max(denoise, decode_peak(pixels)), {"tile": None})
        ]
        for tile in (512, 256):
            if tile * tile < pixels:
                plans.append(
                    MemoryPlan(
                        f"vae_tile_{tile}",
                        max(denoise, decode_peak(tile * tile)),
                        {"tile": tile},
                    )
                )
        return plans

    # ---- runtime -------------------------------------------------------

    def load(
        self, model_dir: Path, quantize: Optional[int], options: Optional[dict] = None
    ) -> Any:
        try:
            from mflux.models.krea2 import Krea2
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise ImportError(MFLUX_INSTALL_HINT) from exc

        model = Krea2(quantize=quantize, model_path=str(model_dir))
        model.prompt_cache = BoundedCache(maxsize=8)
        if (options or {}).get("bf16_activations", True):
            _run_transformer_in_bf16(model.transformer)
        apply_load_accel(model, options)
        return model

    def generate(
        self,
        model: Any,
        params: ImageGenParams,
        on_step: StepCallback,
        plan: Optional[MemoryPlan] = None,
    ):
        with run_context(model, on_step, params.steps, plan):
            result = model.generate_image(
                seed=params.seed,
                prompt=params.prompt,
                num_inference_steps=params.steps,
                width=params.width,
                height=params.height,
                guidance=params.guidance,
                negative_prompt=params.negative_prompt,
                scheduler=params.extra.get("scheduler"),
            )
        return result.image

    def release(self, model: Any) -> None:
        cache = getattr(model, "prompt_cache", None)
        if cache is not None:
            cache.clear()

    def runtime_stats(self, model: Any) -> dict:
        cache = getattr(model, "prompt_cache", None)
        out = {"prompt_cache": cache.stats()} if isinstance(cache, BoundedCache) else {}
        out["accel_status"] = getattr(model, "_omlx_accel_status", {})
        return out
