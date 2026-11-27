# SPDX-License-Identifier: Apache-2.0
"""Krea 2 (Turbo / Raw) adapter backed by mflux's native MLX implementation."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx

from ._mflux import (
    MFLUX_INSTALL_HINT,
    BoundedCache,
    apply_load_accel,
    partition_by,
    run_context,
    safetensors_keys,
    tiling_context,
)
from .base import (
    ImageCapabilities,
    ImageDefaults,
    ImageGenerationCancelled,
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
    supports_batching = True

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
        extra_row = int(0.27e6 * tokens * 1.1)
        denoise = int((1.0e9 + 0.27e6 * tokens) * 1.1)

        # Qwen-Image VAE decode peak grows with sqrt(pixels): ~15.5 GB per
        # sqrt(megapixel) (0.26 MP -> 7.9 GB, 1 MP -> 15.6 GB, 2.4 MP ->
        # 23.3 GB). Tiled decode bounds it by the tile area.
        def decode_peak(pixels: int) -> int:
            return int(15.5e9 * math.sqrt(pixels / 1e6) * 1.1)

        pixels = width * height
        plans = [
            MemoryPlan(
                "full",
                max(denoise, decode_peak(pixels)),
                {"tile": None},
                extra_row_bytes=extra_row,
            )
        ]
        for tile in (512, 256):
            if tile * tile < pixels:
                plans.append(
                    MemoryPlan(
                        f"vae_tile_{tile}",
                        max(denoise, decode_peak(tile * tile)),
                        {"tile": tile},
                        extra_row_bytes=extra_row,
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

    def generate_batch(
        self,
        model: Any,
        params_list: list[ImageGenParams],
        on_step_list: list[StepCallback],
        plan: Optional[MemoryPlan] = None,
    ) -> list[Any]:
        if len(params_list) == 1:
            return super().generate_batch(model, params_list, on_step_list, plan)
        from mflux.models.common.config.config import Config
        from mflux.models.krea2.latent_creator.krea2_latent_creator import (
            Krea2LatentCreator,
        )
        from mflux.models.krea2.model.krea2_sampler import Krea2Sampler
        from mflux.models.krea2.variants.txt2img.krea2 import Krea2
        from mflux.utils.image_util import ImageUtil

        encoded = [
            model._encode_prompts(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                guidance=params.guidance,
            )
            for params in params_list
        ]
        groups = partition_by(
            [
                (
                    int(embeds.shape[1]),
                    None if negative is None else int(negative.shape[1]),
                )
                for embeds, negative in encoded
            ]
        )
        results: list[Any] = [None] * len(params_list)
        for indices in groups.values():
            if len(indices) == 1:
                index = indices[0]
                results[index] = self.generate(
                    model, params_list[index], on_step_list[index], plan
                )
                continue
            first = params_list[indices[0]]
            resolved = Krea2._resolve_scheduler(first.extra.get("scheduler"))
            config = Config(
                model_config=model.model_config,
                num_inference_steps=first.steps,
                height=first.height,
                width=first.width,
                guidance=first.guidance,
                scheduler=resolved,
            )
            sigmas = config.scheduler.sigmas
            steppers = {
                index: Krea2Sampler.make_stepper(
                    resolved, sigmas, params_list[index].seed
                )
                for index in indices
            }
            latents = mx.concatenate(
                [
                    Krea2LatentCreator.create_noise(
                        params_list[index].seed, first.height, first.width
                    )
                    for index in indices
                ],
                axis=0,
            )
            embeds = mx.concatenate([encoded[index][0] for index in indices], axis=0)
            negative = (
                None
                if encoded[indices[0]][1] is None
                else mx.concatenate([encoded[index][1] for index in indices], axis=0)
            )
            predict = Krea2._predict(model.transformer, embeds, negative, first.guidance)
            active = list(indices)
            with tiling_context(model, plan):
                for t in range(config.num_inference_steps):
                    step_sigma = sigmas[t]
                    velocity = predict(
                        latents=latents, timestep=step_sigma.reshape(1)
                    )
                    denoised = latents - step_sigma * velocity
                    survivors: list[int] = []
                    next_latents: list[Any] = []
                    for position, index in enumerate(active):
                        try:
                            on_step_list[index](t + 1, first.steps)
                        except ImageGenerationCancelled:
                            continue
                        next_latents.append(
                            steppers[index].step(
                                t,
                                latents[position : position + 1],
                                velocity[position : position + 1],
                                denoised[position : position + 1],
                            )
                        )
                        survivors.append(index)
                    if not survivors:
                        return results
                    latents = mx.concatenate(next_latents, axis=0)
                    mx.eval(latents)
                    if len(survivors) != len(active):
                        active = survivors
                        embeds = mx.concatenate(
                            [encoded[index][0] for index in active], axis=0
                        )
                        negative = (
                            None
                            if encoded[active[0]][1] is None
                            else mx.concatenate(
                                [encoded[index][1] for index in active], axis=0
                            )
                        )
                        predict = Krea2._predict(
                            model.transformer, embeds, negative, first.guidance
                        )
                for position, index in enumerate(active):
                    decoded = model._decode_latents(
                        latents=latents[position : position + 1],
                        prompt=params_list[index].prompt,
                        seed=params_list[index].seed,
                    )
                    results[index] = ImageUtil.to_pil(decoded)
        return results

    def release(self, model: Any) -> None:
        cache = getattr(model, "prompt_cache", None)
        if cache is not None:
            cache.clear()

    def runtime_stats(self, model: Any) -> dict:
        cache = getattr(model, "prompt_cache", None)
        out = {"prompt_cache": cache.stats()} if isinstance(cache, BoundedCache) else {}
        out["accel_status"] = getattr(model, "_omlx_accel_status", {})
        return out
