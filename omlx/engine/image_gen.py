# SPDX-License-Identifier: Apache-2.0
"""Image-generation (text-to-image) engine.

Family-agnostic: model-specific work is delegated to an
:class:`~omlx.image_gen.base.ImageModelAdapter`. Heavy dependencies (mflux)
are imported lazily inside the adapter's ``load``.

Integration with oMLX memory management:

* load / generate / unload all run on the single MLX executor;
* the engine pool sizes and evicts it like any other model (LRU, TTL, pin);
* the process memory enforcer pushes its soft watermark through
  :meth:`set_memory_soft_limit`; before each generation the engine picks the
  fastest of the adapter's calibrated :class:`MemoryPlan`s whose transient
  peak fits under it (e.g. untiled -> 512px -> 256px tiled VAE decode), and
  refuses (``OutOfMemoryError`` -> HTTP 507) when none fits;
* while generating, an owner-scoped MLX memory limit
  (:func:`omlx.utils.metal_sync.set_chunk_memory_limit`) throttles lazy
  graph encoding instead of letting the peak overshoot;
* the Metal buffer pool is cleared after every request and on stop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import io
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..exceptions import OutOfMemoryError
from ..image_gen import DEFAULT_IMAGE_QUANTIZE
from ..image_gen.base import (
    ImageGenerationCancelled,
    ImageGenParams,
    ImageModelAdapter,
    MemoryPlan,
)
from ..image_gen.registry import find_adapter
from ..utils.metal_sync import set_chunk_memory_limit
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

_VALID_QUANTIZE = (None, 3, 4, 5, 6, 8)


def _encode_image(image: Any, fmt: str) -> bytes:
    """Encode a PIL image; keep alpha for PNG/WebP (e.g. Qwen RGBA output)."""
    buf = io.BytesIO()
    keep_alpha = fmt in ("png", "webp") and image.mode in ("RGBA", "LA")
    if keep_alpha and image.getextrema()[-1] == (255, 255):
        keep_alpha = False  # fully opaque: drop the channel
    out = image.convert("RGBA" if keep_alpha else "RGB")
    save_kwargs = {"quality": 95} if fmt != "png" else {}
    out.save(buf, format=fmt.upper(), **save_kwargs)
    return buf.getvalue()


class ImageGenEngine(BaseNonStreamingEngine):
    """Engine serving one text-to-image model."""

    def __init__(
        self,
        model_name: str,
        quantize: Optional[int] = DEFAULT_IMAGE_QUANTIZE,
        adapter: Optional[ImageModelAdapter] = None,
        accel: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if quantize == 0 or quantize == 16:
            quantize = None
        if quantize not in _VALID_QUANTIZE:
            raise ValueError(
                f"Unsupported image quantization {quantize!r}; "
                f"expected one of {_VALID_QUANTIZE}"
            )
        self._model_name = model_name
        self._quantize = quantize
        self._adapter = adapter
        self._model: Any = None
        self._kwargs = kwargs
        # Engine-level acceleration options handed to the adapter at load
        # (near-lossless ones default on) and per-run defaults for opt-in
        # lossy ones.
        self._accel: Dict[str, Any] = {"a8": True, "step_cache": False, **(accel or {})}
        # One diffusion run at a time: a single run already saturates the GPU,
        # and serialising bounds the transient memory peak to one run.
        self._lock = asyncio.Lock()
        self._soft_limit: int = 0
        self._cancel_events: Dict[str, threading.Event] = {}
        self._images_generated = 0
        self._total_steps = 0
        self._total_step_seconds = 0.0
        self._load_seconds: Optional[float] = None

    # ------------------------------------------------------------------
    # properties / stats

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def adapter(self) -> Optional[ImageModelAdapter]:
        return self._adapter

    @property
    def quantize(self) -> Optional[int]:
        return self._quantize

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def set_memory_soft_limit(self, soft_limit_bytes: int) -> None:
        self._soft_limit = max(0, int(soft_limit_bytes or 0))

    def get_stats(self) -> Dict[str, Any]:
        avg = (
            self._total_step_seconds / self._total_steps if self._total_steps else None
        )
        return {
            "model_name": self._model_name,
            "engine_type": "image",
            "adapter": self._adapter.name if self._adapter else None,
            "quantize": self._quantize,
            "loaded": self._model is not None,
            "load_seconds": self._load_seconds,
            "images_generated": self._images_generated,
            "avg_seconds_per_step": avg,
            "memory_soft_limit": self._soft_limit,
            "accel": dict(self._accel),
            "capabilities": self._adapter.capabilities.as_dict() if self._adapter else None,
            **(
                self._adapter.runtime_stats(self._model)
                if self._adapter is not None and self._model is not None
                else {}
            ),
        }

    def defaults(self) -> Dict[str, Any]:
        if self._adapter is None:
            return {}
        d = self._adapter.defaults
        return {
            "steps": d.steps,
            "guidance": d.guidance,
            "width": d.width,
            "height": d.height,
            "max_side": d.max_side,
            "min_side": d.min_side,
            "size_multiple": d.size_multiple,
        }

    # ------------------------------------------------------------------
    # lifecycle

    async def start(self) -> None:
        if self._model is not None:
            return
        path = Path(self._model_name)
        if self._adapter is None:
            self._adapter = find_adapter(path)
        if self._adapter is None:
            raise ValueError(f"No image-generation adapter matches {self._model_name}")

        adapter = self._adapter
        quantize = self._quantize
        accel = self._accel
        if quantize is not None and adapter.is_prequantized(path):
            # Pre-quantized checkpoints keep their stored bits.
            quantize = None
            self._quantize = None
        logger.info(
            "Starting image engine: %s (adapter=%s, quantize=%s)",
            self._model_name,
            adapter.name,
            quantize or "bf16",
        )

        def _load_sync():
            t0 = time.monotonic()
            model = adapter.load(path, quantize, dict(accel))
            mx.synchronize()
            mx.clear_cache()
            return model, time.monotonic() - t0

        loop = asyncio.get_running_loop()
        model, elapsed = await loop.run_in_executor(get_mlx_executor(), _load_sync)
        self._model = model
        self._load_seconds = elapsed
        logger.info("Image engine started: %s in %.1fs", self._model_name, elapsed)

    async def stop(self) -> None:
        for ev in list(self._cancel_events.values()):
            ev.set()
        if self._model is None:
            return
        logger.info("Stopping image engine: %s", self._model_name)
        model = self._model
        self._model = None
        adapter = self._adapter
        loop = asyncio.get_running_loop()

        def _release():
            if adapter is not None:
                try:
                    adapter.release(model)
                except Exception:  # pragma: no cover - best effort
                    logger.exception("Image adapter release failed")

        # Wait for an in-flight generation to observe cancellation first.
        async with self._lock:
            await loop.run_in_executor(get_mlx_executor(), _release)
        del model
        gc.collect()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info("Image engine stopped: %s", self._model_name)

    # ------------------------------------------------------------------
    # generation

    def build_params(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        references: Optional[list] = None,
        mask: Any = None,
        strength: Optional[float] = None,
        accel: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> ImageGenParams:
        if self._adapter is None:
            raise RuntimeError("Image engine not started")
        adapter = self._adapter
        caps = adapter.capabilities
        d = adapter.defaults
        references = list(references or [])
        if references:
            if not caps.edit:
                raise ValueError("This model does not support image editing.")
            if len(references) > caps.max_references:
                raise ValueError(
                    f"This model accepts at most {caps.max_references} reference images."
                )
        elif not caps.generate:
            raise ValueError("This model needs at least one reference image.")
        if mask is not None:
            if not caps.mask:
                raise ValueError("This model does not support edit masks.")
            if not references:
                raise ValueError("A mask needs a reference image.")
        if strength is not None and strength < 1:
            if not caps.strength:
                raise ValueError("This model does not support edit strength.")
            if not references:
                raise ValueError("strength needs a reference image.")
            if strength <= 0:
                raise ValueError("strength must be in (0, 1].")
        if negative_prompt and not caps.negative_prompt:
            negative_prompt = None
        explicit_size = bool(width or height)
        if references and not explicit_size:
            w, h = adapter.edit_canvas(references)
        else:
            w, h = adapter.normalize_size(width or d.width, height or d.height)
        steps = int(steps) if steps else d.steps
        if not 1 <= steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        run_accel = {"step_cache": bool(self._accel.get("step_cache"))}
        for key, value in (accel or {}).items():
            if value and key not in caps.lossy_accel:
                raise ValueError(f"Acceleration {key!r} is not available for this model.")
            run_accel[key] = bool(value)
        extra = {k: v for k, v in extra.items() if v is not None}
        extra["explicit_size"] = explicit_size
        return ImageGenParams(
            prompt=prompt,
            width=w,
            height=h,
            steps=steps,
            guidance=float(d.guidance if guidance is None else guidance),
            seed=int(seed) if seed is not None else random.randint(0, 2**31 - 1),
            negative_prompt=negative_prompt or None,
            extra=extra,
            references=references,
            mask=mask,
            strength=1.0 if strength is None else float(strength),
            accel={k: v for k, v in run_accel.items() if v},
        )

    def _plan_memory(
        self, params: ImageGenParams
    ) -> tuple[MemoryPlan, Optional[int]]:
        """Pick the fastest plan that fits; return (plan, owner_memory_limit)."""
        plans = self._adapter.memory_plans(
            params.width, params.height, len(params.references)
        )
        if not self._soft_limit:
            return plans[0], None
        active = int(mx.get_active_memory())
        budget = self._soft_limit - active
        for plan in plans:
            if plan.peak_bytes <= budget:
                if plan is not plans[0]:
                    logger.info(
                        "Image %dx%d: preferred peak %.1f GB exceeds the %.1f GB "
                        "budget; using plan %s (%.1f GB)",
                        params.width,
                        params.height,
                        plans[0].peak_bytes / 1e9,
                        budget / 1e9,
                        plan.name,
                        plan.peak_bytes / 1e9,
                    )
                return plan, self._soft_limit
        smallest = plans[-1].peak_bytes
        raise OutOfMemoryError(
            f"Generating a {params.width}x{params.height} image needs about "
            f"{smallest / 1e9:.1f} GB more memory, but only "
            f"{max(0, budget) / 1e9:.1f} GB is available under the memory "
            f"soft limit. Try a smaller size or unload other models.",
            requested_bytes=smallest,
            available_bytes=max(0, budget),
        )

    async def generate(
        self,
        params: ImageGenParams,
        n: int = 1,
        output_format: str = "png",
        request_id: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        """Generate ``n`` images. Returns ``[{"data": bytes, "seed": int}]``."""
        if self._model is None or self._adapter is None:
            raise RuntimeError("Image engine not started")
        n = max(1, int(n))
        fmt = output_format.lower()
        if fmt == "jpg":
            fmt = "jpeg"
        if fmt not in ("png", "jpeg", "webp"):
            raise ValueError(f"Unsupported output_format {output_format!r}")

        total_steps = params.steps * n
        activity_id = self._begin_activity(
            "image",
            detail=params.prompt[:80],
            total_items=total_steps,
            metadata={
                "completed_items": 0,
                "width": params.width,
                "height": params.height,
                "n": n,
                "references": len(params.references),
            },
        )
        cancel = threading.Event()
        key = request_id or activity_id
        self._cancel_events[key] = cancel
        loop = asyncio.get_running_loop()
        results: list[Dict[str, Any]] = []
        try:
            async with self._lock:
                if self._model is None:
                    raise RuntimeError("Image engine was stopped")
                model = self._model
                adapter = self._adapter
                for i in range(n):
                    run = dataclasses.replace(
                        params, seed=params.seed + i, extra=dict(params.extra)
                    )
                    plan, limit = self._plan_memory(run)
                    done_before = i * params.steps

                    def on_step(step: int, total: int, _base=done_before) -> None:
                        if cancel.is_set():
                            raise ImageGenerationCancelled()
                        self._update_activity(
                            activity_id, completed_items=_base + step
                        )

                    owner = id(self)

                    def _run_sync(run=run, plan=plan, limit=limit):
                        if limit:
                            set_chunk_memory_limit(owner, limit)
                        try:
                            t0 = time.monotonic()
                            image = adapter.generate(model, run, on_step, plan)
                            elapsed = time.monotonic() - t0
                            return _encode_image(image, fmt), elapsed
                        finally:
                            if limit:
                                set_chunk_memory_limit(owner, None)
                            mx.synchronize()
                            mx.clear_cache()

                    data, elapsed = await loop.run_in_executor(get_mlx_executor(), _run_sync)
                    self._images_generated += 1
                    self._total_steps += run.steps
                    self._total_step_seconds += elapsed
                    results.append({"data": data, "seed": run.seed, "seconds": elapsed})
            return results
        finally:
            self._cancel_events.pop(key, None)
            await self._finish_activity(activity_id)

    def cancel(self, request_id: str) -> bool:
        ev = self._cancel_events.get(request_id)
        if ev is None:
            return False
        ev.set()
        return True
