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
from ..exceptions import (
    OutOfMemoryError,
    SchedulerQueueFullError,
    is_metal_out_of_memory,
)
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


@dataclasses.dataclass
class _Row:
    """One image of one request, as scheduled by the dispatcher."""

    request: "_Request"
    index: int
    params: ImageGenParams
    force_single: bool = False


@dataclasses.dataclass
class _Request:
    """State of one ``generate`` call across its ``n`` rows."""

    request_id: str
    activity_id: str
    cancel: threading.Event
    n: int
    output_format: str
    base_params: ImageGenParams
    next_index: int
    future: asyncio.Future
    results: list = dataclasses.field(default_factory=list)


class ImageGenEngine(BaseNonStreamingEngine):
    """Engine serving one text-to-image model."""

    #: One row per request waits here; the dispatcher merges compatible rows.
    _MAX_WAITING_REQUESTS = 32
    #: How long a lone batchable row waits for peers that arrive together.
    _BATCH_WINDOW_S = 0.03

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
        self._waiting: list[_Row] = []
        self._work: Optional[asyncio.Event] = None
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._stopping = False
        self._requests: Dict[str, _Request] = {}
        self._batch_count = 0
        self._row_count = 0
        self._max_batch_rows = 0
        self._last_batch_rows = 0
        self._last_plan_name: Optional[str] = None
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
            "batching": {
                "enabled": bool(self._adapter and self._adapter.supports_batching),
                "batches": self._batch_count,
                "rows": self._row_count,
                "max_batch": self._max_batch_rows,
                "last_batch": self._last_batch_rows,
                "last_plan": self._last_plan_name,
                "waiting": len(self._waiting),
            },
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
        self._ensure_dispatcher()
        logger.info("Image engine started: %s in %.1fs", self._model_name, elapsed)

    async def stop(self) -> None:
        self._stopping = True
        for ev in list(self._cancel_events.values()):
            ev.set()
        for request in list(self._requests.values()):
            self._cancel_request(request)
        self._waiting.clear()
        if self._work is not None:
            self._work.set()
        task = self._dispatcher_task
        self._dispatcher_task = None
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - dispatcher already logs
                logger.exception("Image dispatcher stopped with an error")
        if self._model is None:
            return
        logger.info("Stopping image engine: %s", self._model_name)
        model = self._model
        self._model = None
        adapter = self._adapter
        loop = asyncio.get_running_loop()

        def _release(current):
            if adapter is not None:
                try:
                    adapter.release(current)
                except Exception:  # pragma: no cover - best effort
                    logger.exception("Image adapter release failed")

        await loop.run_in_executor(get_mlx_executor(), _release, model)
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

    def _plan_for(
        self, params: ImageGenParams, batch_size: int
    ) -> tuple[MemoryPlan, Optional[int]]:
        """Pick the fastest plan that fits ``batch_size`` rows."""
        plans = self._adapter.memory_plans(
            params.width, params.height, len(params.references)
        )
        if not self._soft_limit:
            return plans[0], None
        budget = self._soft_limit - int(mx.get_active_memory())
        for plan in plans:
            if plan.peak_for(batch_size) <= budget:
                return plan, self._soft_limit
        smallest = plans[-1].peak_for(batch_size)
        raise OutOfMemoryError(
            f"Generating a {params.width}x{params.height} image needs about "
            f"{smallest / 1e9:.1f} GB more memory, but only "
            f"{max(0, budget) / 1e9:.1f} GB is available under the memory "
            f"soft limit. Try a smaller size or unload other models.",
            requested_bytes=smallest,
            available_bytes=max(0, budget),
        )

    def _choose_batch(
        self, head: _Row, count: int
    ) -> tuple[int, MemoryPlan, Optional[int]]:
        """Largest batch of the group that fits, with its plan and owner limit."""
        params = head.params
        if head.force_single:
            plans = self._adapter.memory_plans(
                params.width, params.height, len(params.references)
            )
            return 1, plans[-1], (self._soft_limit or None)
        if count <= 1:
            plan, limit = self._plan_for(params, 1)
            return 1, plan, limit
        last: Optional[OutOfMemoryError] = None
        for size in range(count, 0, -1):
            try:
                plan, limit = self._plan_for(params, size)
            except OutOfMemoryError as exc:
                last = exc
                continue
            if size < count:
                logger.info(
                    "Image %dx%d: memory allows a batch of %d of %d requests "
                    "(plan %s, %.1f GB)",
                    params.width,
                    params.height,
                    size,
                    count,
                    plan.name,
                    plan.peak_for(size) / 1e9,
                )
            return size, plan, limit
        raise last

    @staticmethod
    def _batch_key(params: ImageGenParams) -> tuple:
        return (
            params.width,
            params.height,
            params.steps,
            round(params.guidance, 4),
            params.extra.get("scheduler"),
            tuple(sorted(params.accel.items())),
            params.strength,
        )

    def _can_batch(self, params: ImageGenParams) -> bool:
        return (
            bool(self._adapter and self._adapter.supports_batching)
            and not params.references
            and params.mask is None
            and params.strength == 1.0
        )

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher_task is not None and not self._dispatcher_task.done():
            return
        self._stopping = False
        if self._work is None:
            self._work = asyncio.Event()
        else:
            self._work.clear()
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())

    def _enqueue_row(self, request: _Request, index: int) -> None:
        base = request.base_params
        run = dataclasses.replace(base, seed=base.seed + index, extra=dict(base.extra))
        self._waiting.append(_Row(request=request, index=index, params=run))
        if self._work is not None:
            self._work.set()

    def _step_callback(self, row: _Row):
        def on_step(step: int, total: int) -> None:
            if row.request.cancel.is_set():
                raise ImageGenerationCancelled()
            self._update_activity(
                row.request.activity_id,
                completed_items=row.index * row.params.steps + step,
            )

        return on_step

    def _fail_request(self, request: _Request, exc: BaseException) -> None:
        if request.future.done():
            return
        self._waiting[:] = [row for row in self._waiting if row.request is not request]
        self._requests.pop(request.request_id, None)
        request.future.set_exception(exc)

    def _cancel_request(self, request: _Request) -> None:
        self._fail_request(request, ImageGenerationCancelled())

    def _finish_row(self, row: _Row, result: Dict[str, Any]) -> None:
        request = row.request
        if request.future.done():
            return
        request.results.append(result)
        request.next_index += 1
        if request.next_index >= request.n or request.cancel.is_set():
            self._requests.pop(request.request_id, None)
            if request.cancel.is_set():
                request.future.set_exception(ImageGenerationCancelled())
            else:
                request.future.set_result(list(request.results))
            return
        self._enqueue_row(request, request.next_index)

    async def _run_batch(
        self, rows: list[_Row], plan: MemoryPlan, limit: Optional[int]
    ) -> None:
        model = self._model
        adapter = self._adapter
        if model is None or adapter is None:
            raise RuntimeError("Image engine was stopped")
        loop = asyncio.get_running_loop()
        callbacks = [self._step_callback(row) for row in rows]
        owner = id(self)

        def _run_sync():
            if limit:
                set_chunk_memory_limit(owner, limit)
            try:
                t0 = time.monotonic()
                images = adapter.generate_batch(
                    model, [row.params for row in rows], callbacks, plan
                )
                elapsed = time.monotonic() - t0
                return [
                    None
                    if image is None
                    else (
                        _encode_image(image, row.request.output_format),
                        row.params.seed,
                    )
                    for row, image in zip(rows, images)
                ], elapsed
            finally:
                if limit:
                    set_chunk_memory_limit(owner, None)
                mx.synchronize()
                mx.clear_cache()

        try:
            results, elapsed = await loop.run_in_executor(get_mlx_executor(), _run_sync)
        except Exception as exc:
            oom = isinstance(exc, OutOfMemoryError) or is_metal_out_of_memory(exc)
            if oom and len(rows) > 1:
                for row in rows:
                    row.force_single = True
                self._waiting[:0] = rows
                return
            failure = (
                exc
                if isinstance(exc, OutOfMemoryError) or not oom
                else OutOfMemoryError(str(exc))
            )
            for row in rows:
                if row.request.cancel.is_set():
                    self._fail_request(row.request, ImageGenerationCancelled())
                else:
                    self._fail_request(row.request, failure)
            return
        if len(results) < len(rows):
            results += [None] * (len(rows) - len(results))
        self._batch_count += 1
        self._row_count += len(rows)
        self._last_batch_rows = len(rows)
        self._max_batch_rows = max(self._max_batch_rows, len(rows))
        self._last_plan_name = plan.name
        per_row = elapsed / max(1, len(rows))
        for row, result in zip(rows, results):
            if result is None:
                self._fail_request(row.request, ImageGenerationCancelled())
                continue
            data, seed = result
            self._images_generated += 1
            self._total_steps += row.params.steps
            self._total_step_seconds += per_row
            self._finish_row(
                row, {"data": data, "seed": seed, "seconds": round(per_row, 3)}
            )

    async def _dispatch_loop(self) -> None:
        while not self._stopping:
            if not self._waiting:
                if self._work is None:
                    self._work = asyncio.Event()
                self._work.clear()
                if self._waiting:
                    continue
                await self._work.wait()
                continue
            head = self._waiting[0]
            if head.request.cancel.is_set():
                self._cancel_request(head.request)
                continue
            group = [head]
            if self._can_batch(head.params) and not head.force_single:
                if len(self._waiting) == 1 and self._BATCH_WINDOW_S:
                    await asyncio.sleep(self._BATCH_WINDOW_S)
                    if self._stopping or not any(row is head for row in self._waiting):
                        continue
                    if head.request.cancel.is_set():
                        self._cancel_request(head.request)
                        continue
                key = self._batch_key(head.params)
                for row in self._waiting[1:]:
                    if row.request.cancel.is_set():
                        self._cancel_request(row.request)
                        continue
                    if (
                        not row.force_single
                        and self._can_batch(row.params)
                        and self._batch_key(row.params) == key
                    ):
                        group.append(row)
            try:
                size, plan, limit = self._choose_batch(head, len(group))
            except OutOfMemoryError as exc:
                self._fail_request(head.request, exc)
                continue
            rows = group[:size]
            for row in rows:
                self._waiting.remove(row)
            rows = [row for row in rows if not row.request.cancel.is_set()]
            if not rows:
                continue
            if len(rows) > 1:
                logger.info("Image batch: %d requests (%s)", len(rows), plan.name)
            try:
                await self._run_batch(rows, plan, limit)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Image batch failed")
                for row in rows:
                    self._fail_request(row.request, exc)
        for row in list(self._waiting):
            self._cancel_request(row.request)
        self._waiting.clear()

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

        waiting = len(self._waiting)
        if waiting >= self._MAX_WAITING_REQUESTS:
            raise SchedulerQueueFullError(waiting, self._MAX_WAITING_REQUESTS)

        total_steps = params.steps * n
        activity_id = self._begin_activity(
            "image",
            detail=None,
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
        request = _Request(
            request_id=key,
            activity_id=activity_id,
            cancel=cancel,
            n=n,
            output_format=fmt,
            base_params=params,
            next_index=0,
            future=loop.create_future(),
        )
        self._requests[key] = request
        self._ensure_dispatcher()
        self._enqueue_row(request, 0)
        try:
            return await request.future
        except asyncio.CancelledError:
            cancel.set()
            if not request.future.done():
                request.future.cancel()
            raise
        finally:
            self._cancel_events.pop(key, None)
            self._requests.pop(key, None)
            await self._finish_activity(activity_id)

    def cancel(self, request_id: str) -> bool:
        ev = self._cancel_events.get(request_id)
        if ev is None:
            return False
        ev.set()
        return True
