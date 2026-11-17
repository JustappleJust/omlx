# SPDX-License-Identifier: Apache-2.0
"""W8A8 / W4A8 for diffusion transformers on M5 tensor units (NAX).

Diffusion transformers run every projection on thousands of image tokens per
step: the GEMM shapes are those of LLM prefill. oMLX already ships an
INT8-activation GEMM for prefill (``omlx.custom_kernels.quant_matmul``
``oq_a8_*``, the same idea as vpipe's ``i8_gemm``). This module routes a
loaded DiT's affine-quantized projections through it:

* activations are quantized to int8 on the fly (Stage A), the weights stay in
  their stored 8/4/5-bit affine form, and the GEMM runs on the int8 tensor
  units;
* projections with a bias run the GEMM and add the bias afterwards (the
  kernel itself has no bias input);
* calls with too few rows, unsupported dtypes, or shapes the kernel does not
  tile fall back to the stock ``nn.QuantizedLinear`` path.

Only explicitly enabled module trees are touched (instance class swap, no
global patching), so LLM engines in the same process are unaffected.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_PLAN_ATTR = "_omlx_img_a8_plan"
_MIN_ROWS_ATTR = "_omlx_img_a8_min_rows"
_MIN_ROWS = 128
_GROUP_SIZE = 64
_SUPPORTED_BITS = (4, 5, 8)


def a8_available() -> bool:
    """True when the INT8 NAX GEMM can run here (M5 + built custom kernels)."""
    try:
        from omlx.custom_kernels.quant_matmul import fast
    except ImportError:
        return False
    try:
        return bool(fast.oq_a8_available())
    except Exception:
        return False


def _classify(linear: nn.QuantizedLinear):
    """Kernel dispatch plan for ``linear`` or None (bias allowed)."""
    from omlx.patches.qwen35_oq_a8 import _plan_for

    if getattr(linear, "mode", "affine") != "affine":
        return None
    bits = getattr(linear, "bits", None)
    if bits not in _SUPPORTED_BITS or getattr(linear, "group_size", None) != _GROUP_SIZE:
        return None
    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    biases = getattr(linear, "biases", None)
    if weight is None or scales is None or biases is None:
        return None
    if weight.dtype != mx.uint32 or weight.ndim != 2:
        return None
    if scales.dtype not in (mx.float16, mx.bfloat16) or biases.dtype != scales.dtype:
        return None
    if scales.ndim != 2 or scales.shape != biases.shape or scales.shape[0] != weight.shape[0]:
        return None
    input_dim = scales.shape[1] * _GROUP_SIZE
    if weight.shape[1] * 32 != input_dim * int(bits):
        return None
    return _plan_for(int(bits), weight.shape[0])


class A8QuantizedLinear(nn.QuantizedLinear):
    """``nn.QuantizedLinear`` whose large-batch calls use the A8 kernel.

    Instances are created by swapping ``__class__`` on already-loaded
    ``nn.QuantizedLinear`` modules, so parameters are shared, not copied.
    """

    def __call__(self, x: mx.array) -> mx.array:
        plan = self.__dict__.get(_PLAN_ATTR)
        if (
            plan is None
            or x.ndim < 2
            or x.dtype not in (mx.float16, mx.bfloat16)
            or x.shape[-1] % _GROUP_SIZE
            or _rows(x) < self.__dict__.get(_MIN_ROWS_ATTR, _MIN_ROWS)
        ):
            return super().__call__(x)
        from omlx.patches.qwen35_oq_a8 import apply_plan, stage_a

        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        y = apply_plan(self, stage_a(x2, plan.act_mode, plan.layout), plan)
        if "bias" in self:
            y = y + self["bias"]
        y = y.reshape(*lead, y.shape[-1])
        return y.astype(x.dtype) if y.dtype != x.dtype else y


def _rows(x: mx.array) -> int:
    n = 1
    for d in x.shape[:-1]:
        n *= d
    return n


# (bits, K, N, rows) -> bool: whether A8 beat the stock kernel on this shape.
_TUNE_CACHE: dict[tuple[int, int, int, int], bool] = {}


def _time_ms(fn, x, iters: int = 5) -> float:
    import time

    for _ in range(2):
        mx.eval(fn(x))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(x))
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def _a8_wins(linear: nn.QuantizedLinear, plan, rows: int) -> bool:
    """Time stock vs A8 once per (bits, K, N) on a representative batch.

    The INT8 GEMM wins clearly on Q8 weights, but for some Q4 shapes (narrow
    N, long K) the stock dequantizing kernel is faster; measure instead of
    guessing. Runs on the MLX executor during load.
    """
    in_dim = linear.scales.shape[1] * _GROUP_SIZE
    out_dim = linear.weight.shape[0]
    key = (int(linear.bits), in_dim, out_dim, rows)
    cached = _TUNE_CACHE.get(key)
    if cached is not None:
        return cached
    from omlx.patches.qwen35_oq_a8 import apply_plan, stage_a

    x = (mx.random.normal((rows, in_dim)) * 0.5).astype(linear.scales.dtype)
    try:
        stock = _time_ms(lambda v: nn.QuantizedLinear.__call__(linear, v), x)
        a8 = _time_ms(lambda v: apply_plan(linear, stage_a(v, plan.act_mode, plan.layout), plan), x)
    except Exception as exc:  # pragma: no cover - kernel refused the shape
        logger.debug("A8 tune failed for %s: %s", key, exc)
        _TUNE_CACHE[key] = False
        return False
    win = a8 < stock * 0.97
    _TUNE_CACHE[key] = win
    logger.debug("A8 tune %s: stock %.2fms a8 %.2fms -> %s", key, stock, a8, win)
    return win


def enable_a8(
    module: nn.Module, *, min_rows: int = _MIN_ROWS, tune_rows: int | None = 4096
) -> dict:
    """Route every eligible quantized projection under ``module`` through A8.

    ``tune_rows``: when set, each distinct (bits, K, N) shape is timed once at
    that batch size and only routed if A8 is faster. None routes everything
    eligible.

    Returns a summary ``{"routed": n, "skipped": m, "slower": k, "available": bool}``.
    """
    if not a8_available():
        return {"routed": 0, "skipped": 0, "slower": 0, "available": False}
    routed = skipped = slower = 0
    for _, child in module.named_modules():
        if type(child) is not nn.QuantizedLinear:
            continue
        plan = _classify(child)
        if plan is None:
            skipped += 1
            continue
        if tune_rows and not _a8_wins(child, plan, tune_rows):
            slower += 1
            continue
        child.__dict__[_PLAN_ATTR] = plan
        child.__dict__[_MIN_ROWS_ATTR] = int(min_rows)
        child.__class__ = A8QuantizedLinear
        routed += 1
    return {"routed": routed, "skipped": skipped, "slower": slower, "available": True}


def disable_a8(module: nn.Module) -> int:
    """Undo :func:`enable_a8` (used by benchmarks and tests)."""
    n = 0
    for _, child in module.named_modules():
        if isinstance(child, A8QuantizedLinear):
            child.__class__ = nn.QuantizedLinear
            child.__dict__.pop(_PLAN_ATTR, None)
            child.__dict__.pop(_MIN_ROWS_ATTR, None)
            n += 1
    return n


def a8_status(module: Any) -> dict:
    routed = sum(1 for _, c in module.named_modules() if isinstance(c, A8QuantizedLinear))
    return {"routed": routed}
