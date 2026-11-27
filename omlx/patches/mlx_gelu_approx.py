# SPDX-License-Identifier: Apache-2.0
"""Uncompiled ``mlx.nn.gelu_approx`` for MLX 0.32.x.

MLX ships ``gelu_approx`` wrapped in ``mx.compile(shapeless=True)``. In 0.32.5
a trace inside another compiled graph whose GELU input has a fixed width caches
that width in the shared wrapper, so a later eager call with a different width
returns the cached one. Krea-2's compiled predict traces 6144-wide GELUs; a
following 4096-wide call (Qwen-Image's text projection) then comes back 6144
wide. Keeping the same math without the compiled wrapper removes the cache.
"""

import math

import mlx.core as mx
import mlx.nn as nn


def _gelu_approx(x: mx.array) -> mx.array:
    t = mx.clip(x, -10, 10)
    u = 1 + mx.tanh(math.sqrt(2 / math.pi) * (t + 0.044715 * t**3))
    v = mx.stop_gradient(u)
    return 0.5 * x * v + 0.5 * t * (u - v)


_gelu_approx._omlx_uncompiled = True


def apply_gelu_approx_patch() -> bool:
    """Replace ``nn.gelu_approx`` with an uncompiled equivalent once."""
    if getattr(nn.gelu_approx, "_omlx_uncompiled", False):
        return False
    nn.gelu_approx = _gelu_approx
    return True
