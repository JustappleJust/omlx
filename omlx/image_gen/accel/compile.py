# SPDX-License-Identifier: Apache-2.0
"""Per-block ``mx.compile`` for diffusion transformers that mflux runs eagerly.

Some mflux pipelines cannot compile their whole denoise step (e.g. FLUX.3
builds RoPE tables on the host in float64), so every block runs as dozens of
separate small kernels: AdaLN modulation, float32 RMSNorm, the RoPE
stack/reshape, gating and residual adds. Compiling each block's forward
fuses those elementwise chains into a few kernels while leaving the host-side
parts of the step untouched. Results are bit-for-bit those of the same ops
fused, i.e. exact up to floating-point reassociation inside fused kernels.

Blocks are swapped to a per-class compiled subclass on the instance only, so
other models in the process are unaffected.
"""

from __future__ import annotations

import logging
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_COMPILED_ATTR = "_omlx_compiled_call"
_SUBCLASS_CACHE: dict[type, type] = {}


def _compiled_subclass(cls: type) -> type:
    sub = _SUBCLASS_CACHE.get(cls)
    if sub is not None:
        return sub

    def __call__(self, *args):  # noqa: N807
        fn = self.__dict__.get(_COMPILED_ATTR)
        if fn is None:
            return cls.__call__(self, *args)
        return fn(*args)

    sub = type(f"Compiled{cls.__name__}", (cls,), {"__call__": __call__})
    _SUBCLASS_CACHE[cls] = sub
    return sub


def compile_blocks(blocks: Iterable[nn.Module]) -> int:
    """Compile each block's ``__call__`` (array-only positional arguments).

    The compiled function closes over the block's (frozen, inference-only)
    parameters. Returns the number of blocks compiled.
    """
    n = 0
    for block in blocks:
        cls = type(block)
        if _COMPILED_ATTR in block.__dict__:
            continue
        original = cls.__call__

        def fn(*args, _block=block, _call=original):
            return _call(_block, *args)

        block.__dict__[_COMPILED_ATTR] = mx.compile(fn)
        block.__class__ = _compiled_subclass(cls)
        n += 1
    return n


def uncompile_blocks(blocks: Iterable[nn.Module]) -> int:
    n = 0
    for block in blocks:
        if block.__dict__.pop(_COMPILED_ATTR, None) is not None:
            block.__class__ = type(block).__mro__[1]
            n += 1
    return n
