# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the mflux-backed adapters."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Optional

from .base import MemoryPlan, StepCallback

MFLUX_INSTALL_HINT = (
    "mflux is required for image generation. "
    'Install it with: pip install "omlx[image]"'
)


def _nbytes(value: Any) -> int:
    """Best-effort byte size of nested mx.array containers."""
    if value is None:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if isinstance(value, dict):
        return sum(_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(v) for v in value)
    return 0


class BoundedCache(OrderedDict):
    """LRU dict bounded by entry count and total array bytes.

    Used for mflux's ``prompt_cache`` (text-encoder outputs) and for oMLX's
    cross-request reference-image caches. mflux keeps every entry forever;
    on a long-running server that is an unbounded leak.
    """

    def __init__(self, maxsize: int = 8, max_bytes: int = 1 << 30):
        super().__init__()
        self._maxsize = maxsize
        self._max_bytes = max_bytes
        self._sizes: dict[Any, int] = {}
        self.hits = 0
        self.misses = 0

    @property
    def nbytes(self) -> int:
        return sum(self._sizes.values())

    def __contains__(self, key) -> bool:
        found = super().__contains__(key)
        if found:
            self.hits += 1
        else:
            self.misses += 1
        return found

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._sizes[key] = _nbytes(value)
        self.move_to_end(key)
        while len(self) > 1 and (
            len(self) > self._maxsize or self.nbytes > self._max_bytes
        ):
            old, _ = self.popitem(last=False)
            self._sizes.pop(old, None)

    def __delitem__(self, key):
        super().__delitem__(key)
        self._sizes.pop(key, None)

    def clear(self) -> None:
        super().clear()
        self._sizes.clear()

    def stats(self) -> dict:
        return {
            "entries": len(self),
            "bytes": self.nbytes,
            "hits": self.hits,
            "misses": self.misses,
        }


class StepReporter:
    """mflux InLoopCallback forwarding progress (and cancellation)."""

    def __init__(self, on_step: StepCallback, total: int):
        self._on_step = on_step
        self._total = total
        self._i = 0

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):  # noqa: ARG002
        self._i += 1
        # Exceptions propagate out of mflux's denoise loop and abort it.
        self._on_step(self._i, self._total)


def partition_by(keys: list[Any]) -> dict[Any, list[int]]:
    """Group row indices by a per-row key, preserving input order."""
    groups: dict[Any, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    return groups


@contextlib.contextmanager
def tiling_context(model: Any, plan: Optional[MemoryPlan]) -> Iterator[None]:
    """Install the plan's VAE tiling without subscribing to model callbacks."""
    from mflux.models.common.vae.tiling_config import TilingConfig

    tile = (plan.options.get("tile") if plan else None) or None
    previous_tiling = getattr(model, "tiling_config", None)
    # Overlap stays at mflux's default (8 latent px = 64 image px).
    model.tiling_config = TilingConfig(vae_decode_tile_size=tile) if tile else None
    try:
        yield
    finally:
        model.tiling_config = previous_tiling


@contextlib.contextmanager
def run_context(model: Any, on_step: StepCallback, steps: int, plan: Optional[MemoryPlan]) -> Iterator[None]:
    """Attach the progress callback and the plan's VAE tiling for one run."""
    reporter = StepReporter(on_step, steps)
    registry = model.callbacks
    registry.in_loop.append(reporter)
    try:
        with tiling_context(model, plan):
            yield
    finally:
        try:
            registry.in_loop.remove(reporter)
        except ValueError:
            pass


def safetensors_keys(path: Path) -> list[str]:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            return list(f.keys())
    except Exception:
        return []


def safetensors_metadata(path: Path) -> dict:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            return f.metadata() or {}
    except Exception:
        return {}


def index_or_header_keys(directory: Path, preferred: tuple[str, ...] = ()) -> list[str]:
    """Weight names of a component directory (index first, header fallback)."""
    for index in sorted(directory.glob("*.safetensors.index.json")):
        try:
            weight_map = json.loads(index.read_text()).get("weight_map") or {}
        except (OSError, ValueError, AttributeError):
            continue
        if weight_map:
            return list(weight_map)
    candidates = [directory / n for n in preferred] + sorted(directory.glob("*.safetensors"))
    for path in candidates:
        if path.is_file():
            return safetensors_keys(path)
    return []


def read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def image_digest(image: Any) -> str:
    """Stable content hash of a PIL image (mode + size + pixels)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{image.mode}:{image.size}".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def is_prequantized_dir(directory: Path) -> bool:
    """mflux-save shards carry ``quantization_level`` metadata."""
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        return False
    level = safetensors_metadata(files[0]).get("quantization_level")
    return level not in (None, "", "None")


def apply_load_accel(model: Any, options: Optional[dict]) -> None:
    """Apply near-lossless load-time accelerations to an mflux model.

    * ``a8`` (default on): route the DiT's quantized projections through the
      INT8-activation NAX GEMM (no-op on bf16 weights or without NAX).
    """
    import logging

    import mlx.core as mx

    # Some mflux initializers leave weights lazy (read on first use). Load them
    # now so the first request is not slow and the pool measures the real
    # resident size right after load.
    mx.eval(model.parameters())
    options = options or {}
    status: dict = {"a8": {"enabled": False}}
    transformer = getattr(model, "transformer", None)
    if options.get("a8", True) and transformer is not None:
        from .accel import enable_a8

        result = enable_a8(transformer)
        status["a8"] = {"enabled": result["routed"] > 0, **result}
        logging.getLogger(__name__).info("Image A8 GEMM: %s", result)
    model._omlx_accel_status = status
