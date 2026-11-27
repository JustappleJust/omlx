# SPDX-License-Identifier: Apache-2.0
"""Image-model accelerations: A8 routing and cross-request caches."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.image_gen._mflux import BoundedCache, image_digest
from omlx.image_gen.accel import a8 as a8mod
from omlx.image_gen.accel import a8_available, disable_a8, enable_a8


def _qlinear(k=256, n=256, bits=8, bias=True):
    lin = nn.Linear(k, n, bias=bias)
    lin.set_dtype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)
    mx.eval(q.parameters())
    return q


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = _qlinear(bias=False)
        self.o = _qlinear(bias=True)
        self.dense = nn.Linear(256, 256)  # not quantized: never touched

    def __call__(self, x):
        return self.o(self.q(x)) + self.dense(x)


def test_enable_is_noop_without_kernels(monkeypatch):
    monkeypatch.setattr(a8mod, "a8_available", lambda: False)
    block = _Block()
    assert enable_a8(block) == {"routed": 0, "skipped": 0, "slower": 0, "available": False}
    assert type(block.q) is nn.QuantizedLinear


def test_classification_and_swap_with_fake_plan(monkeypatch):
    monkeypatch.setattr(a8mod, "a8_available", lambda: True)
    monkeypatch.setattr(a8mod, "_classify", lambda lin: object())
    block = _Block()
    result = enable_a8(block, tune_rows=None)
    assert result["routed"] == 2 and result["available"]
    assert isinstance(block.q, a8mod.A8QuantizedLinear)
    assert type(block.dense) is nn.Linear
    # parameters are shared, not copied
    assert "weight" in block.q and block.q["weight"].dtype == mx.uint32
    assert disable_a8(block) == 2
    assert type(block.o) is nn.QuantizedLinear


def test_small_batches_fall_back_to_stock(monkeypatch):
    monkeypatch.setattr(a8mod, "a8_available", lambda: True)
    monkeypatch.setattr(a8mod, "_classify", lambda lin: object())
    q = _qlinear()
    x = mx.random.normal((1, 8, 256)).astype(mx.bfloat16)
    ref = q(x)
    enable_a8(q, tune_rows=None)  # fake plan would crash if used
    assert mx.allclose(q(x), ref).item()  # 8 rows < min_rows -> stock path


def test_unsupported_layouts_are_not_classified():
    assert a8mod._classify(_qlinear(bits=3)) is None
    lin = nn.Linear(256, 256)
    q = nn.QuantizedLinear.from_linear(lin, group_size=32, bits=8)
    assert a8mod._classify(q) is None


@pytest.mark.skipif(not a8_available(), reason="INT8 NAX kernels unavailable")
@pytest.mark.parametrize("bits,bias", [(8, True), (8, False), (4, True)])
def test_real_kernel_matches_stock(bits, bias):
    q = _qlinear(k=1024, n=1024, bits=bits, bias=bias)
    x = (mx.random.normal((1, 512, 1024)) * 0.5).astype(mx.bfloat16)
    ref = q(x).astype(mx.float32)
    enable_a8(q, tune_rows=None)
    assert isinstance(q, a8mod.A8QuantizedLinear)
    out = q(x)
    assert out.shape == ref.shape and out.dtype == mx.bfloat16
    a, b = ref.reshape(-1), out.astype(mx.float32).reshape(-1)
    cos = ((a * b).sum() / mx.sqrt((a * a).sum() * (b * b).sum())).item()
    assert cos > 0.999


def test_bounded_cache_lru_and_bytes():
    c = BoundedCache(maxsize=3, max_bytes=10_000)
    for i in range(4):
        c[i] = mx.zeros((100,), dtype=mx.float32)  # 400 B each
    assert list(c) == [1, 2, 3]
    _ = c[1]  # refresh
    c[4] = mx.zeros((100,))
    assert list(c) == [3, 1, 4]
    c["big"] = mx.zeros((3000,), dtype=mx.float32)  # 12 KB > cap: evicts others
    assert list(c) == ["big"]
    assert c.nbytes == 12_000
    assert ("big" in c) and ("nope" not in c)
    assert c.stats()["hits"] >= 1 and c.stats()["misses"] >= 1
    c.clear()
    assert c.nbytes == 0 and len(c) == 0


def test_image_digest_is_content_based():
    from PIL import Image

    a = Image.new("RGB", (8, 8), (1, 2, 3))
    b = Image.new("RGB", (8, 8), (1, 2, 3))
    c = Image.new("RGB", (8, 8), (1, 2, 4))
    assert image_digest(a) == image_digest(b) != image_digest(c)


def test_gelu_approx_keeps_each_width_after_a_compiled_trace():
    from omlx.patches.mlx_gelu_approx import apply_gelu_approx_patch

    assert apply_gelu_approx_patch() is False

    def fixed_width():
        return nn.gelu_approx(mx.ones((1, 1, 6144), dtype=mx.bfloat16))

    mx.eval(mx.compile(fixed_width)())
    out = nn.gelu_approx(mx.ones((1, 10, 4096), dtype=mx.bfloat16))
    assert out.shape == (1, 10, 4096)
    value = nn.gelu_approx(mx.array([1.0], dtype=mx.float32))
    assert float(value[0]) == pytest.approx(0.84119, abs=1e-5)


def test_tiling_context_does_not_subscribe_callbacks():
    from omlx.image_gen._mflux import tiling_context
    from omlx.image_gen.base import MemoryPlan

    class Model:
        def __init__(self):
            self.callbacks = type("Callbacks", (), {"in_loop": []})()
            self.tiling_config = "before"

    model = Model()
    with tiling_context(model, MemoryPlan("tiled", 1, {"tile": 256})):
        assert model.tiling_config is not None
        assert model.callbacks.in_loop == []
    assert model.tiling_config == "before"
