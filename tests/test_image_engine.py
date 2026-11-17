# SPDX-License-Identifier: Apache-2.0
"""ImageGenEngine with a fake adapter (no mflux / weights needed)."""

import asyncio
import io
from pathlib import Path

import pytest
from PIL import Image

from omlx.engine.image_gen import ImageGenEngine
from omlx.exceptions import OutOfMemoryError
from omlx.image_gen.base import ImageDefaults, ImageModelAdapter, MemoryPlan

GB = 1024**3


class FakeAdapter(ImageModelAdapter):
    name = "fake"
    pipeline_classes = ("FakePipeline",)
    defaults = ImageDefaults(steps=4, guidance=1.0, width=512, height=512)

    def __init__(self, peak=GB, tiled_peak=GB // 4):
        self.loaded_with = None
        self.released = False
        self.calls = []
        self.peak = peak
        self.tiled_peak = tiled_peak

    def component_files(self, model_dir):
        return {}

    def memory_plans(self, width, height, n_refs=0):
        return [
            MemoryPlan("full", self.peak, {"tile": None}),
            MemoryPlan("tiled", self.tiled_peak, {"tile": 256}),
        ]

    def load(self, model_dir, quantize, options=None):
        self.loaded_with = (Path(model_dir), quantize)
        self.options = options
        return object()

    def generate(self, model, params, on_step, plan=None):
        self.calls.append((params, plan.name if plan else None))
        for i in range(params.steps):
            on_step(i + 1, params.steps)
        return Image.new("RGB", (params.width, params.height), (255, 0, 0))

    def release(self, model):
        self.released = True


@pytest.fixture
def engine():
    return ImageGenEngine("/models/fake", quantize=8, adapter=FakeAdapter())


async def test_start_generate_stop(engine):
    await engine.start()
    assert engine.adapter.loaded_with == (Path("/models/fake"), 8)
    params = engine.build_params("a fox", seed=7)
    assert (params.width, params.height, params.steps) == (512, 512, 4)
    out = await engine.generate(params, n=2)
    assert [r["seed"] for r in out] == [7, 8]
    img = Image.open(io.BytesIO(out[0]["data"]))
    assert img.format == "PNG" and img.size == (512, 512)
    assert engine.get_stats()["images_generated"] == 2
    assert not engine.has_active_requests()
    await engine.stop()
    assert engine.adapter.released
    assert not engine.is_loaded


async def test_quantize_zero_means_bf16():
    e = ImageGenEngine("/m", quantize=0, adapter=FakeAdapter())
    assert e.quantize is None
    with pytest.raises(ValueError):
        ImageGenEngine("/m", quantize=7, adapter=FakeAdapter())


async def test_build_params_validation(engine):
    await engine.start()
    with pytest.raises(ValueError):
        engine.build_params("x", steps=1000)
    p = engine.build_params("x", width=5000, height=100)
    assert (p.width, p.height) == (2048, 256)


async def test_activity_progress_reported(engine):
    await engine.start()
    seen = []
    orig = engine._update_activity

    def spy(aid, **kw):
        seen.append(kw.get("completed_items"))
        orig(aid, **kw)

    engine._update_activity = spy
    await engine.generate(engine.build_params("x", seed=1), n=2)
    assert seen == [1, 2, 3, 4, 5, 6, 7, 8]


async def test_soft_limit_switches_to_tiled_then_refuses(monkeypatch):
    import omlx.engine.image_gen as mod

    monkeypatch.setattr(mod.mx, "get_active_memory", lambda: 10 * GB)
    adapter = FakeAdapter(peak=4 * GB, tiled_peak=GB)
    e = ImageGenEngine("/m", adapter=adapter)
    await e.start()
    e.set_memory_soft_limit(12 * GB)  # 2 GB budget -> tiled fits
    await e.generate(e.build_params("x", seed=1))
    assert adapter.calls[-1][1] == "tiled"

    e.set_memory_soft_limit(10 * GB + GB // 2)  # 0.5 GB budget -> refuse
    with pytest.raises(OutOfMemoryError):
        await e.generate(e.build_params("x", seed=1))
    assert not e.has_active_requests()

    e.set_memory_soft_limit(0)  # unknown limit -> no planning
    await e.generate(e.build_params("x", seed=1))
    assert adapter.calls[-1][1] == "full"


async def test_generations_are_serialised(engine):
    await engine.start()
    running = 0
    peak = 0
    adapter = engine.adapter
    orig = adapter.generate

    def slow(model, params, on_step, plan=None):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        import time

        time.sleep(0.05)
        running -= 1
        return orig(model, params, on_step, plan)

    adapter.generate = slow
    p = engine.build_params("x", seed=1)
    await asyncio.gather(*(engine.generate(p) for _ in range(3)))
    assert peak == 1


async def test_cancel_aborts(engine):
    await engine.start()
    adapter = engine.adapter
    from omlx.image_gen.base import ImageGenerationCancelled

    def gen(model, params, on_step, plan=None):
        engine.cancel("rid")
        on_step(1, params.steps)
        raise AssertionError("should have been cancelled")

    adapter.generate = gen
    with pytest.raises(ImageGenerationCancelled):
        await engine.generate(engine.build_params("x", seed=1), request_id="rid")
    assert not engine.has_active_requests()


async def test_jpeg_output(engine):
    await engine.start()
    out = await engine.generate(engine.build_params("x", seed=1), output_format="jpg")
    assert Image.open(io.BytesIO(out[0]["data"])).format == "JPEG"


async def test_edit_params_validated_against_capabilities():
    from omlx.image_gen.base import ImageCapabilities

    class Editor(FakeAdapter):
        capabilities = ImageCapabilities(edit=True, max_references=2, mask=False, lossy_accel=("step_cache",))

    plain = ImageGenEngine("/m", adapter=FakeAdapter())
    editor = ImageGenEngine("/m", adapter=Editor())
    await plain.start()
    await editor.start()
    ref = Image.new("RGB", (640, 480))
    with pytest.raises(ValueError, match="editing"):
        plain.build_params("x", references=[ref])
    with pytest.raises(ValueError, match="at most 2"):
        editor.build_params("x", references=[ref] * 3)
    with pytest.raises(ValueError, match="mask"):
        editor.build_params("x", references=[ref], mask=ref)
    with pytest.raises(ValueError, match="not available"):
        plain.build_params("x", accel={"step_cache": True})
    p = editor.build_params("x", references=[ref], accel={"step_cache": True}, seed=1)
    assert (p.width, p.height) == (640, 480)  # canvas follows the reference
    assert p.accel == {"step_cache": True}
    assert p.extra["explicit_size"] is False
    out = await editor.generate(p, n=2)
    assert len(out) == 2 and editor.adapter.calls[-1][0].references == [ref]


async def test_engine_default_step_cache_applies():
    from omlx.image_gen.base import ImageCapabilities

    class Editor(FakeAdapter):
        capabilities = ImageCapabilities(edit=True, max_references=1, lossy_accel=("step_cache",))

    e = ImageGenEngine("/m", adapter=Editor(), accel={"step_cache": True})
    await e.start()
    assert e.adapter.options == {"a8": True, "step_cache": True}
    assert e.build_params("x", seed=1).accel == {"step_cache": True}
    assert e.build_params("x", seed=1, accel={"step_cache": False}).accel == {}


def test_rgba_kept_for_png_dropped_for_jpeg():
    from omlx.engine.image_gen import _encode_image

    img = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
    assert Image.open(io.BytesIO(_encode_image(img, "png"))).mode == "RGBA"
    assert Image.open(io.BytesIO(_encode_image(img, "jpeg"))).mode == "RGB"
    opaque = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    assert Image.open(io.BytesIO(_encode_image(opaque, "png"))).mode == "RGB"
