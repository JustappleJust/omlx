# SPDX-License-Identifier: Apache-2.0
"""ImageGenerationEngine tests against a fake mflux backend (no MLX needed)."""

import asyncio

import pytest

pytest.importorskip("mlx")

from PIL import Image  # noqa: E402

from omlx.engine.image import (  # noqa: E402
    ImageGenerationEngine,
    ImageRequest,
    MfluxBackend,
    resolve_edit_dimensions,
)
from omlx.image_models import (  # noqa: E402
    IMAGE_MODELS,
    ImageBackendUnavailable,
    ImageGenerationError,
    ImageModelUnsupported,
    ImageRequestError,
)


class _Generated:
    def __init__(self, image):
        self.image = image


class _FakePipeline:
    def __init__(self, spec, kind):
        self.spec = spec
        self.kind = kind
        self.calls = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        width = kwargs.get("width")
        height = kwargs.get("height")
        if width is None or height is None:
            reference = kwargs.get("image_paths") or kwargs.get("image_path")
            if isinstance(reference, list):
                reference = reference[0]
            with Image.open(reference) as source:
                width, height = source.size
        mode = "RGBA" if self.spec.outputs_alpha else "RGB"
        return _Generated(Image.new(mode, (int(width), int(height)), (10, 20, 30, 255)))

    failure = None


class _FakeBackend:
    name = "fake-mflux"

    def __init__(self):
        self.pipelines = []

    def pipeline(self, spec, kind, *, model_path=None, quantize=None):
        pipeline = _FakePipeline(spec, kind)
        pipeline.model_path = model_path
        pipeline.quantize = quantize
        self.pipelines.append(pipeline)
        return pipeline

    def last(self):
        return self.pipelines[-1]


@pytest.fixture
def weights_dir(tmp_path):
    model = tmp_path / "Krea-2-Turbo-mflux-q8"
    transformer = model / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "0.safetensors").write_bytes(b"0" * 16)
    return model


def _engine(tmp_path, key, *, name=None, backend=None, quantize=None):
    model = tmp_path / (name or f"{key}-mflux-q8")
    model.mkdir(exist_ok=True)
    (model / "mflux_model.json").write_text("{}")
    (model / "transformer").mkdir(exist_ok=True)
    (model / "transformer" / "0.safetensors").write_bytes(b"0" * 16)
    backend = backend or _FakeBackend()
    engine = ImageGenerationEngine(
        str(model),
        spec=IMAGE_MODELS[key],
        quantize=quantize,
        backend=backend,
    )
    return engine, backend


def _png(tmp_path, name="ref.png", size=(96, 64)):
    path = tmp_path / name
    Image.new("RGB", size, (200, 100, 50)).save(path)
    return path


def test_generate_uses_krea_vendor_defaults(tmp_path):
    async def run():
        engine, backend = _engine(tmp_path, "krea-2-turbo")
        result = await engine.generate(ImageRequest(prompt="a red fox", seed=7))
        return engine, backend, result

    engine, backend, result = asyncio.run(run())

    call = backend.last().calls[0]
    assert call["num_inference_steps"] == 8
    assert call["guidance"] == 1.0
    assert (call["width"], call["height"]) == (1024, 1024)
    assert call["seed"] == 7
    assert result.images[0].startswith(b"\x89PNG")
    assert (result.width, result.height) == (1024, 1024)
    assert result.mode == "generate"
    assert engine.get_stats()["images_generated"] == 1


def test_generate_honours_per_request_overrides(tmp_path):
    async def run():
        engine, backend = _engine(tmp_path, "krea-2-turbo")
        await engine.generate(
            ImageRequest(
                prompt="wide shot",
                width=768,
                height=512,
                steps=4,
                guidance=2.5,
            )
        )
        return backend

    backend = asyncio.run(run())
    call = backend.last().calls[0]
    assert (call["width"], call["height"]) == (768, 512)
    assert call["num_inference_steps"] == 4
    assert call["guidance"] == 2.5


def test_flux3_generate_omits_guidance(tmp_path):
    async def run():
        engine, backend = _engine(
            tmp_path, "flux-3-image-klein", name="flux-3-image-klein-mflux-bf16"
        )
        result = await engine.generate(ImageRequest(prompt="a lighthouse"))
        return backend, result

    backend, result = asyncio.run(run())
    call = backend.last().calls[0]
    assert call["num_inference_steps"] == 16
    assert "guidance" not in call
    assert result.guidance is None


def test_flux3_rejects_guidance(tmp_path):
    async def run():
        engine, _ = _engine(
            tmp_path, "flux-3-image-klein", name="flux-3-image-klein-mflux-bf16"
        )
        with pytest.raises(ImageRequestError):
            await engine.generate(ImageRequest(prompt="x", guidance=3.0))

    asyncio.run(run())


def test_multiple_images_advance_the_seed(tmp_path):
    async def run():
        engine, backend = _engine(tmp_path, "krea-2-turbo")
        return await engine.generate(ImageRequest(prompt="cat", n=2, seed=100)), backend

    result, backend = asyncio.run(run())
    assert [c["seed"] for c in backend.last().calls] == [100, 101]
    assert len(result.images) == 2


def test_jpeg_output_is_encoded(tmp_path):
    async def run():
        engine, _ = _engine(tmp_path, "krea-2-turbo")
        return await engine.generate(
            ImageRequest(prompt="cat", output_format="jpeg")
        )

    result = asyncio.run(run())
    assert result.images[0].startswith(b"\xff\xd8")
    assert result.mime_type == "image/jpeg"


def test_output_compression_is_honoured(tmp_path):
    async def run():
        engine, _ = _engine(tmp_path, "krea-2-turbo")
        return await engine.generate(
            ImageRequest(prompt="cat", output_format="jpeg", output_compression=40)
        )

    result = asyncio.run(run())
    with Image.open(__import__("io").BytesIO(result.images[0])) as decoded:
        assert decoded.format == "JPEG"


def test_pipeline_construction_failure_is_a_generation_error(tmp_path):
    async def run():
        engine, backend = _engine(tmp_path, "krea-2-turbo")

        def boom(spec, kind, **kwargs):
            raise ValueError("checkpoint layout not recognised")

        backend.pipeline = boom
        with pytest.raises(ImageGenerationError):
            await engine.generate(ImageRequest(prompt="x"))

    asyncio.run(run())


def test_unsupported_output_format_is_a_client_error(tmp_path):
    async def run():
        engine, _ = _engine(tmp_path, "krea-2-turbo")
        with pytest.raises(ImageRequestError):
            await engine.generate(ImageRequest(prompt="cat", output_format="gif"))

    asyncio.run(run())


def test_edit_swaps_the_resident_pipeline(tmp_path):
    async def run():
        engine, backend = _engine(
            tmp_path, "flux-3-image-klein", name="flux-3-image-klein-mflux-bf16"
        )
        await engine.generate(ImageRequest(prompt="a fox", seed=1))
        await engine.edit(
            ImageRequest(prompt="make it night", seed=2), [_png(tmp_path)]
        )
        return engine, backend

    engine, backend = asyncio.run(run())
    assert [p.kind for p in backend.pipelines] == ["generate", "edit"]
    assert engine.get_stats()["load_count"] == 2
    assert engine.get_stats()["pipeline"] == "edit"


def test_flux3_edit_passes_references_and_derives_size(tmp_path):
    async def run():
        engine, backend = _engine(
            tmp_path, "flux-3-image-klein", name="flux-3-image-klein-mflux-bf16"
        )
        references = [_png(tmp_path, "a.png", (96, 64)), _png(tmp_path, "b.png", (64, 64))]
        result = await engine.edit(
            ImageRequest(prompt="combine these", seed=5), references
        )
        return backend, result

    backend, result = asyncio.run(run())
    call = backend.last().calls[0]
    assert len(call["image_paths"]) == 2
    assert "image_path" not in call
    assert "width" not in call and "height" not in call
    assert call["num_inference_steps"] == 16
    assert (result.width, result.height) == (96, 64)


def test_qwen_edit_forwards_mask_and_alpha(tmp_path):
    async def run():
        engine, backend = _engine(
            tmp_path, "qwen-image-2.1-turbo", name="Qwen-Image-2.1-Turbo-mflux-q8"
        )
        mask = _png(tmp_path, "mask.png", (96, 64))
        result = await engine.edit(
            ImageRequest(prompt="replace the sky", seed=9, output_format="png"),
            [_png(tmp_path)],
            mask,
        )
        return backend, result

    backend, result = asyncio.run(run())
    call = backend.last().calls[0]
    assert call["mask_image"].endswith("mask.png")
    assert call["guidance"] == 1.0
    assert result.images[0].startswith(b"\x89PNG")
    # Qwen-Image-2.1 emits alpha; the fake returns RGBA so PNG keeps 4 channels.
    with Image.open(__import__("io").BytesIO(result.images[0])) as decoded:
        assert decoded.mode == "RGBA"


def test_krea_edit_is_strength_based_img2img(tmp_path):
    async def run():
        engine, backend = _engine(tmp_path, "krea-2-turbo")
        result = await engine.edit(
            ImageRequest(prompt="a red fox", seed=3), [_png(tmp_path, size=(1000, 700))]
        )
        return backend, result

    backend, result = asyncio.run(run())
    call = backend.last().calls[0]
    assert backend.last().kind == "generate"
    assert call["image_path"].endswith("ref.png")
    assert call["image_strength"] == 0.65
    assert (call["width"], call["height"]) == (992, 688)
    assert (result.width, result.height) == (992, 688)


def test_krea_edit_strength_override_and_validation(tmp_path):
    async def run():
        engine, backend = _engine(tmp_path, "krea-2-turbo")
        await engine.edit(
            ImageRequest(prompt="x", strength=0.4), [_png(tmp_path)]
        )
        with pytest.raises(ImageRequestError):
            await engine.edit(
                ImageRequest(prompt="x", strength=0.0), [_png(tmp_path)]
            )
        return backend

    backend = asyncio.run(run())
    assert backend.last().calls[0]["image_strength"] == 0.4


def test_krea_rejects_multi_reference_and_mask(tmp_path):
    async def run():
        engine, _ = _engine(tmp_path, "krea-2-turbo")
        with pytest.raises(ImageRequestError):
            await engine.edit(
                ImageRequest(prompt="x"),
                [_png(tmp_path, "a.png"), _png(tmp_path, "b.png")],
            )
        with pytest.raises(ImageRequestError):
            await engine.edit(
                ImageRequest(prompt="x"), [_png(tmp_path)], _png(tmp_path, "m.png")
            )

    asyncio.run(run())


def test_too_many_references_rejected(tmp_path):
    async def run():
        engine, _ = _engine(
            tmp_path, "flux-3-image-klein", name="flux-3-image-klein-mflux-bf16"
        )
        references = [_png(tmp_path, f"r{i}.png") for i in range(11)]
        with pytest.raises(ImageRequestError):
            await engine.edit(ImageRequest(prompt="x"), references)

    asyncio.run(run())


def test_reference_edit_rejects_multiple_images_per_request(tmp_path):
    async def run():
        engine, _ = _engine(
            tmp_path, "flux-3-image-klein", name="flux-3-image-klein-mflux-bf16"
        )
        with pytest.raises(ImageRequestError):
            await engine.edit(ImageRequest(prompt="x", n=2), [_png(tmp_path)])

    asyncio.run(run())


def test_negative_seed_rejected(tmp_path):
    async def run():
        engine, _ = _engine(tmp_path, "krea-2-turbo")
        with pytest.raises(ImageRequestError):
            await engine.generate(ImageRequest(prompt="x", seed=-1))

    asyncio.run(run())


def test_qwen_turbo_rejects_guidance_other_than_one(tmp_path):
    async def run():
        engine, _ = _engine(
            tmp_path, "qwen-image-2.1-turbo", name="Qwen-Image-2.1-Turbo-mflux-q8"
        )
        with pytest.raises(ImageRequestError):
            await engine.generate(ImageRequest(prompt="x", guidance=4.0))

    asyncio.run(run())


def test_pipeline_value_error_maps_to_request_error(tmp_path):
    async def run():
        engine, backend = _engine(tmp_path, "krea-2-turbo")
        backend.pipelines = []

        def boom(spec, kind, **kwargs):
            pipeline = _FakePipeline(spec, kind)
            pipeline.failure = ValueError("bad scheduler")
            backend.pipelines.append(pipeline)
            return pipeline

        backend.pipeline = boom
        with pytest.raises(ImageRequestError):
            await engine.generate(ImageRequest(prompt="x"))

    asyncio.run(run())


def test_pipeline_runtime_error_maps_to_generation_error(tmp_path):
    async def run():
        engine, backend = _engine(tmp_path, "krea-2-turbo")

        def boom(spec, kind, **kwargs):
            pipeline = _FakePipeline(spec, kind)
            pipeline.failure = RuntimeError("Metal OOM")
            backend.pipelines.append(pipeline)
            return pipeline

        backend.pipeline = boom
        with pytest.raises(ImageGenerationError):
            await engine.generate(ImageRequest(prompt="x"))

    asyncio.run(run())


def test_start_requires_mflux(tmp_path, monkeypatch):
    async def run():
        model = tmp_path / "Krea-2-Turbo-mflux-q8"
        model.mkdir()
        (model / "mflux_model.json").write_text("{}")
        engine = ImageGenerationEngine(str(model), spec=IMAGE_MODELS["krea-2-turbo"])
        monkeypatch.setattr(MfluxBackend, "available", staticmethod(lambda: False))
        with pytest.raises(ImageBackendUnavailable):
            await engine.start()

    asyncio.run(run())


def test_start_and_stop_manage_the_pipeline(tmp_path):
    async def run():
        engine, backend = _engine(tmp_path, "krea-2-turbo")
        await engine.start()
        assert engine.get_stats()["loaded"] is True
        await engine.start()  # idempotent
        assert engine.get_stats()["load_count"] == 1
        await engine.stop()
        assert engine.get_stats()["loaded"] is False
        assert engine.get_stats()["pipeline"] is None

    asyncio.run(run())


def test_unknown_model_is_rejected(tmp_path):
    model = tmp_path / "llama-3-8b"
    model.mkdir()
    with pytest.raises(ImageModelUnsupported):
        ImageGenerationEngine(str(model), backend=_FakeBackend())


def test_model_path_is_forwarded_to_the_backend(tmp_path):
    async def run():
        engine, backend = _engine(tmp_path, "krea-2-turbo", quantize=8)
        await engine.generate(ImageRequest(prompt="x"))
        return backend

    backend = asyncio.run(run())
    pipeline = backend.last()
    assert str(pipeline.model_path).endswith("krea-2-turbo-mflux-q8")
    assert pipeline.quantize == 8


def test_resolve_edit_dimensions_auto(tmp_path):
    reference = _png(tmp_path, size=(1000, 700))
    assert resolve_edit_dimensions(
        IMAGE_MODELS["flux-3-image-klein"], None, None, reference
    ) == (None, None)
    # Krea 2 snaps the reference to the 16-px grid.
    assert resolve_edit_dimensions(
        IMAGE_MODELS["krea-2-turbo"], None, None, reference
    ) == (992, 688)
    assert resolve_edit_dimensions(
        IMAGE_MODELS["krea-2-turbo"], 768, 512, reference
    ) == (768, 512)
