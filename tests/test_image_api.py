# SPDX-License-Identifier: Apache-2.0
"""HTTP-level tests for /v1/images/* with a stubbed engine pool."""

import base64
import importlib.util
import io
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from omlx.api import image_routes  # noqa: E402
from omlx.image_models import (  # noqa: E402
    IMAGE_MODELS,
    ImageGenerationError,
    ImageRequestError,
)


def _png_bytes(size=(64, 48), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeEntry:
    def __init__(self, model_id, model_path, model_type="image", engine_type="image"):
        self.id = model_id
        self.model_id = model_id
        self.model_path = str(model_path)
        self.model_type = model_type
        self.engine_type = engine_type


class _FakeEngine:
    is_image_engine = True

    def __init__(self, spec, *, images=None):
        self.spec = spec
        self.images = images if images is not None else [_png_bytes()]
        self.generate_calls = []
        self.edit_calls = []
        self.failure = None

    async def generate(self, request):
        self.generate_calls.append(request)
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(
            images=self.images,
            mime_type="image/png",
            seed=request.seed if request.seed is not None else 1234,
        )

    async def edit(self, request, image_paths, mask_path=None):
        self.edit_calls.append(
            (
                request,
                [str(path) for path in image_paths],
                [path.exists() for path in image_paths],
                None if mask_path is None else str(mask_path),
                None if mask_path is None else mask_path.exists(),
            )
        )
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(
            images=self.images,
            mime_type="image/png",
            seed=request.seed if request.seed is not None else 1234,
        )


class _FakePool:
    def __init__(self, entries, engine, failure=None):
        self.entries = {entry.id: entry for entry in entries}
        self.engine = engine
        self.failure = failure

    def get_entry(self, model_id):
        return self.entries.get(model_id)

    def get_status(self):
        return {
            "models": [
                {
                    "id": entry.id,
                    "model_path": entry.model_path,
                    "model_type": entry.model_type,
                    "engine_type": entry.engine_type,
                }
                for entry in self.entries.values()
            ]
        }

    @asynccontextmanager
    async def acquire(self, model_id):
        if self.failure is not None:
            raise self.failure
        yield self.engine


@pytest.fixture
def image_model_dir(tmp_path):
    model = tmp_path / "Krea-2-Turbo-mflux-q8"
    (model / "transformer").mkdir(parents=True)
    (model / "transformer" / "0.safetensors").write_bytes(b"0" * 16)
    (model / "mflux_model.json").write_text('{"base_model": "krea-2"}')
    return model


@pytest.fixture
def client(monkeypatch, image_model_dir):
    engine = _FakeEngine(IMAGE_MODELS["krea-2-turbo"])
    entry = _FakeEntry("Krea-2-Turbo-mflux-q8", image_model_dir)
    pool = _FakePool([entry], engine)

    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(image_routes, "_resolve_model", lambda model_id: model_id)
    monkeypatch.setattr(image_routes, "_record_request", lambda model_id: None)
    monkeypatch.setattr(image_routes, "_require_mflux", lambda: None)

    app = FastAPI()
    app.include_router(image_routes.router)
    with TestClient(app) as test_client:
        test_client.pool = pool
        test_client.engine = engine
        yield test_client


def test_generations_defaults_to_base64(client):
    response = client.post(
        "/v1/images/generations",
        json={"model": "Krea-2-Turbo-mflux-q8", "prompt": "a red fox"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"][0]["url"] is None
    payload = base64.b64decode(body["data"][0]["b64_json"])
    assert payload.startswith(b"\x89PNG")
    assert body["seed"] == 1234
    assert body["model"] == "Krea-2-Turbo-mflux-q8"

    request = client.engine.generate_calls[0]
    assert request.prompt == "a red fox"
    assert request.width is None and request.height is None


def test_generations_forwards_size_and_sampling_overrides(client):
    response = client.post(
        "/v1/images/generations",
        json={
            "model": "Krea-2-Turbo-mflux-q8",
            "prompt": "wide",
            "size": "768x512",
            "steps": 4,
            "guidance": 2.0,
            "seed": 99,
            "n": 1,
        },
    )

    assert response.status_code == 200
    request = client.engine.generate_calls[0]
    assert (request.width, request.height) == (768, 512)
    assert request.steps == 4
    assert request.guidance == 2.0
    assert request.seed == 99
    assert response.json()["seed"] == 99


@pytest.mark.parametrize(
    "payload",
    [
        {"size": "1024x1024", "width": 512, "height": 512},
        {"width": 512},
        {"height": 512},
        {"size": "huge"},
        {"size": "1024"},
    ],
)
def test_generations_rejects_conflicting_or_invalid_sizes(client, payload):
    response = client.post(
        "/v1/images/generations",
        json={"model": "Krea-2-Turbo-mflux-q8", "prompt": "x", **payload},
    )
    assert response.status_code == 400
    assert client.engine.generate_calls == []


def test_generations_url_response_round_trips(client):
    response = client.post(
        "/v1/images/generations",
        json={
            "model": "Krea-2-Turbo-mflux-q8",
            "prompt": "a fox",
            "response_format": "url",
        },
    )

    assert response.status_code == 200
    url = response.json()["data"][0]["url"]
    assert "/v1/images/generated/" in url

    fetched = client.get(url.split("testserver", 1)[1])
    assert fetched.status_code == 200
    assert fetched.headers["content-type"] == "image/png"
    assert fetched.content.startswith(b"\x89PNG")

    assert client.get("/v1/images/generated/does-not-exist").status_code == 404


def test_generations_rejects_non_image_model(monkeypatch, client, tmp_path):
    llm = tmp_path / "qwen3-4b"
    llm.mkdir()
    client.pool.entries["qwen3-4b"] = _FakeEntry(
        "qwen3-4b", llm, model_type="llm", engine_type="batched"
    )

    response = client.post(
        "/v1/images/generations", json={"model": "qwen3-4b", "prompt": "x"}
    )

    assert response.status_code == 400
    assert "not an image model" in response.json()["detail"]


def test_generations_unknown_model_lists_available(client):
    response = client.post(
        "/v1/images/generations", json={"model": "flux1-dev", "prompt": "x"}
    )
    assert response.status_code == 404
    assert "Krea-2-Turbo-mflux-q8" in response.json()["detail"]


def test_generations_no_models_installed(monkeypatch):
    pool = _FakePool([], _FakeEngine(IMAGE_MODELS["krea-2-turbo"]))
    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(image_routes, "_resolve_model", lambda model_id: model_id)
    monkeypatch.setattr(image_routes, "_require_mflux", lambda: None)
    app = FastAPI()
    app.include_router(image_routes.router)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/images/generations", json={"model": "krea-2-turbo", "prompt": "x"}
        )
    assert response.status_code == 404
    assert "no image models are installed" in response.json()["detail"]


def test_generations_resolves_registry_names(monkeypatch, client):
    response = client.post(
        "/v1/images/generations", json={"model": "krea-2-turbo", "prompt": "x"}
    )
    assert response.status_code == 200
    assert len(client.engine.generate_calls) == 1


def test_generations_ambiguous_registry_name(client, tmp_path):
    second = tmp_path / "Krea-2-Turbo-mflux-bf16"
    second.mkdir()
    (second / "mflux_model.json").write_text('{"base_model": "krea-2"}')
    client.pool.entries[second.name] = _FakeEntry(second.name, second)

    response = client.post(
        "/v1/images/generations", json={"model": "krea-2-turbo", "prompt": "x"}
    )

    assert response.status_code == 400
    assert "matches several" in response.json()["detail"]


def test_generations_requires_mflux(monkeypatch, image_model_dir):
    engine = _FakeEngine(IMAGE_MODELS["krea-2-turbo"])
    pool = _FakePool([_FakeEntry("krea", image_model_dir)], engine)
    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(image_routes, "_resolve_model", lambda model_id: model_id)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "mflux" else real_find_spec(name, *args),
    )
    app = FastAPI()
    app.include_router(image_routes.router)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/images/generations", json={"model": "krea", "prompt": "x"}
        )
    assert response.status_code == 503
    assert "omlx[image]" in response.json()["detail"]
    assert engine.generate_calls == []


def test_generations_maps_request_errors_to_400(client):
    client.engine.failure = ImageRequestError("bad guidance")
    response = client.post(
        "/v1/images/generations", json={"model": "Krea-2-Turbo-mflux-q8", "prompt": "x"}
    )
    assert response.status_code == 400
    assert "bad guidance" in response.json()["detail"]


def test_generations_maps_runtime_errors_to_500(client):
    client.engine.failure = ImageGenerationError("Metal OOM")
    response = client.post(
        "/v1/images/generations", json={"model": "Krea-2-Turbo-mflux-q8", "prompt": "x"}
    )
    assert response.status_code == 500
    assert "Metal OOM" in response.json()["detail"]


def test_generations_forwards_output_compression(client):
    response = client.post(
        "/v1/images/generations",
        json={
            "model": "Krea-2-Turbo-mflux-q8",
            "prompt": "x",
            "output_format": "webp",
            "output_compression": 25,
        },
    )
    assert response.status_code == 200
    assert client.engine.generate_calls[0].output_compression == 25


def test_pool_errors_are_classified(client):
    from omlx.exceptions import ModelLoadingError, ModelNotFoundError

    client.pool.failure = ModelNotFoundError("gone", [])
    missing = client.post(
        "/v1/images/generations", json={"model": "Krea-2-Turbo-mflux-q8", "prompt": "x"}
    )
    assert missing.status_code == 404

    client.pool.failure = ModelLoadingError("Krea-2-Turbo-mflux-q8", "not enough memory")
    unloadable = client.post(
        "/v1/images/generations", json={"model": "Krea-2-Turbo-mflux-q8", "prompt": "x"}
    )
    assert unloadable.status_code == 503


def test_generations_validates_prompt(client):
    response = client.post(
        "/v1/images/generations", json={"model": "Krea-2-Turbo-mflux-q8", "prompt": ""}
    )
    assert response.status_code == 422


def test_transparent_background_requires_alpha_capable_family(client):
    response = client.post(
        "/v1/images/generations",
        json={
            "model": "Krea-2-Turbo-mflux-q8",
            "prompt": "x",
            "background": "transparent",
        },
    )
    assert response.status_code == 400
    assert "transparent" in response.json()["detail"]


def test_edits_accept_multiple_references_and_mask(monkeypatch, client):
    client.engine.spec = IMAGE_MODELS["qwen-image-2.1-turbo"]
    files = [
        ("image", ("a.png", _png_bytes((64, 48)), "image/png")),
        ("image", ("b.png", _png_bytes((32, 32)), "image/png")),
        ("mask", ("mask.png", _png_bytes((64, 48), mode="L"), "image/png")),
    ]
    response = client.post(
        "/v1/images/edits",
        data={"model": "Krea-2-Turbo-mflux-q8", "prompt": "replace the sky"},
        files=files,
    )

    assert response.status_code == 200, response.text
    request, paths, existed, mask_path, mask_existed = client.engine.edit_calls[0]
    assert len(paths) == 2
    assert existed == [True, True]  # staged before the engine ran
    assert mask_path is not None and mask_existed is True
    assert request.prompt == "replace the sky"
    # Staged files are cleaned up once the response is built.
    assert not any(Path(path).exists() for path in paths)


def test_edits_accepts_bracket_alias(client):
    files = [("image[]", ("a.png", _png_bytes(), "image/png"))]
    response = client.post(
        "/v1/images/edits",
        data={"model": "Krea-2-Turbo-mflux-q8", "prompt": "x"},
        files=files,
    )
    assert response.status_code == 200, response.text
    assert len(client.engine.edit_calls[0][1]) == 1


def test_edits_requires_an_image(client):
    response = client.post(
        "/v1/images/edits",
        data={"model": "Krea-2-Turbo-mflux-q8", "prompt": "x"},
    )
    assert response.status_code == 400
    assert "At least one image" in response.json()["detail"]


def test_edits_rejects_unreadable_image(client):
    files = [("image", ("a.png", b"not an image", "image/png"))]
    response = client.post(
        "/v1/images/edits",
        data={"model": "Krea-2-Turbo-mflux-q8", "prompt": "x"},
        files=files,
    )
    assert response.status_code == 400
    assert "not a readable image" in response.json()["detail"]


def test_edits_forwards_size_strength_and_format(client):
    files = [("image", ("a.png", _png_bytes(), "image/png"))]
    response = client.post(
        "/v1/images/edits",
        data={
            "model": "Krea-2-Turbo-mflux-q8",
            "prompt": "x",
            "width": "640",
            "height": "512",
            "strength": "0.5",
            "steps": "6",
            "seed": "11",
            "response_format": "url",
        },
        files=files,
    )

    assert response.status_code == 200, response.text
    request = client.engine.edit_calls[0][0]
    assert (request.width, request.height) == (640, 512)
    assert request.strength == 0.5
    assert request.steps == 6
    assert request.seed == 11
    assert response.json()["data"][0]["url"]


def test_edits_transparent_background_for_qwen(client):
    client.engine.spec = IMAGE_MODELS["qwen-image-2.1-turbo"]
    files = [("image", ("a.png", _png_bytes(), "image/png"))]
    response = client.post(
        "/v1/images/edits",
        data={
            "model": "Krea-2-Turbo-mflux-q8",
            "prompt": "x",
            "background": "transparent",
            "output_format": "png",
        },
        files=files,
    )
    assert response.status_code == 200, response.text

    rejected = client.post(
        "/v1/images/edits",
        data={
            "model": "Krea-2-Turbo-mflux-q8",
            "prompt": "x",
            "background": "transparent",
            "output_format": "jpeg",
        },
        files=files,
    )
    assert rejected.status_code == 400


def test_edits_validates_strength_range(client):
    files = [("image", ("a.png", _png_bytes(), "image/png"))]
    response = client.post(
        "/v1/images/edits",
        data={"model": "Krea-2-Turbo-mflux-q8", "prompt": "x", "strength": "1.5"},
        files=files,
    )
    assert response.status_code == 422
