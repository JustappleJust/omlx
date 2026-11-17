# SPDX-License-Identifier: Apache-2.0
"""/v1/images/generations with a fake engine."""

import base64
import io
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import omlx.api.image_routes as routes
from omlx.api.image_models import parse_size
from omlx.engine.image_gen import ImageGenEngine
from tests.test_image_engine import FakeAdapter


class EditFakeAdapter(FakeAdapter):
    from omlx.image_gen.base import ImageCapabilities as _C

    capabilities = _C(generate=True, edit=True, max_references=2, mask=True, strength=True)

    def generate(self, model, params, on_step, plan=None):
        self.last = params
        return super().generate(model, params, on_step, plan)


@pytest.fixture
def client(monkeypatch):
    engine = ImageGenEngine("/models/fake", adapter=FakeAdapter())
    editor = ImageGenEngine("/models/editor", adapter=EditFakeAdapter())
    state = {"started": False}

    @asynccontextmanager
    async def fake_acquire(model, needs_edit=False):
        if model == "missing":
            raise HTTPException(status_code=404, detail="not found")
        if model == "llm":
            raise HTTPException(status_code=400, detail="not an image model")
        if not state["started"]:
            await engine.start()
            await editor.start()
            state["started"] = True
        yield (editor if model == "editor" else engine), model

    class _State:
        oq_manager = None
        settings_manager = None
        engine_pool = None

    monkeypatch.setattr(routes, "_acquire", fake_acquire)
    monkeypatch.setattr(routes, "_server_state", lambda: _State())
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), engine


def test_generate_b64(client):
    c, _ = client
    r = c.post(
        "/v1/images/generations",
        json={"model": "fake", "prompt": "a fox", "size": "640x384", "seed": 3, "n": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["size"] == "640x384"
    assert [d["seed"] for d in body["data"]] == [3, 4]
    img = Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"])))
    assert img.size == (640, 384)
    assert body["usage"]["steps"] == 4


def test_url_response_and_defaults(client):
    c, _ = client
    r = c.post(
        "/v1/images/generations",
        json={"model": "fake", "prompt": "x", "response_format": "url", "quality": "hd"},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["size"] == "512x512"
    assert d["data"][0]["url"].startswith("data:image/png;base64,")


def test_errors(client):
    c, _ = client
    assert c.post("/v1/images/generations", json={"model": "missing", "prompt": "x"}).status_code == 404
    assert c.post("/v1/images/generations", json={"model": "llm", "prompt": "x"}).status_code == 400
    assert c.post("/v1/images/generations", json={"model": "fake", "prompt": "x", "size": "big"}).status_code == 422
    assert c.post("/v1/images/generations", json={"model": "fake", "prompt": ""}).status_code == 422
    assert c.post("/v1/images/generations", json={"model": "fake", "prompt": "x", "steps": 500}).status_code == 422


def test_parse_size():
    assert parse_size("1024x768") == (1024, 768)
    assert parse_size(" 512 X 512 ") == (512, 512)
    with pytest.raises(ValueError):
        parse_size("1024")


def _png(size=(640, 480), color=(0, 128, 255), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_edit_multipart_multiple_images_and_mask(client):
    c, _ = client
    files = [
        ("image[]", ("a.png", _png(), "image/png")),
        ("image[]", ("b.png", _png((320, 320)), "image/png")),
        ("mask", ("m.png", _png((640, 480), 255, "L"), "image/png")),
    ]
    r = c.post(
        "/v1/images/edits",
        data={"model": "editor", "prompt": "make it red", "seed": "9", "strength": "0.8"},
        files=files,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["usage"]["references"] == 2
    assert body["size"] == "640x480"  # canvas follows the first reference


def test_edit_json_base64_and_data_url(client):
    c, _ = client
    b64 = base64.b64encode(_png()).decode()
    r = c.post(
        "/v1/images/edits",
        json={"model": "editor", "prompt": "x", "images": [f"data:image/png;base64,{b64}"], "size": "512x512"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["size"] == "512x512"


def test_edit_errors(client):
    c, _ = client
    b64 = base64.b64encode(_png()).decode()
    # model without edit support
    r = c.post("/v1/images/edits", json={"model": "fake", "prompt": "x", "images": [b64]})
    assert r.status_code == 400 and "edit" in r.text
    # no images
    assert c.post("/v1/images/edits", json={"model": "editor", "prompt": "x"}).status_code == 400
    # too many for the model
    r = c.post("/v1/images/edits", json={"model": "editor", "prompt": "x", "images": [b64] * 3})
    assert r.status_code == 400
    # not an image / not base64
    assert c.post("/v1/images/edits", json={"model": "editor", "prompt": "x", "images": ["@@@"]}).status_code == 400
    junk = base64.b64encode(b"not an image").decode()
    assert c.post("/v1/images/edits", json={"model": "editor", "prompt": "x", "images": [junk]}).status_code == 400
    # lossy accel the model does not implement
    r = c.post(
        "/v1/images/edits",
        json={"model": "editor", "prompt": "x", "images": [b64], "accel": {"step_cache": True}},
    )
    assert r.status_code == 400


def test_edit_rejects_decompression_bomb(client, monkeypatch):
    import omlx.api.image_io as io_mod

    monkeypatch.setattr(io_mod, "MAX_IMAGE_PIXELS", 1000)
    c, _ = client
    b64 = base64.b64encode(_png((100, 100))).decode()
    r = c.post("/v1/images/edits", json={"model": "editor", "prompt": "x", "images": [b64]})
    assert r.status_code == 400 and "pixel limit" in r.text
