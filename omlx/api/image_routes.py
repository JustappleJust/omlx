# SPDX-License-Identifier: Apache-2.0
"""Image endpoints (OpenAI-compatible ``/v1/images/generations`` and ``/edits``).

Model resolution, loading, memory admission and the eviction-proof lease all
go through :func:`omlx.server.get_engine` (``EngineType.IMAGE``), so image models share the
engine pool's LRU / TTL / pinning and the process memory enforcer with every
other model type.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from ..exceptions import OutOfMemoryError
from ..server_metrics import get_server_metrics
from .image_io import MAX_EDIT_IMAGES, decode_base64_image, decode_image, read_upload
from .image_models import (
    ImageData,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageUsage,
    parse_size,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@asynccontextmanager
async def _acquire(model: str, needs_edit: bool = False):
    """Yield ``(engine, pool_model_id)`` under an eviction-proof lease.

    A diffusion run holds the GPU for seconds to minutes; the lease keeps the
    memory enforcer's LRU from evicting the model mid-generation.
    """
    from omlx.server import (
        EngineType,
        _pool_resolve_model_id,
        _server_state,
        get_claude_tier_aliases,
        get_engine,
        get_engine_pool,
    )

    # Reject a known non-image model before get_engine loads it (and
    # possibly evicts other models) only to fail the type check.
    pool = get_engine_pool()
    try:
        resolved = _pool_resolve_model_id(
            pool, model, _server_state.settings_manager, get_claude_tier_aliases()
        )
        entry = pool.get_entry(resolved)
    except Exception:
        entry = None
    if entry is not None and entry.engine_type != "image":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not an image-generation model "
            f"(type: {entry.model_type}).",
        )
    if needs_edit and entry is not None:
        from ..image_gen.registry import find_adapter

        adapter = find_adapter(entry.model_path)
        if adapter is not None and not adapter.capabilities.edit:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' does not support image editing.",
            )

    leased: list = []
    engine = await get_engine(model, EngineType.IMAGE, _lease=True, _leased_out=leased)
    try:
        yield engine, (leased[0] if leased else model)
    finally:
        if leased:
            await get_engine_pool().release_engine(leased[0])


def _server_state():
    from omlx.server import _server_state

    return _server_state


def _model_defaults(model_id: str) -> dict:
    """Per-model defaults saved in the admin model settings."""
    try:
        sm = getattr(_server_state(), "settings_manager", None)
        if sm is None:
            return {}
        s = sm.get_settings(model_id)
        out = {}
        if s.image_default_steps:
            out["steps"] = s.image_default_steps
        if s.image_default_guidance is not None:
            out["guidance"] = s.image_default_guidance
        if s.image_default_size:
            out["width"], out["height"] = parse_size(s.image_default_size)
        return out
    except Exception:  # pragma: no cover - settings are best effort
        logger.debug("Could not read image model defaults", exc_info=True)
        return {}


def _check_not_quantizing() -> None:
    oq_manager = getattr(_server_state(), "oq_manager", None)
    if oq_manager and oq_manager.is_quantizing:
        raise HTTPException(
            status_code=503,
            detail="Server is busy with oQ quantization. Please try again after quantization completes.",
        )


async def _run(
    request: ImageGenerationRequest,
    raw: Request,
    references: Optional[list] = None,
    mask=None,
    strength: Optional[float] = None,
) -> ImageGenerationResponse:
    """Shared generate/edit flow: lease, build params, run, encode response."""
    _check_not_quantizing()
    request_id = f"img-{uuid.uuid4().hex[:12]}"
    async with _acquire(request.model, needs_edit=bool(references)) as (engine, model_id):
        defaults = _model_defaults(model_id)
        width = height = None
        if request.size:
            width, height = parse_size(request.size)
        elif not references:
            width, height = defaults.get("width"), defaults.get("height")
        steps = request.effective_steps or defaults.get("steps")
        guidance = request.effective_guidance
        if guidance is None:
            guidance = defaults.get("guidance")
        try:
            params = engine.build_params(
                prompt=request.prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=request.seed,
                negative_prompt=request.negative_prompt,
                references=references,
                mask=mask,
                strength=strength,
                accel=request.accel,
                scheduler=request.scheduler,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        gen = asyncio.ensure_future(
            engine.generate(
                params,
                n=request.n,
                output_format=request.output_format,
                request_id=request_id,
            )
        )
        # Abort the diffusion loop if the client goes away.
        try:
            while True:
                done, _ = await asyncio.wait({gen}, timeout=1.0)
                if done:
                    break
                if await raw.is_disconnected():
                    engine.cancel(request_id)
                    try:
                        await gen
                    except Exception:
                        pass
                    raise HTTPException(status_code=499, detail="Client disconnected")
            results = gen.result()
        except asyncio.CancelledError:
            engine.cancel(request_id)
            raise
        except OutOfMemoryError as e:
            raise HTTPException(status_code=507, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Image generation failed")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {e}") from e

    mime = f"image/{request.output_format}"
    data = []
    for r in results:
        b64 = base64.b64encode(r["data"]).decode("ascii")
        if request.response_format == "url":
            data.append(ImageData(url=f"data:{mime};base64,{b64}", seed=r["seed"]))
        else:
            data.append(ImageData(b64_json=b64, seed=r["seed"]))

    try:
        get_server_metrics().record_request_complete(
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            model_id=model_id,
        )
    except Exception:  # pragma: no cover
        pass

    return ImageGenerationResponse(
        created=int(time.time()),
        data=data,
        output_format=request.output_format,
        size=f"{params.width}x{params.height}",
        usage=ImageUsage(
            steps=params.steps,
            width=params.width,
            height=params.height,
            seconds=round(sum(r["seconds"] for r in results), 3),
            references=len(params.references),
        ),
    )


@router.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image(request: ImageGenerationRequest, raw: Request):
    return await _run(request, raw)


_FORM_LIST_FIELDS = ("image", "image[]", "images", "images[]")


async def _parse_edit_request(raw: Request) -> tuple[ImageEditRequest, list, Any]:
    """Accept OpenAI multipart (image / image[] / mask files) or JSON (base64)."""
    content_type = raw.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await raw.form()
        files = []
        for key in _FORM_LIST_FIELDS:
            files.extend(v for v in form.getlist(key) if not isinstance(v, str))
        fields: dict[str, Any] = {}
        for key, value in form.multi_items():
            if key in _FORM_LIST_FIELDS or key == "mask" or not isinstance(value, str):
                continue
            fields[key] = value
        if "accel" in fields:
            fields["accel"] = _parse_json_field("accel", fields["accel"])
        try:
            req = ImageEditRequest.model_validate(fields)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors(include_url=False)) from e
        refs = [decode_image(await read_upload(f), f"image[{i}]") for i, f in enumerate(files)]
        mask_file = form.get("mask")
        mask = (
            decode_image(await read_upload(mask_file), "mask")
            if mask_file is not None and not isinstance(mask_file, str)
            else None
        )
    else:
        try:
            body = await raw.json()
            req = ImageEditRequest.model_validate(body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors(include_url=False)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}") from e
        refs = [decode_image(decode_base64_image(v), f"images[{i}]") for i, v in enumerate(req.images)]
        mask = decode_image(decode_base64_image(req.mask), "mask") if req.mask else None
    if not refs:
        raise HTTPException(status_code=400, detail="At least one input image is required.")
    if len(refs) > MAX_EDIT_IMAGES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_EDIT_IMAGES} input images.")
    return req, refs, mask


def _parse_json_field(name: str, value: str):
    import json

    try:
        return json.loads(value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Field {name!r} must be JSON") from e


@router.post("/v1/images/edits", response_model=ImageGenerationResponse)
async def edit_image(raw: Request):
    """OpenAI-compatible image editing (one or more reference images)."""
    req, refs, mask = await _parse_edit_request(raw)
    return await _run(req, raw, references=refs, mask=mask, strength=req.strength)


@router.get("/v1/images/models")
async def list_image_models():
    """Image-generation models known to the pool, with their defaults."""
    state = _server_state()
    pool = getattr(state, "engine_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    from ..image_gen.registry import find_adapter

    out = []
    for mid, entry in pool._entries.items():
        if entry.engine_type != "image":
            continue
        adapter = find_adapter(entry.model_path)
        d = adapter.defaults if adapter else None
        out.append(
            {
                "id": mid,
                "adapter": adapter.name if adapter else None,
                "capabilities": adapter.capabilities.as_dict() if adapter else None,
                "loaded": entry.engine is not None,
                "estimated_size": entry.estimated_size,
                "defaults": (
                    {
                        "steps": d.steps,
                        "guidance": d.guidance,
                        "size": f"{d.width}x{d.height}",
                        "max_side": d.max_side,
                        "min_side": d.min_side,
                        "size_multiple": d.size_multiple,
                    }
                    if d
                    else None
                ),
            }
        )
    return {"object": "list", "data": out}
