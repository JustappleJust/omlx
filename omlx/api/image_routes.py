# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible image generation endpoints.

- ``POST /v1/images/generations`` - text-to-image
- ``POST /v1/images/edits``       - reference editing / img2img
- ``GET  /v1/images/generated/{id}`` - fetch an image returned as a URL

The heavy lifting lives in :mod:`omlx.engine.image`; this module validates the
HTTP surface, picks the model, and translates engine failures into status
codes. mflux is imported lazily so the rest of the API keeps working when the
optional ``omlx[image]`` extra is not installed.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import logging
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import ValidationError

from ..image_models import (
    ImageBackendUnavailable,
    ImageGenerationError,
    ImageModelSpec,
    ImageRequest,
    ImageRequestError,
    detect_image_spec,
    get_spec,
    parse_image_size,
)
from .image_models import (
    ImageDataItem,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from .image_store import ImageNotFound, get_image, store_image

logger = logging.getLogger(__name__)

router = APIRouter()

#: Largest reference image accepted by ``/v1/images/edits`` (25 MB).
MAX_IMAGE_UPLOAD_BYTES = 25 * 1024 * 1024

_MISSING_MFLUX_DETAIL = (
    "Image generation is not available: the optional mflux runtime is not "
    'installed. Install it with: pip install "omlx[image]" (Apple Silicon, '
    "macOS 15+), then restart the server."
)

_IMAGE_SUFFIXES = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "GIF": ".gif",
    "TIFF": ".tiff",
}


# ---------------------------------------------------------------------------
# Dependencies (patchable in tests)
# ---------------------------------------------------------------------------


def _get_engine_pool():
    """Return the active EnginePool from server state.

    Imported lazily to avoid a circular import at module load time. Can be
    replaced in tests via ``patch('omlx.api.image_routes._get_engine_pool')``.
    """
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    """Resolve a model alias to its real model ID."""
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


def _is_image_engine(engine: object) -> bool:
    """True for the image engine (fakes set ``is_image_engine`` too)."""
    if getattr(engine, "is_image_engine", False):
        return True
    from ..engine.image import ImageGenerationEngine

    return isinstance(engine, ImageGenerationEngine)


def _require_mflux() -> None:
    """Fail early with an install hint when mflux is missing."""
    if importlib.util.find_spec("mflux") is None:
        raise HTTPException(status_code=503, detail=_MISSING_MFLUX_DETAIL)


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def _entry_is_image(entry: object) -> bool:
    return getattr(entry, "model_type", None) == "image" or getattr(
        entry, "engine_type", None
    ) == "image"


def _image_models_from_pool(pool) -> list[dict]:
    models = pool.get_status().get("models", [])
    return [m for m in models if m.get("model_type") == "image" or m.get("engine_type") == "image"]


def _resolve_image_model(requested: str) -> tuple[str, str]:
    """Return ``(model_id, model_path)`` for an image request.

    Resolution order: the discovered id (or alias), then a registry name such
    as ``"krea-2-turbo"`` matched against the image models the pool knows
    about. Matching by registry name lets clients use the documented model
    names without configuring an alias first.
    """
    pool = _get_engine_pool()
    resolved = _resolve_model(requested)
    entry = pool.get_entry(resolved)
    if entry is not None:
        if not _entry_is_image(entry):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{resolved}' is not an image model "
                    f"(type: {entry.model_type}). Use /v1/chat/completions, "
                    "/v1/embeddings, or the model's dedicated endpoint."
                ),
            )
        return resolved, entry.model_path

    spec = get_spec(requested)
    if spec is not None:
        matches = [
            model
            for model in _image_models_from_pool(pool)
            if (detected := detect_image_spec(model["model_path"])) is not None
            and detected.key == spec.key
        ]
        if len(matches) == 1:
            return matches[0]["id"], matches[0]["model_path"]
        if len(matches) > 1:
            ids = ", ".join(m["id"] for m in matches)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{requested}' matches several loaded checkpoints: {ids}. "
                    "Pass the model directory name or configure an alias."
                ),
            )

    available = [m["id"] for m in _image_models_from_pool(pool)]
    if not available:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{requested}' not found and no image models are "
                "installed. Download FLUX 3 Image Klein, Krea 2 Turbo, or "
                "Qwen-Image-2.1-Turbo into the model directory."
            ),
        )
    raise HTTPException(
        status_code=404,
        detail=(
            f"Image model '{requested}' not found. Available image models: "
            f"{', '.join(available)}"
        ),
    )


def _spec_for_engine(engine: object) -> ImageModelSpec | None:
    spec = getattr(engine, "spec", None)
    if isinstance(spec, ImageModelSpec):
        return spec
    model_name = getattr(engine, "model_name", None)
    return detect_image_spec(model_name) if model_name else None


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _request_dimensions(
    size: str, width: int | None, height: int | None
) -> tuple[int | None, int | None]:
    """Combine OpenAI's ``size`` with the explicit width/height extension."""
    try:
        size_width, size_height = parse_image_size(size)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    has_explicit = width is not None or height is not None
    if has_explicit and size_width is not None:
        raise HTTPException(
            status_code=400,
            detail="Pass either 'size' or 'width'/'height', not both.",
        )
    if has_explicit:
        if width is None or height is None:
            raise HTTPException(
                status_code=400, detail="Pass both 'width' and 'height', or neither."
            )
        return width, height
    return size_width, size_height


def _check_background(
    request_background: str | None,
    spec: ImageModelSpec | None,
    output_format: str = "png",
) -> None:
    """Reject ``background="transparent"`` when the request cannot honour it."""
    if (request_background or "").lower() != "transparent":
        return
    if spec is None or not spec.outputs_alpha:
        name = spec.display_name if spec else "This model"
        raise HTTPException(
            status_code=400,
            detail=f"{name} cannot produce transparent images.",
        )
    if output_format not in ("png", "webp"):
        raise HTTPException(
            status_code=400,
            detail="Transparent backgrounds require output_format png or webp.",
        )


def _translate_engine_error(error: Exception) -> HTTPException:
    """Map engine and pool exceptions onto HTTP status codes."""
    if isinstance(error, HTTPException):
        return error
    if isinstance(error, ImageRequestError):
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, ImageBackendUnavailable):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, ImageGenerationError):
        return HTTPException(
            status_code=500, detail=f"Image generation failed: {error}"
        )

    # Pool failures: the model vanished or could not be loaded/admitted.
    from ..exceptions import ModelLoadingError, ModelNotFoundError

    if isinstance(error, ModelNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, ModelLoadingError):
        return HTTPException(
            status_code=503,
            detail=f"Could not load the image model: {error}",
        )
    return HTTPException(status_code=500, detail=f"Image generation failed: {error}")


def _record_request(model_id: str) -> None:
    """Count the request without inventing token usage."""
    try:
        from ..server_metrics import get_server_metrics

        get_server_metrics().record_request_complete(
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            model_id=model_id,
        )
    except Exception as exc:  # pragma: no cover - metrics are best effort
        logger.debug("Could not record image metrics for %s: %s", model_id, exc)


def _build_response(
    *,
    result,
    request_model: str,
    response_format: str,
    base_url: str,
    created: int,
) -> ImageGenerationResponse:
    data: list[ImageDataItem] = []
    for payload in result.images:
        if response_format == "url":
            image_id = store_image(payload, result.mime_type)
            data.append(
                ImageDataItem(url=f"{base_url}/v1/images/generated/{image_id}")
            )
        else:
            data.append(
                ImageDataItem(b64_json=base64.b64encode(payload).decode("ascii"))
            )
    return ImageGenerationResponse(
        created=created,
        data=data,
        model=request_model,
        seed=result.seed,
    )


def _base_url_for(request: Request) -> str:
    return str(request.base_url).rstrip("/")


async def _read_upload_limited(upload: UploadFile) -> bytes:
    """Read an upload in chunks, rejecting anything over the size cap."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_IMAGE_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Reference image exceeds the maximum allowed size "
                    f"({MAX_IMAGE_UPLOAD_BYTES} bytes)."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _write_reference(data: bytes, directory: Path, name: str, *, is_mask: bool) -> Path:
    """Validate that ``data`` is an image and write it to ``directory``."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = probe.format
            probe.verify()
    except Exception as error:  # noqa: BLE001 - every decoder failure is a 400
        kind = "mask" if is_mask else "reference image"
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded {kind} is not a readable image: {error}",
        ) from error

    suffix = _IMAGE_SUFFIXES.get(str(image_format).upper(), ".png")
    path = directory / f"{name}{suffix}"
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/v1/images/generations",
    response_model=ImageGenerationResponse,
)
async def create_image_generation(
    body: ImageGenerationRequest,
    request: Request,
) -> ImageGenerationResponse:
    """Generate one or more images from a text prompt."""
    _require_mflux()
    model_id, _ = _resolve_image_model(body.model)
    width, height = _request_dimensions(body.size, body.width, body.height)

    pool = _get_engine_pool()
    created = int(time.time())
    try:
        async with pool.acquire(model_id) as engine:
            if not _is_image_engine(engine):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_id}' is not an image model.",
                )
            _check_background(
                body.background,
                _spec_for_engine(engine),
                body.output_format,
            )
            result = await engine.generate(
                ImageRequest(
                    prompt=body.prompt,
                    n=body.n,
                    width=width,
                    height=height,
                    steps=body.steps,
                    guidance=body.guidance,
                    seed=body.seed,
                    negative_prompt=body.negative_prompt,
                    output_format=body.output_format,
                    output_compression=body.output_compression,
                )
            )
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001 - classified below
        raise _translate_engine_error(error) from error

    _record_request(model_id)
    return _build_response(
        result=result,
        request_model=body.model,
        response_format=body.response_format,
        base_url=_base_url_for(request),
        created=created,
    )


@router.post(
    "/v1/images/edits",
    response_model=ImageGenerationResponse,
)
async def create_image_edit(
    request: Request,
    model: str = Form(...),
    prompt: str = Form(...),
    image: list[UploadFile] | None = File(default=None),
    mask: UploadFile | None = File(default=None),
    image_array: list[UploadFile] | None = File(default=None, alias="image[]"),
    n: int = Form(default=1),
    size: str = Form(default="auto"),
    width: int | None = Form(default=None),
    height: int | None = Form(default=None),
    response_format: str = Form(default="b64_json"),
    output_format: str = Form(default="png"),
    steps: int | None = Form(default=None),
    guidance: float | None = Form(default=None),
    seed: int | None = Form(default=None),
    negative_prompt: str | None = Form(default=None),
    strength: float | None = Form(default=None),
    quality: str | None = Form(default=None),
    style: str | None = Form(default=None),
    background: str | None = Form(default=None),
    user: str | None = Form(default=None),
) -> ImageGenerationResponse:
    """Edit one or more reference images.

    FLUX 3 and Qwen-Image-2.1 perform instruction/multi-reference editing;
    Krea 2 performs strength-based img2img on a single reference.
    """
    # Validate the form through the shared model so limits stay in one place.
    try:
        form = ImageEditRequest(
            model=model,
            prompt=prompt,
            n=n,
            size=size,
            width=width,
            height=height,
            response_format=response_format,
            output_format=output_format,
            steps=steps,
            guidance=guidance,
            seed=seed,
            negative_prompt=negative_prompt,
            strength=strength,
            quality=quality,
            style=style,
            background=background,
            user=user,
        )
    except ValidationError as error:
        raise HTTPException(
            status_code=422,
            detail=error.errors(include_url=False),
        ) from error

    _require_mflux()
    model_id, _ = _resolve_image_model(form.model)
    resolved_width, resolved_height = _request_dimensions(
        form.size, form.width, form.height
    )
    uploads = [upload for upload in [*(image or []), *(image_array or [])] if upload]
    if not uploads:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    pool = _get_engine_pool()
    created = int(time.time())
    workdir = Path(tempfile.mkdtemp(prefix="omlx-image-edit-"))
    try:
        references = [
            _write_reference(
                await _read_upload_limited(upload),
                workdir,
                f"reference-{index}",
                is_mask=False,
            )
            for index, upload in enumerate(uploads)
        ]
        mask_path = None
        if mask is not None:
            mask_path = _write_reference(
                await _read_upload_limited(mask), workdir, "mask", is_mask=True
            )

        async with pool.acquire(model_id) as engine:
            if not _is_image_engine(engine):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_id}' is not an image model.",
                )
            _check_background(
                form.background,
                _spec_for_engine(engine),
                form.output_format,
            )
            result = await engine.edit(
                ImageRequest(
                    prompt=form.prompt,
                    n=form.n,
                    width=resolved_width,
                    height=resolved_height,
                    steps=form.steps,
                    guidance=form.guidance,
                    seed=form.seed,
                    negative_prompt=form.negative_prompt,
                    strength=form.strength,
                    output_format=form.output_format,
                    output_compression=form.output_compression,
                ),
                references,
                mask_path,
            )
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001 - classified below
        raise _translate_engine_error(error) from error
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    _record_request(model_id)
    return _build_response(
        result=result,
        request_model=form.model,
        response_format=form.response_format,
        base_url=_base_url_for(request),
        created=created,
    )


@router.get("/v1/images/generated/{image_id}")
async def get_generated_image(image_id: str) -> Response:
    """Serve an image previously returned with ``response_format="url"``."""
    try:
        payload, mime_type = get_image(image_id)
    except ImageNotFound:
        raise HTTPException(
            status_code=404,
            detail=(
                "Generated image not found or expired; URL responses are kept "
                'in memory for one hour. Request "b64_json" for durable output.'
            ),
        ) from None
    extension = mime_type.rsplit("/", 1)[-1].replace("jpeg", "jpg")
    return Response(
        content=payload,
        media_type=mime_type,
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Disposition": f'inline; filename="omlx-image.{extension}"',
        },
    )
