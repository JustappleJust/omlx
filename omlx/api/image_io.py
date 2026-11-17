# SPDX-License-Identifier: Apache-2.0
"""Safe decoding of user-supplied images for the edit endpoint.

Uploaded images are untrusted: bound the byte size, refuse decompression
bombs before pixels are decoded, normalise EXIF orientation, and convert to a
plain RGB/RGBA PIL image the adapters can consume.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
from typing import Any

from fastapi import HTTPException

#: Per-image upload limit.
MAX_IMAGE_BYTES = 32 * 1024 * 1024
#: Decoded pixel limit (a 64 MP image is far beyond any model's canvas).
MAX_IMAGE_PIXELS = 64 * 1024 * 1024
#: Reference images per request (models enforce their own lower limits).
MAX_EDIT_IMAGES = 16

_DATA_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.ASCII)


def decode_base64_image(value: str) -> bytes:
    """Plain base64 or a ``data:image/...;base64,`` URL -> bytes."""
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=400, detail="Image must be a non-empty base64 string.")
    payload = _DATA_URL_RE.sub("", value.strip(), count=1)
    if len(payload) > MAX_IMAGE_BYTES * 4 // 3 + 16:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_BYTES} bytes.")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail="Image is not valid base64.") from e


async def read_upload(upload: Any) -> bytes:
    """Read an uploaded file in chunks, refusing anything over the limit."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_BYTES} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


def decode_image(data: bytes, field: str = "image"):
    """Bytes -> RGB/RGBA PIL image with EXIF orientation applied."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not data:
        raise HTTPException(status_code=400, detail=f"{field}: empty image.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{field}: image exceeds {MAX_IMAGE_BYTES} bytes.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size  # header only; pixels not decoded yet
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=400,
                    detail=f"{field}: {width}x{height} exceeds the {MAX_IMAGE_PIXELS} pixel limit.",
                )
            probe.load()
            image = ImageOps.exif_transpose(probe)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "A" in image.getbands() or image.mode == "P" else "RGB")
            else:
                image = image.copy()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        raise HTTPException(status_code=400, detail=f"{field}: not a valid image ({e}).") from e
    return image
