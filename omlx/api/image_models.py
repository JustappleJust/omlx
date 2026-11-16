# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the OpenAI-compatible image API.

The request bodies follow OpenAI's Images API. Fields this server does not act
on (``quality``, ``style``, ``moderation``, ``user``) are accepted so existing
clients keep working, and the sampling knobs mflux actually exposes are added
as documented extensions (``steps``, ``guidance``, ``seed``,
``negative_prompt``, ``width``/``height``, ``strength``).
"""

from typing import Literal

from pydantic import BaseModel, Field

#: Bound the size of a single prompt. mflux tokenizers truncate far below this.
MAX_PROMPT_CHARS = 40000
#: Images per request. Every image is a full denoising run on the GPU.
MAX_IMAGES = 4


class ImageGenerationRequest(BaseModel):
    """``POST /v1/images/generations`` body."""

    model: str
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    n: int = Field(default=1, ge=1, le=MAX_IMAGES)
    size: str = "auto"
    # Extension: explicit canvas, overrides ``size`` when both are given.
    width: int | None = Field(default=None, ge=16, le=8192)
    height: int | None = Field(default=None, ge=16, le=8192)
    response_format: Literal["b64_json", "url"] = "b64_json"
    output_format: Literal["png", "jpeg", "webp"] = "png"
    output_compression: int | None = Field(default=None, ge=0, le=100)

    # Sampling extensions.
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0)
    negative_prompt: str | None = None

    # OpenAI compatibility: accepted, not used.
    quality: str | None = None
    style: str | None = None
    background: str | None = None
    moderation: str | None = None
    user: str | None = None


class ImageEditRequest(BaseModel):
    """``POST /v1/images/edits`` form fields.

    The reference images themselves arrive as multipart uploads; everything
    else mirrors the generation request. ``strength`` only applies to
    strength-based img2img (Krea 2), and ``mask`` to Qwen-Image-2.1.
    """

    model: str
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    n: int = Field(default=1, ge=1, le=MAX_IMAGES)
    size: str = "auto"
    width: int | None = Field(default=None, ge=16, le=8192)
    height: int | None = Field(default=None, ge=16, le=8192)
    response_format: Literal["b64_json", "url"] = "b64_json"
    output_format: Literal["png", "jpeg", "webp"] = "png"
    output_compression: int | None = Field(default=None, ge=0, le=100)

    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0)
    negative_prompt: str | None = None
    strength: float | None = Field(default=None, gt=0, le=1)

    quality: str | None = None
    style: str | None = None
    background: str | None = None
    user: str | None = None


class ImageDataItem(BaseModel):
    """One generated image, exactly one of ``b64_json``/``url`` set."""

    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class ImageGenerationResponse(BaseModel):
    """OpenAI-compatible response, plus the effective seed."""

    created: int
    data: list[ImageDataItem]
    model: str | None = None
    seed: int | None = None
