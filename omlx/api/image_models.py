# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible request/response models for image generation."""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SIZE_RE = re.compile(r"^\s*(\d{2,5})\s*[xX×]\s*(\d{2,5})\s*$")


def parse_size(size: str) -> tuple[int, int]:
    """Parse ``"WIDTHxHEIGHT"`` into ``(width, height)``."""
    m = _SIZE_RE.match(size or "")
    if not m:
        raise ValueError(f"Invalid size {size!r}; expected WIDTHxHEIGHT, e.g. 1024x1024")
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        raise ValueError("size dimensions must be positive")
    return w, h


class ImageGenerationRequest(BaseModel):
    """``POST /v1/images/generations``.

    OpenAI fields plus oMLX extensions (``seed``, ``steps``, ``guidance``,
    ``negative_prompt``, ``scheduler``). Unknown OpenAI fields such as
    ``quality``/``style``/``user`` are accepted and ignored.
    """

    model_config = ConfigDict(extra="ignore")

    model: str
    prompt: str = Field(min_length=1, max_length=8000)
    n: int = Field(default=1, ge=1, le=8)
    size: Optional[str] = None  # None / "auto" -> model default
    response_format: Literal["b64_json", "url"] = "b64_json"
    output_format: Literal["png", "jpeg", "webp"] = "png"
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)
    steps: Optional[int] = Field(default=None, ge=1, le=100)
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=100)
    guidance: Optional[float] = Field(default=None, ge=0, le=30)
    guidance_scale: Optional[float] = Field(default=None, ge=0, le=30)
    negative_prompt: Optional[str] = Field(default=None, max_length=8000)
    scheduler: Optional[str] = None
    #: Opt-in lossy accelerations, e.g. {"step_cache": true}.
    accel: Optional[Dict[str, bool]] = None

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value.strip().lower() == "auto":
            return None
        parse_size(value)
        return value

    @property
    def effective_steps(self) -> Optional[int]:
        return self.steps if self.steps is not None else self.num_inference_steps

    @property
    def effective_guidance(self) -> Optional[float]:
        return self.guidance if self.guidance is not None else self.guidance_scale


class ImageEditRequest(ImageGenerationRequest):
    """``POST /v1/images/edits``.

    Multipart requests carry the images as ``image`` / ``image[]`` files (and
    ``mask``); JSON requests carry them base64-encoded (``images``, ``mask``,
    plain base64 or ``data:`` URLs).
    """

    images: List[str] = Field(default_factory=list, max_length=16)
    mask: Optional[str] = None
    strength: Optional[float] = Field(default=None, gt=0, le=1)


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None
    seed: Optional[int] = None


class ImageUsage(BaseModel):
    steps: int
    width: int
    height: int
    seconds: float
    references: int = 0


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageData]
    output_format: str = "png"
    size: Optional[str] = None
    usage: Optional[ImageUsage] = None
