# SPDX-License-Identifier: Apache-2.0
"""Image-generation model support (text-to-image diffusion).

See :mod:`omlx.image_gen.base` for the adapter interface and
:mod:`omlx.image_gen.registry` for the list of supported families.
"""

from .base import (
    OMLX_IMAGE_MARKER,
    ImageDefaults,
    ImageGenerationCancelled,
    ImageGenParams,
    ImageModelAdapter,
)
from .registry import (
    ADAPTERS,
    find_adapter,
    get_adapter,
    is_image_model_dir,
    mflux_available,
)

#: Default load-time quantization (bits) for image models; None = bf16.
DEFAULT_IMAGE_QUANTIZE = 8

__all__ = [
    "ADAPTERS",
    "DEFAULT_IMAGE_QUANTIZE",
    "OMLX_IMAGE_MARKER",
    "ImageDefaults",
    "ImageGenParams",
    "ImageGenerationCancelled",
    "ImageModelAdapter",
    "find_adapter",
    "get_adapter",
    "is_image_model_dir",
    "mflux_available",
]
