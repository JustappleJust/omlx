# SPDX-License-Identifier: Apache-2.0
"""Registry of image-generation model adapters.

To add a model family: implement :class:`~omlx.image_gen.base.ImageModelAdapter`
in a new module and append an instance to ``ADAPTERS``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import ImageModelAdapter, read_model_index
from .flux3 import Flux3KleinAdapter
from .krea2 import Krea2Adapter
from .qwen21 import QwenImage21Adapter

# Order matters only for layouts more than one adapter could claim; the
# index/config-based detectors are unambiguous.
ADAPTERS: list[ImageModelAdapter] = [
    Krea2Adapter(),
    Flux3KleinAdapter(),
    QwenImage21Adapter(),
]


def find_adapter(model_dir: Path | str) -> Optional[ImageModelAdapter]:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return None
    # Adapters may also recognise layouts without a model_index.json
    # (e.g. mflux-saved checkpoints), so an empty index is passed through.
    index = read_model_index(model_dir) or {}
    for adapter in ADAPTERS:
        if adapter.matches(model_dir, index):
            return adapter
    return None


def is_image_model_dir(model_dir: Path | str) -> bool:
    return find_adapter(model_dir) is not None


def get_adapter(name: str) -> Optional[ImageModelAdapter]:
    for adapter in ADAPTERS:
        if adapter.name == name:
            return adapter
    return None


def mflux_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("mflux") is not None
