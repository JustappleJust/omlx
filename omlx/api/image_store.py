# SPDX-License-Identifier: Apache-2.0
"""In-memory store for ``/v1/images/*`` responses requested as URLs.

OpenAI's Images API can return either base64 JSON or a URL. oMLX has no object
store, so URL responses are served from this bounded, expiring in-memory cache:
the caller gets a link it can fetch back from the same server, and images are
dropped when the cache rolls over or the TTL expires. Clients that need the
bytes permanently (or across restarts) should ask for ``b64_json``.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict

#: How many generated images stay fetchable at once.
MAX_STORED_IMAGES = 128
#: How long a URL stays valid after generation, in seconds.
IMAGE_TTL_SECONDS = 3600.0

_LOCK = threading.Lock()
_IMAGES: OrderedDict[str, tuple[bytes, str, float]] = OrderedDict()


class ImageNotFound(KeyError):  # noqa: N818
    """The requested image URL is unknown or has expired."""


def store_image(data: bytes, mime_type: str) -> str:
    """Store ``data`` and return its opaque id."""
    image_id = uuid.uuid4().hex
    now = time.monotonic()
    with _LOCK:
        _evict_expired_locked(now)
        _IMAGES[image_id] = (data, mime_type, now)
        while len(_IMAGES) > MAX_STORED_IMAGES:
            _IMAGES.popitem(last=False)
    return image_id


def get_image(image_id: str) -> tuple[bytes, str]:
    """Return ``(bytes, mime_type)`` for ``image_id`` or raise ImageNotFound."""
    now = time.monotonic()
    with _LOCK:
        _evict_expired_locked(now)
        entry = _IMAGES.get(image_id)
        if entry is None:
            raise ImageNotFound(image_id)
        # Refresh recency so actively fetched images survive the cap.
        _IMAGES.move_to_end(image_id)
        data, mime_type, _ = entry
        return data, mime_type


def clear_images() -> None:
    """Drop every stored image (used by tests and shutdown)."""
    with _LOCK:
        _IMAGES.clear()


def _evict_expired_locked(now: float) -> None:
    expired = [
        image_id
        for image_id, (_, _, created) in _IMAGES.items()
        if now - created > IMAGE_TTL_SECONDS
    ]
    for image_id in expired:
        _IMAGES.pop(image_id, None)
