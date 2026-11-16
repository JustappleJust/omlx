# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-memory store behind ``response_format="url"``."""

import pytest

from omlx.api import image_store


@pytest.fixture(autouse=True)
def _clean_store():
    image_store.clear_images()
    yield
    image_store.clear_images()


def test_store_and_get_round_trip():
    image_id = image_store.store_image(b"\x89PNG payload", "image/png")
    assert image_store.get_image(image_id) == (b"\x89PNG payload", "image/png")


def test_unknown_id_raises():
    with pytest.raises(image_store.ImageNotFound):
        image_store.get_image("nope")


def test_ids_are_unique():
    ids = {image_store.store_image(b"x", "image/png") for _ in range(32)}
    assert len(ids) == 32


def test_cap_evicts_the_oldest(monkeypatch):
    monkeypatch.setattr(image_store, "MAX_STORED_IMAGES", 2)
    first = image_store.store_image(b"1", "image/png")
    second = image_store.store_image(b"2", "image/png")
    third = image_store.store_image(b"3", "image/png")

    with pytest.raises(image_store.ImageNotFound):
        image_store.get_image(first)
    assert image_store.get_image(second)[0] == b"2"
    assert image_store.get_image(third)[0] == b"3"


def test_reading_refreshes_recency(monkeypatch):
    monkeypatch.setattr(image_store, "MAX_STORED_IMAGES", 2)
    first = image_store.store_image(b"1", "image/png")
    second = image_store.store_image(b"2", "image/png")
    image_store.get_image(first)  # first becomes the most recently used
    image_store.store_image(b"3", "image/png")

    assert image_store.get_image(first)[0] == b"1"
    with pytest.raises(image_store.ImageNotFound):
        image_store.get_image(second)


def test_expired_images_are_dropped(monkeypatch):
    now = {"value": 1000.0}
    monkeypatch.setattr(image_store.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(image_store, "IMAGE_TTL_SECONDS", 60.0)

    image_id = image_store.store_image(b"payload", "image/png")
    now["value"] += 30
    assert image_store.get_image(image_id)[0] == b"payload"

    now["value"] += 31
    with pytest.raises(image_store.ImageNotFound):
        image_store.get_image(image_id)
