# SPDX-License-Identifier: Apache-2.0
"""Engine-pool integration of image models (no weights loaded)."""

import pytest

from omlx.engine.image_gen import ImageGenEngine
from omlx.engine_pool import EnginePool
from omlx.exceptions import ModelNotFoundError
from omlx.model_settings import ModelSettings
from tests.test_image_discovery import GB, make_krea2


@pytest.fixture
def pool(tmp_path):
    make_krea2(tmp_path / "Krea-2-Turbo", diffusers=False)
    p = EnginePool()
    p._get_final_ceiling = lambda: 0
    p.discover_models(str(tmp_path))
    return p


def test_discovered_as_image_entry(pool):
    entry = pool.get_entry("Krea-2-Turbo")
    assert entry.model_type == "image" and entry.engine_type == "image"


def test_entry_without_config_json_is_not_treated_as_deleted(pool):
    entry = pool.get_entry("Krea-2-Turbo")
    pool._raise_if_model_path_missing_locked("Krea-2-Turbo", entry)  # no raise
    assert pool.get_entry("Krea-2-Turbo") is entry


def test_deleted_image_dir_still_detected(pool, tmp_path):
    import shutil

    entry = pool.get_entry("Krea-2-Turbo")
    shutil.rmtree(entry.model_path)
    with pytest.raises(ModelNotFoundError):
        pool._raise_if_model_path_missing_locked("Krea-2-Turbo", entry)


def test_quantize_setting_resolution():
    q = EnginePool._image_quantize_setting
    assert q(None) == 8
    assert q(ModelSettings()) == 8
    assert q(ModelSettings(image_quantize=0)) is None
    assert q(ModelSettings(image_quantize=4)) == 4


def test_refresh_estimate_follows_quantization(pool):
    entry = pool.get_entry("Krea-2-Turbo")
    default = entry.estimated_size
    pool.refresh_image_estimate(entry, ModelSettings(image_quantize=0))
    bf16 = entry.estimated_size
    pool.refresh_image_estimate(entry, ModelSettings(image_quantize=4))
    q4 = entry.estimated_size
    assert q4 < default < bf16
    assert bf16 - q4 > 10 * GB


def test_runtime_signature_includes_quantization(pool):
    a = pool._engine_runtime_signature("Krea-2-Turbo", ModelSettings(image_quantize=8))
    b = pool._engine_runtime_signature("Krea-2-Turbo", ModelSettings(image_quantize=4))
    assert a != b


async def test_load_engine_constructs_image_engine(pool, monkeypatch):
    started = []

    async def fake_start(self):
        started.append(self)

    monkeypatch.setattr(ImageGenEngine, "start", fake_start)
    engine = await pool.get_engine("Krea-2-Turbo")
    assert isinstance(engine, ImageGenEngine)
    assert engine.quantize == 8
    assert started == [engine]
