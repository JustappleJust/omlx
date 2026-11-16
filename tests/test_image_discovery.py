# SPDX-License-Identifier: Apache-2.0
"""Discovery tests for image-generation models."""

import json
from pathlib import Path

from omlx.model_discovery import (
    IMAGE_ACTIVATION_RESERVE_BYTES,
    detect_model_type,
    discover_models,
    estimate_model_size,
)


def _write_mflux_checkpoint(path: Path, base_model: str, *, quantize: int = 8) -> None:
    path.mkdir(parents=True)
    (path / "mflux_model.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_name": base_model,
                "base_model": base_model,
                "quantize": quantize,
            }
        )
    )
    for component in ("transformer", "text_encoder", "vae"):
        component_dir = path / component
        component_dir.mkdir()
        (component_dir / "0.safetensors").write_bytes(b"0" * 512)


def _write_diffusers_checkpoint(
    path: Path, class_name: str, name_or_path: str
) -> None:
    path.mkdir(parents=True)
    (path / "model_index.json").write_text(
        json.dumps({"_class_name": class_name, "_name_or_path": name_or_path})
    )
    transformer = path / "transformer"
    transformer.mkdir()
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"0" * 1024)


def test_discovers_mflux_saved_image_models(tmp_path):
    _write_mflux_checkpoint(tmp_path / "Krea-2-Turbo-mflux-q8", "krea-2")
    _write_mflux_checkpoint(
        tmp_path / "Qwen-Image-2.1-Turbo-mflux-q8", "qwen-image-2.1-turbo"
    )

    models = discover_models(tmp_path)

    krea = models["Krea-2-Turbo-mflux-q8"]
    assert krea.model_type == "image"
    assert krea.engine_type == "image"
    assert models["Qwen-Image-2.1-Turbo-mflux-q8"].model_type == "image"


def test_discovers_diffusers_image_checkpoint(tmp_path):
    _write_diffusers_checkpoint(
        tmp_path / "flux-3-image-klein",
        "Flux3ImageKleinPipeline",
        "black-forest-labs/flux-3-image-klein",
    )

    models = discover_models(tmp_path)

    assert models["flux-3-image-klein"].model_type == "image"
    assert models["flux-3-image-klein"].engine_type == "image"


def test_ignores_unrelated_diffusion_checkpoints(tmp_path):
    _write_diffusers_checkpoint(
        tmp_path / "sdxl-base",
        "StableDiffusionXLPipeline",
        "stabilityai/stable-diffusion-xl-base-1.0",
    )

    assert discover_models(tmp_path) == {}


def test_image_detection_does_not_shadow_text_models(tmp_path):
    model = tmp_path / "qwen3-4b"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
    )
    (model / "model.safetensors").write_bytes(b"0" * 64)

    assert detect_model_type(model) == "llm"


def test_detect_model_type_for_image_checkpoints(tmp_path):
    _write_mflux_checkpoint(tmp_path / "krea", "krea-2")
    assert detect_model_type(tmp_path / "krea") == "image"


def test_image_estimate_includes_activation_reserve(tmp_path):
    _write_mflux_checkpoint(tmp_path / "Krea-2-Turbo-mflux-q8", "krea-2")
    weights = sum(
        shard.stat().st_size
        for shard in (tmp_path / "Krea-2-Turbo-mflux-q8").rglob("*.safetensors")
    )

    estimate = estimate_model_size(tmp_path / "Krea-2-Turbo-mflux-q8")

    assert estimate == weights + IMAGE_ACTIVATION_RESERVE_BYTES
    assert estimate >= 2 * 1024**3


def test_hf_cache_image_snapshot_is_registered(tmp_path):
    repo = tmp_path / "models--Qwen--Qwen-Image-2.1-Turbo"
    snapshot = repo / "snapshots" / "0123456789abcdef"
    _write_diffusers_checkpoint(
        snapshot,
        "QwenImage21Pipeline",
        "Qwen/Qwen-Image-2.1-Turbo",
    )

    models = discover_models(tmp_path)

    image_entries = [m for m in models.values() if m.model_type == "image"]
    assert len(image_entries) == 1
    entry = image_entries[0]
    assert entry.engine_type == "image"
    assert entry.source_type == "hf_cache"
    assert entry.source_repo_id == "Qwen/Qwen-Image-2.1-Turbo"


def test_organization_folder_does_not_shadow_image_checkpoints(tmp_path):
    """A container folder named after the family is not a model itself."""
    child = tmp_path / "krea-2" / "Krea-2-Turbo-mflux-q8"
    _write_mflux_checkpoint(child, "krea-2")

    models = discover_models(tmp_path)

    assert list(models) == ["Krea-2-Turbo-mflux-q8"]
    assert models["Krea-2-Turbo-mflux-q8"].model_type == "image"
