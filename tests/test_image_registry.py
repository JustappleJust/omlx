# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the image-model registry and its request validation."""

import json

import pytest

from omlx.image_models import (
    IMAGE_MODELS,
    ImageModelSpec,
    detect_image_spec,
    get_spec,
    is_mflux_checkpoint,
    list_specs,
    looks_like_image_model_dir,
    match_image_spec,
    normalize_model_name,
    parse_image_size,
    resolve_defaults,
    resolve_dimensions,
)


@pytest.mark.parametrize(
    ("name", "expected_key"),
    [
        # Directory names, as discovered by the pool.
        ("Krea-2-Turbo-mflux-q8", "krea-2"),
        ("krea2_turbo_fp8_scaled", "krea-2"),
        ("Qwen-Image-2.1-Turbo-mflux-q8", "qwen-image-2.1-turbo"),
        ("Qwen-Image-2.1", "qwen-image-2.1"),
        ("FLUX-3-Image-Klein-mflux-bf16", "flux3-image-klein"),
        ("flux-3-image-klein-base", "flux3-image-klein-base"),
        # mflux registry keys.
        ("krea-2", "krea-2"),
        ("qwen-image-2.1-turbo", "qwen-image-2.1-turbo"),
        ("flux3-image-klein", "flux3-image-klein"),
        # Diffusers pipeline / component class names.
        ("Krea2Pipeline", "krea-2"),
        ("Flux3ImageKleinPipeline", "flux3-image-klein"),
        ("QwenImage21Pipeline", "qwen-image-2.1"),
        ("Flux3Transformer2DModel", "flux3-image-klein"),
    ],
)
def test_registry_matches_supported_families(name, expected_key):
    spec = match_image_spec(name)
    assert spec is not None, name
    assert spec.key == expected_key


@pytest.mark.parametrize(
    "name",
    [
        "llama-3-8b",
        "stable-diffusion-xl-base-1.0",
        "StableDiffusionXLPipeline",
        "flux1-dev",
        "krea-dev",  # legacy Krea dev is FLUX.1-based, not Krea 2
        "qwen-image-edit",  # Qwen-Image 1.x edit
        "",
    ],
)
def test_registry_rejects_unrelated_models(name):
    assert match_image_spec(name) is None


def test_normalize_model_name_collapses_separators():
    assert normalize_model_name("Qwen_Image.2-1") == "qwen-image-2-1"
    assert normalize_model_name("Krea/2") == "krea-2"


def test_registry_contains_the_three_requested_checkpoints():
    ids = [spec.model_id for spec in list_specs()]
    assert {
        "flux-3-image-klein",
        "krea-2-turbo",
        "qwen-image-2.1-turbo",
    }.issubset(set(ids))
    assert all(spec.repo_id and spec.defaults.steps > 0 for spec in list_specs())


def test_get_spec_accepts_ids_and_aliases():
    assert get_spec("qwen-image-2.1-turbo") is IMAGE_MODELS["qwen-image-2.1-turbo"]
    assert get_spec("krea2") is IMAGE_MODELS["krea-2-turbo"]
    assert get_spec("not-a-model") is None


def test_detect_image_spec_from_mflux_metadata(tmp_path):
    (tmp_path / "mflux_model.json").write_text(
        json.dumps(
            {
                "base_model": "qwen-image-2.1-turbo",
                "model_name": "Qwen/Qwen-Image-2.1-Turbo",
                "quantize": 8,
            }
        )
    )
    spec = detect_image_spec(tmp_path)
    assert spec is not None and spec.key == "qwen-image-2.1-turbo"
    assert looks_like_image_model_dir(tmp_path)


def test_detect_image_spec_from_saved_model_name(tmp_path):
    # A save of a third-party repo records only the upstream repo id.
    (tmp_path / "mflux_model.json").write_text(
        json.dumps({"model_name": "krea/Krea-2-Turbo"})
    )
    spec = detect_image_spec(tmp_path)
    assert spec is not None and spec.key == "krea-2"


def test_detect_image_spec_from_diffusers_index(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux3ImageKleinPipeline",
                "_name_or_path": "black-forest-labs/flux-3-image-klein",
                "transformer": ["diffusers", "Flux3Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
    )
    spec = detect_image_spec(tmp_path)
    assert spec is not None and spec.key == "flux3-image-klein"


def test_detect_image_spec_distinguishes_turbo_from_base(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImage21Pipeline",
                "_name_or_path": "Qwen/Qwen-Image-2.1-Turbo",
            }
        )
    )
    spec = detect_image_spec(tmp_path)
    assert spec is not None and spec.key == "qwen-image-2.1-turbo"


def test_detect_image_spec_from_directory_name(tmp_path):
    model = tmp_path / "Krea-2-Turbo-mflux-q8"
    (model / "transformer").mkdir(parents=True)
    (model / "transformer" / "0.safetensors").write_bytes(b"0" * 16)
    spec = detect_image_spec(model)
    assert spec is not None and spec.key == "krea-2"


def test_organization_folder_is_not_a_model(tmp_path):
    # A container named after the family must not shadow its checkpoints.
    parent = tmp_path / "krea-2"
    child = parent / "Krea-2-Turbo-mflux-q8"
    (child / "transformer").mkdir(parents=True)
    (child / "transformer" / "0.safetensors").write_bytes(b"0" * 16)

    assert detect_image_spec(parent) is None
    assert not looks_like_image_model_dir(parent)
    assert detect_image_spec(child) is not None


def test_detect_image_spec_ignores_unsupported_diffusion(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"})
    )
    assert detect_image_spec(tmp_path) is None
    assert not looks_like_image_model_dir(tmp_path)


def test_is_mflux_checkpoint(tmp_path):
    assert not is_mflux_checkpoint(tmp_path)
    component = tmp_path / "transformer"
    component.mkdir()
    (component / "model.safetensors.index.json").write_text("{}")
    assert is_mflux_checkpoint(tmp_path)
    (tmp_path / "mflux_model.json").write_text("{}")
    assert is_mflux_checkpoint(tmp_path)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (None, (None, None)),
        ("auto", (None, None)),
        ("AUTO", (None, None)),
        ("1024x1024", (1024, 1024)),
        ("1536x1024", (1536, 1024)),
        ("1024×768", (1024, 768)),
    ],
)
def test_parse_image_size(size, expected):
    assert parse_image_size(size) == expected


@pytest.mark.parametrize("size", ["1024", "big", "1024x", "0x1024", "-16x16"])
def test_parse_image_size_rejects_bad_values(size):
    with pytest.raises(ValueError):
        parse_image_size(size)


def test_resolve_dimensions_uses_family_defaults():
    krea = IMAGE_MODELS["krea-2-turbo"]
    assert resolve_dimensions(krea, None, None) == (1024, 1024)
    assert resolve_dimensions(krea, 2048, 1024) == (2048, 1024)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (1000, 1024),  # not a multiple of 16
        (1024, None),  # half a size
        (None, 1024),
        (64, 64),  # below the minimum
        (8192, 1024),  # above the maximum
    ],
)
def test_resolve_dimensions_rejects_bad_sizes(width, height):
    with pytest.raises(ValueError):
        resolve_dimensions(IMAGE_MODELS["krea-2-turbo"], width, height)


def test_vendor_defaults_per_family():
    assert resolve_defaults(IMAGE_MODELS["krea-2-turbo"], None, None) == (8, 1.0)
    assert resolve_defaults(IMAGE_MODELS["qwen-image-2.1-turbo"], None, None) == (
        8,
        1.0,
    )
    assert resolve_defaults(IMAGE_MODELS["qwen-image-2.1"], None, None) == (40, 1.0)
    assert resolve_defaults(IMAGE_MODELS["flux-3-image-klein"], None, None) == (
        16, None,
    )
    assert resolve_defaults(IMAGE_MODELS["flux-3-image-klein-base"], None, None) == (
        50,
        3.5,
    )


def test_request_overrides_defaults():
    spec: ImageModelSpec = IMAGE_MODELS["krea-2-turbo"]
    assert resolve_defaults(spec, 12, 2.5) == (12, 2.5)


def test_flux3_rejects_guidance():
    with pytest.raises(ValueError):
        resolve_defaults(IMAGE_MODELS["flux-3-image-klein"], None, 3.0)


def test_family_capabilities():
    krea = IMAGE_MODELS["krea-2-turbo"]
    assert krea.supports_img2img and not krea.supports_edit
    flux3 = IMAGE_MODELS["flux-3-image-klein"]
    assert flux3.supports_edit and flux3.max_references == 10
    qwen = IMAGE_MODELS["qwen-image-2.1-turbo"]
    assert qwen.supports_edit and qwen.outputs_alpha
