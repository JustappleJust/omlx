# SPDX-License-Identifier: Apache-2.0
"""Discovery and sizing of image-generation (diffusers-layout) models."""

import json
from pathlib import Path

import pytest

from omlx.image_gen import OMLX_IMAGE_MARKER, find_adapter, is_image_model_dir
from omlx.image_gen.krea2 import Krea2Adapter
from omlx.model_discovery import (
    _is_hf_cache_mlx_compatible,
    _is_model_dir,
    detect_model_type,
    discover_models,
    estimate_image_model_size,
)

GB = 1024**3


def _sparse(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)


def make_krea2(root: Path, *, native=True, diffusers=True, te=8 * GB, vae=GB // 2) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "Krea2Pipeline", "transformer": ["diffusers", "Krea2Transformer2DModel"]})
    )
    if native:
        _sparse(root / "turbo.safetensors", 24 * GB)
    if diffusers:
        for i in range(3):
            _sparse(root / "transformer" / f"diffusion_pytorch_model-0000{i+1}-of-00003.safetensors", 8 * GB)
    _sparse(root / "text_encoder" / "model.safetensors", te)
    (root / "text_encoder" / "config.json").write_text("{}")
    _sparse(root / "vae" / "diffusion_pytorch_model.safetensors", vae)
    (root / "tokenizer").mkdir(exist_ok=True)
    return root


def test_detects_krea2_by_model_index(tmp_path):
    d = make_krea2(tmp_path / "Krea-2-Turbo")
    assert is_image_model_dir(d)
    assert isinstance(find_adapter(d), Krea2Adapter)
    assert _is_model_dir(d)
    assert detect_model_type(d) == "image"
    assert _is_hf_cache_mlx_compatible(d, "krea/Krea-2-Turbo")


def test_unknown_pipeline_is_not_image(tmp_path):
    d = tmp_path / "other"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({"_class_name": "SomeOtherPipeline"}))
    assert not is_image_model_dir(d)
    assert not _is_model_dir(d)


def test_marker_file_detects_resaved_dir(tmp_path):
    d = tmp_path / "krea-q8"
    _sparse(d / "0.safetensors", GB)
    (d / OMLX_IMAGE_MARKER).write_text(json.dumps({"adapter": "krea2"}))
    assert isinstance(find_adapter(d), Krea2Adapter)


def test_llm_dir_unaffected(tmp_path):
    d = tmp_path / "llm"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))
    assert detect_model_type(d) == "llm"


def test_size_counts_only_the_transformer_the_loader_uses(tmp_path):
    adapter = Krea2Adapter()
    d = make_krea2(tmp_path / "k", native=True, diffusers=True)
    files = adapter.component_files(d)
    assert [p.name for p in files["transformer"]] == ["turbo.safetensors"]
    bf16 = adapter.estimate_weight_bytes(d, None)
    assert bf16 == 24 * GB + 8 * GB + GB // 2  # duplicate shards not counted

    d2 = make_krea2(tmp_path / "k2", native=False, diffusers=True)
    assert len(adapter.component_files(d2)["transformer"]) == 3


def test_quantization_shrinks_only_the_transformer(tmp_path):
    adapter = Krea2Adapter()
    d = make_krea2(tmp_path / "k")
    q8 = adapter.estimate_weight_bytes(d, 8)
    q4 = adapter.estimate_weight_bytes(d, 4)
    te_vae = 8 * GB + GB // 2
    assert q8 == pytest.approx(te_vae + 24 * GB * 0.5 * 1.07, rel=1e-6)
    assert q4 < q8 < adapter.estimate_weight_bytes(d, None)


def test_estimate_image_model_size_default_is_8bit(tmp_path):
    d = make_krea2(tmp_path / "k")
    default = estimate_image_model_size(d)
    assert default == estimate_image_model_size(d, 8)
    assert estimate_image_model_size(d, 0) > default  # 0 = bf16


def test_memory_plans_ordered_fastest_first():
    a = Krea2Adapter()
    plans = a.memory_plans(2048, 2048)
    assert [p.name for p in plans] == ["full", "vae_tile_512", "vae_tile_256"]
    peaks = [p.peak_bytes for p in plans]
    assert peaks == sorted(peaks, reverse=True)
    # Calibrated: untiled 1 MP decode ~15.6 GB measured.
    assert 15e9 < a.estimate_activation_bytes(1024, 1024) < 19e9
    # No tiling plan smaller than the image itself.
    assert [p.name for p in a.memory_plans(512, 512)] == ["full", "vae_tile_256"]
    assert a.estimate_activation_bytes(512, 512) < a.estimate_activation_bytes(1024, 1024)


def test_normalize_size():
    a = Krea2Adapter()
    assert a.normalize_size(1000, 1030) == (992, 1024)
    assert a.normalize_size(10, 99999) == (256, 2048)


def test_discover_models_registers_image(tmp_path):
    make_krea2(tmp_path / "Krea-2-Turbo")
    models = discover_models(tmp_path)
    m = models["Krea-2-Turbo"]
    assert m.model_type == "image"
    assert m.engine_type == "image"
    assert m.estimated_size > 20 * GB


def _tiny_safetensors(path: Path, keys, metadata=None) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: np.zeros((1,), dtype=np.float16) for k in keys}, str(path), metadata=metadata)


KREA_KEYS = ["first.weight", "blocks.0.attn.wq.weight", "txtfusion.proj.weight", "txtmlp.fc.weight", "last.weight"]


def _components(d: Path) -> None:
    _tiny_safetensors(d / "text_encoder" / "model.safetensors", ["layers.0.w"])
    _tiny_safetensors(d / "vae" / "diffusion_pytorch_model.safetensors", ["decoder.w"])


def test_mflux_download_layout_without_model_index(tmp_path):
    # What `mflux` itself downloads for krea/Krea-2-Turbo (HF cache snapshot).
    d = tmp_path / "snap"
    _tiny_safetensors(d / "turbo.safetensors", KREA_KEYS)
    _components(d)
    assert isinstance(find_adapter(d), Krea2Adapter)
    assert detect_model_type(d) == "image"
    assert not Krea2Adapter().is_prequantized(d)


def test_mflux_saved_quantized_layout(tmp_path):
    d = tmp_path / "krea-2-turbo-mflux-q4"
    _tiny_safetensors(d / "0.safetensors", KREA_KEYS, {"quantization_level": "4", "mflux_version": "0.18.0"})
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "0.safetensors" for k in KREA_KEYS}})
    )
    _components(d)
    a = find_adapter(d)
    assert isinstance(a, Krea2Adapter)
    assert a.is_prequantized(d)
    assert a.incomplete_reason(d) is None
    # Pre-quantized weights are not scaled again.
    assert a.estimate_weight_bytes(d, 8) == a.estimate_weight_bytes(d, None)


def test_non_krea_layout_with_components_is_not_image(tmp_path):
    d = tmp_path / "x"
    _tiny_safetensors(d / "0.safetensors", ["model.layers.0.w"])
    _components(d)
    assert find_adapter(d) is None


def test_incomplete_download_is_skipped(tmp_path):
    d = tmp_path / "models" / "krea-q8"
    (d / "vae").mkdir(parents=True)
    (d / "text_encoder").mkdir()
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "0.safetensors" for k in KREA_KEYS}})
    )
    a = find_adapter(d)
    assert isinstance(a, Krea2Adapter)
    assert a.incomplete_reason(d) == "no transformer weights"
    assert "krea-q8" not in discover_models(tmp_path / "models")

    _tiny_safetensors(d / "0.safetensors", KREA_KEYS)
    _components(d)
    (d / "vae" / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "1.safetensors"}}))
    assert a.incomplete_reason(d) == "missing shard vae/1.safetensors"


def make_klein(root: Path, flow_map: bool = True) -> Path:
    (root / "transformer").mkdir(parents=True)
    (root / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "Flux3Transformer2DModel", "flow_map": flow_map})
    )
    _sparse(root / "transformer" / "diffusion_pytorch_model-00001-of-00001.safetensors", 9 * GB)
    _sparse(root / "text_encoder" / "model-00001-of-00001.safetensors", 8 * GB)
    _sparse(root / "vae" / "diffusion_pytorch_model.safetensors", GB // 4)
    return root


def make_qwen21(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImage21Pipeline", "sample_sigmas": [1.0, 0.5]})
    )
    _sparse(root / "transformer" / "diffusion_pytorch_model-00001-of-00001.safetensors", 14 * GB)
    _sparse(root / "text_encoder" / "model-00001-of-00001.safetensors", 16 * GB)
    _sparse(root / "vae" / "diffusion_pytorch_model.safetensors", GB // 2)
    return root


def test_detects_flux3_klein_without_model_index(tmp_path):
    from omlx.image_gen.flux3 import Flux3KleinAdapter

    d = make_klein(tmp_path / "flux-3-image-klein")
    a = find_adapter(d)
    assert isinstance(a, Flux3KleinAdapter)
    assert detect_model_type(d) == "image"
    assert a.capabilities.edit and not a.capabilities.mask
    assert a._flow_map(d)
    # only the DiT is scaled by load-time quantization (conservative)
    assert a.estimate_weight_bytes(d, 8) == pytest.approx(8 * GB + GB // 4 + 9 * GB * 0.5 * 1.07, rel=1e-6)


def test_detects_qwen_image21(tmp_path):
    from omlx.image_gen.qwen21 import QwenImage21Adapter

    d = make_qwen21(tmp_path / "Qwen-Image-2.1-Turbo")
    a = find_adapter(d)
    assert isinstance(a, QwenImage21Adapter)
    assert a._is_turbo(d)
    assert a.capabilities.mask and "step_cache" in a.capabilities.lossy_accel
    assert a.defaults.size_multiple == 32
    assert a.normalize_size(1000, 1000) == (992, 992)


def test_edit_plans_grow_with_references(tmp_path):
    from omlx.image_gen.flux3 import Flux3KleinAdapter

    a = Flux3KleinAdapter()
    assert a.estimate_activation_bytes(1024, 1024, 2) > a.estimate_activation_bytes(1024, 1024, 0)


def test_all_three_models_discovered(tmp_path):
    make_krea2(tmp_path / "Krea-2-Turbo")
    make_klein(tmp_path / "flux-3-image-klein")
    make_qwen21(tmp_path / "Qwen-Image-2.1-Turbo")
    models = discover_models(tmp_path)
    assert {k: v.model_type for k, v in models.items()} == {
        "Krea-2-Turbo": "image",
        "flux-3-image-klein": "image",
        "Qwen-Image-2.1-Turbo": "image",
    }


def test_memory_plan_scales_with_batch_rows():
    plans = Krea2Adapter().memory_plans(1024, 1024)
    assert plans[0].peak_for(1) == plans[0].peak_bytes
    assert plans[0].extra_row_bytes > 0
    assert plans[0].peak_for(3) - plans[0].peak_for(2) == plans[0].extra_row_bytes


def test_only_krea2_and_flux3_support_batching():
    from omlx.image_gen.base import ImageModelAdapter
    from omlx.image_gen.flux3 import Flux3KleinAdapter
    from omlx.image_gen.qwen21 import QwenImage21Adapter

    assert Krea2Adapter.supports_batching is True
    assert Flux3KleinAdapter.supports_batching is True
    assert QwenImage21Adapter.supports_batching is False
    assert ImageModelAdapter.supports_batching is False


def test_default_generate_batch_runs_each_row_in_order():
    from omlx.image_gen.base import (
        ImageDefaults,
        ImageGenParams,
        ImageModelAdapter,
        MemoryPlan,
    )

    class Recorder(ImageModelAdapter):
        name = "recorder"
        defaults = ImageDefaults(steps=1, guidance=1.0)

        def component_files(self, model_dir):
            return {}

        def memory_plans(self, width, height, n_refs=0):
            return [MemoryPlan("full", 1)]

        def load(self, model_dir, quantize, options=None):
            return object()

        def generate(self, model, params, on_step, plan=None):
            self.seen.append(params.prompt)
            return params.prompt

    recorder = Recorder()
    recorder.seen = []
    rows = [
        ImageGenParams(prompt="a", width=512, height=512, steps=1, guidance=1.0, seed=1),
        ImageGenParams(prompt="b", width=512, height=512, steps=1, guidance=1.0, seed=2),
    ]
    out = recorder.generate_batch(object(), rows, [lambda *_: None, lambda *_: None])
    assert out == ["a", "b"]
    assert recorder.seen == ["a", "b"]
