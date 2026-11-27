# SPDX-License-Identifier: Apache-2.0
"""Model-family adapter interface for image-generation models.

An adapter knows how to recognise one family of image models on disk, how to
estimate its memory footprint, and how to load and run it. The engine
(:mod:`omlx.engine.image_gen`) is family-agnostic and only talks to this
interface, so supporting a new model family means adding one adapter module
and registering it in :mod:`omlx.image_gen.registry`.

Adapters must keep heavy imports (mflux, mlx model code) inside ``load`` so
discovery works when the optional ``omlx[image]`` extra is not installed.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Marker file an operator may drop into a converted / re-saved model directory
# (e.g. the output of ``mflux-save``) that has no ``model_index.json``.
# Content: ``{"adapter": "krea2"}``.
OMLX_IMAGE_MARKER = "omlx_image.json"


class ImageGenerationCancelled(Exception):
    """Raised from the per-step callback to abort a running generation."""


@dataclass
class ImageGenParams:
    """Family-agnostic generation parameters (already validated)."""

    prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    negative_prompt: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Editing: reference images (PIL, RGB/RGBA), optional mask (white =
    # repaint) and strength (1.0 = start from noise).
    references: list[Any] = field(default_factory=list)
    mask: Any = None
    strength: float = 1.0
    #: Opt-in lossy accelerations for this request, e.g. {"step_cache": True}.
    accel: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageDefaults:
    steps: int
    guidance: float
    width: int = 1024
    height: int = 1024
    size_multiple: int = 16
    min_side: int = 256
    max_side: int = 2048


@dataclass(frozen=True)
class ImageCapabilities:
    generate: bool = True
    edit: bool = False
    max_references: int = 0
    mask: bool = False
    strength: bool = False
    #: Opt-in lossy accelerations this family implements (e.g. "step_cache").
    lossy_accel: tuple[str, ...] = ()
    negative_prompt: bool = True

    def as_dict(self) -> dict:
        return {
            "generate": self.generate,
            "edit": self.edit,
            "max_references": self.max_references,
            "mask": self.mask,
            "strength": self.strength,
            "lossy_accel": list(self.lossy_accel),
            "negative_prompt": self.negative_prompt,
        }


@dataclass(frozen=True)
class MemoryPlan:
    """One way to execute a generation, with its expected transient peak.

    ``options`` is opaque to the engine and handed back to
    :meth:`ImageModelAdapter.generate` (e.g. VAE tile size).
    """

    name: str
    peak_bytes: int
    options: dict = field(default_factory=dict)
    #: Extra transient bytes for each additional row in a static batch.
    extra_row_bytes: int = 0

    def peak_for(self, batch_size: int) -> int:
        """Transient peak when ``batch_size`` requests share the pass."""
        return self.peak_bytes + self.extra_row_bytes * (max(1, int(batch_size)) - 1)


# (step_index_1_based, total_steps) -> None. May raise ImageGenerationCancelled.
StepCallback = Callable[[int, int], None]


def read_model_index(model_dir: Path) -> Optional[dict]:
    """Return the parsed ``model_index.json`` (or omlx marker) or None."""
    for name in ("model_index.json", OMLX_IMAGE_MARKER):
        path = model_dir / name
        if path.is_file():
            try:
                with open(path) as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else None
            except (OSError, json.JSONDecodeError):
                return None
    return None


def _sum_files(paths) -> int:
    total = 0
    for p in paths:
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


class ImageModelAdapter(ABC):
    """One image-model family."""

    #: Short stable identifier (used in the omlx_image.json marker).
    name: str = ""
    #: diffusers ``_class_name`` values handled by this adapter.
    pipeline_classes: tuple[str, ...] = ()
    defaults: ImageDefaults
    capabilities: ImageCapabilities = ImageCapabilities()
    #: Whether ``generate_batch`` runs compatible requests in one denoise pass.
    supports_batching: bool = False

    # ---- detection / sizing (must stay import-light) -----------------

    def matches(self, model_dir: Path, model_index: dict) -> bool:
        if model_index.get("adapter") == self.name:
            return True
        return model_index.get("_class_name") in self.pipeline_classes

    @abstractmethod
    def component_files(self, model_dir: Path) -> dict[str, list[Path]]:
        """Weight files the loader will actually read, per component.

        Used for memory estimation; must mirror the loader's file selection so
        redundant copies shipped in the repo are not double counted.
        """

    def quantizable_components(self) -> set[str]:
        """Components whose weights shrink under load-time quantization."""
        return {"transformer"}

    def estimate_weight_bytes(self, model_dir: Path, quantize: Optional[int]) -> int:
        """Resident weight bytes after load (with load-time quantization).

        Pre-quantized checkpoints are already small on disk; quantizing them
        again is a no-op, so only scale float checkpoints.
        """
        files = self.component_files(model_dir)
        prequantized = self.is_prequantized(model_dir)
        total = 0.0
        for comp, paths in files.items():
            size = _sum_files(paths)
            if (
                quantize
                and not prequantized
                and comp in self.quantizable_components()
            ):
                # bf16 -> N bits plus ~1/16 overhead for scales/biases (group 64).
                size = size * (quantize / 16.0) * 1.07
            total += size
        return int(total)

    def is_prequantized(self, model_dir: Path) -> bool:
        return False

    def required_components(self) -> tuple[str, ...]:
        return ("transformer", "text_encoder", "vae")

    def incomplete_reason(self, model_dir: Path) -> Optional[str]:
        """Why the checkpoint cannot load yet (interrupted download), or None.

        Checks every required component has weights and that every shard a
        ``*.safetensors.index.json`` references exists.
        """
        files = self.component_files(model_dir)
        for comp in self.required_components():
            if not files.get(comp):
                return f"no {comp} weights"
        dirs = {p.parent for paths in files.values() for p in paths}
        dirs.add(model_dir)
        for d in dirs:
            for index in d.glob("*.safetensors.index.json"):
                try:
                    weight_map = json.loads(index.read_text()).get("weight_map") or {}
                except (OSError, ValueError, AttributeError):
                    continue
                for shard in set(weight_map.values()):
                    if not (d / shard).is_file():
                        rel = (d / shard).relative_to(model_dir)
                        return f"missing shard {rel}"
        return None

    @abstractmethod
    def memory_plans(
        self, width: int, height: int, n_refs: int = 0
    ) -> list[MemoryPlan]:
        """Execution plans for one generation, fastest (largest peak) first.

        ``n_refs`` reference images (editing) lengthen the token sequence.
        The engine runs the first plan whose transient peak fits the memory
        budget left under the enforcer's soft limit, and refuses the request
        when none fits.
        """

    def estimate_activation_bytes(self, width: int, height: int, n_refs: int = 0) -> int:
        """Peak transient memory of the preferred (fastest) plan."""
        return self.memory_plans(width, height, n_refs)[0].peak_bytes

    def edit_canvas(self, references: list[Any]) -> tuple[int, int]:
        """Default output size for an edit without an explicit size."""
        w, h = references[0].size
        return self.normalize_size(w, h)

    # ---- runtime -------------------------------------------------------

    @abstractmethod
    def load(
        self, model_dir: Path, quantize: Optional[int], options: Optional[dict] = None
    ) -> Any:
        """Load and return the runtime model (called on the MLX executor).

        ``options`` carries engine-level tuning (e.g. ``{"a8": True}``).
        """

    def runtime_stats(self, model: Any) -> dict:
        """Family-specific stats (cache sizes, acceleration status)."""
        return {}

    @abstractmethod
    def generate(
        self,
        model: Any,
        params: ImageGenParams,
        on_step: StepCallback,
        plan: Optional[MemoryPlan] = None,
    ) -> "Any":
        """Run one generation and return a ``PIL.Image.Image``.

        Called on the MLX executor. ``plan`` is one of :meth:`memory_plans`
        (None = the preferred plan).
        """

    def generate_batch(
        self,
        model: Any,
        params_list: list[ImageGenParams],
        on_step_list: list[StepCallback],
        plan: Optional[MemoryPlan] = None,
    ) -> list["Any"]:
        """Run compatible requests, returning one image per row (None = cancelled).

        The default keeps the existing per-request path. Adapters that set
        :attr:`supports_batching` may merge the rows into one denoise pass;
        they must still return the results in input order.
        """
        return [
            self.generate(model, params, on_step, plan)
            for params, on_step in zip(params_list, on_step_list)
        ]

    def release(self, model: Any) -> None:
        """Free family-specific caches before the model is dropped."""

    # ---- helpers -------------------------------------------------------

    def normalize_size(self, width: int, height: int) -> tuple[int, int]:
        d = self.defaults
        m = d.size_multiple

        def _fix(v: int) -> int:
            v = max(d.min_side, min(d.max_side, int(v)))
            return max(m, (v // m) * m)

        return _fix(width), _fix(height)
