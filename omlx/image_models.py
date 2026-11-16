# SPDX-License-Identifier: Apache-2.0
"""Image-generation model registry for oMLX.

oMLX serves image models through `mflux <https://github.com/mflux-community/mflux>`_,
a native MLX port of the model families below. This module is the single
source of truth for:

* which image families oMLX can serve,
* the per-family generation defaults (steps, guidance, canvas size),
* what each family can do (reference editing, img2img, alpha output), and
* how a model directory on disk is recognised as one of those families.

It deliberately imports nothing outside the standard library: model discovery
runs before MLX/mflux are needed, and ``model_discovery`` must stay importable
on a machine without the optional video/image runtime installed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory/file markers written by ``mflux-save``.
MFLUX_METADATA_FILE = "mflux_model.json"
DIFFUSERS_INDEX_FILE = "model_index.json"

# Dimension rules shared by the supported families: every latent grid is
# 16 px per token, so width and height must be multiples of 16.
DEFAULT_SIZE_MULTIPLE = 16
DEFAULT_MIN_SIZE = 256
DEFAULT_MAX_SIZE = 4096


@dataclass
class ImageRequest:
    """Sampling parameters for one ``/v1/images/*`` request.

    ``width``/``height``/``steps``/``guidance`` are ``None`` when the caller
    omitted them, which lets the family defaults fill the gaps.
    """

    prompt: str
    n: int = 1
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    negative_prompt: str | None = None
    strength: float | None = None
    output_format: str = "png"
    #: OpenAI ``output_compression`` (0-100); jpeg/webp quality or png level.
    output_compression: int | None = None


@dataclass
class ImageResult:
    """Encoded images plus the sampling details that produced them."""

    images: list[bytes]
    mime_type: str
    model_id: str
    seed: int
    width: int
    height: int
    steps: int
    guidance: float | None
    mode: str
    elapsed_seconds: float
    extra: dict = field(default_factory=dict)


class ImageGenerationError(Exception):
    """Base class for image-generation failures the API can classify."""


class ImageBackendUnavailable(ImageGenerationError):  # noqa: N818
    """The mflux runtime (or a required mflux module) is not installed."""


class ImageModelUnsupported(ImageGenerationError):  # noqa: N818
    """The model directory is not a registered image family."""


class ImageRequestError(ImageGenerationError):
    """The request asks for something the family cannot do."""


@dataclass(frozen=True)
class ImageDefaults:
    """Vendor-recommended sampling defaults for one checkpoint.

    These are the values the model card (and mflux's CLI) use when the caller
    does not ask for anything specific. Requests may override steps, guidance,
    and the canvas size one field at a time.
    """

    steps: int
    guidance: float | None
    width: int = 1024
    height: int = 1024
    #: True when the pipeline accepts a negative prompt as the CFG
    #: unconditional branch. It only has an effect while guidance > 1.
    negative_prompt: bool = False
    #: Default strength for img2img edits, when the family supports them.
    image_strength: float | None = None


@dataclass(frozen=True)
class ImageModelSpec:
    """Registry entry for one supported image checkpoint."""

    #: mflux registry key (``ModelConfig.<method>()`` and ``--model`` values).
    key: str
    #: Runtime adapter id: ``flux3``, ``krea2`` or ``qwen21``.
    family: str
    #: Canonical, human-facing model id (also what the docs call the model).
    model_id: str
    display_name: str
    #: Upstream Hugging Face repo the checkpoint comes from.
    repo_id: str
    #: ``ModelConfig`` factory on the mflux side.
    config_method: str
    defaults: ImageDefaults
    #: Names that identify this checkpoint in directory names, mflux metadata,
    #: and diffusers ``model_index.json`` files. Matching is substring-based,
    #: longest alias first.
    aliases: tuple[str, ...] = ()
    #: Instruction-based / multi-reference editing in one model.
    supports_edit: bool = False
    #: Strength-based img2img on a single reference image.
    supports_img2img: bool = False
    #: Maximum reference images accepted by ``/v1/images/edits``.
    max_references: int = 1
    #: True when the family can emit an alpha channel (RGBA PNG/WebP).
    outputs_alpha: bool = False
    #: True when the family renders at a fixed output resolution instead of an
    #: arbitrary width/height pair.
    size_multiple: int = DEFAULT_SIZE_MULTIPLE
    min_size: int = DEFAULT_MIN_SIZE
    max_size: int = DEFAULT_MAX_SIZE
    license_name: str = ""
    notes: str = ""

    def matches(self, text: str) -> bool:
        """Return True when ``text`` names this checkpoint specifically."""
        normalized = normalize_model_name(text)
        return any(normalize_model_name(alias) in normalized for alias in self.aliases)

    @property
    def size_hint(self) -> str:
        """Human-readable canvas summary for error messages and docs."""
        return f"{self.defaults.width}x{self.defaults.height}"


# ---------------------------------------------------------------------------
# Supported families
# ---------------------------------------------------------------------------
#
# Defaults below are the vendor recommendations:
#
# * FLUX 3 Image Klein (distilled): 16 steps, no guidance.
# * FLUX 3 Image Klein Base: 50 steps, guidance 3.5.
# * Krea 2 Turbo: 8 steps, guidance 1.0, 1024² to 2048².
# * Qwen-Image-2.1-Turbo: 8 steps, guidance 1.0, 1024²+.
# * Qwen-Image-2.1: 40 steps, guidance 1.0.
#
# The three checkpoints to lead with are the distilled FLUX 3 Klein, Krea 2
# Turbo, and Qwen-Image-2.1-Turbo; the undistilled checkpoints of the same
# families are served through the identical runtime path and are listed so a
# user who downloads one gets sensible defaults instead of a rejection.

_FLUX3_ALIASES = (
    "flux-3-image-klein",
    "flux3-image-klein",
    "flux3imageklein",
    "flux3-klein",
    "flux-3-klein",
    "flux3",
)
_FLUX3_BASE_ALIASES = (
    "flux-3-image-klein-base",
    "flux3-image-klein-base",
    "flux3imagekleinbase",
    "flux3-klein-base",
)
_KREA2_ALIASES = (
    "krea-2-turbo",
    "krea2-turbo",
    "krea2turbo",
    "krea-2",
    "krea2",
)
_QWEN21_TURBO_ALIASES = (
    "qwen-image-2.1-turbo",
    "qwen-image-2-1-turbo",
    "qwen-image-21-turbo",
    "qwenimage21turbo",
    "qwen21-turbo",
    "qwen-image-turbo-21",
)
_QWEN21_ALIASES = (
    "qwen-image-2.1",
    "qwen-image-2-1",
    "qwen-image-21",
    "qwenimage21",
    "qwen21",
)

IMAGE_MODELS: dict[str, ImageModelSpec] = {
    "flux-3-image-klein": ImageModelSpec(
        key="flux3-image-klein",
        family="flux3",
        model_id="flux-3-image-klein",
        display_name="FLUX 3 Image Klein",
        repo_id="black-forest-labs/flux-3-image-klein",
        config_method="flux3_image_klein",
        defaults=ImageDefaults(steps=16, guidance=None, width=1024, height=1024),
        aliases=_FLUX3_ALIASES,
        supports_edit=True,
        supports_img2img=True,
        max_references=10,
        license_name="FLUX Kommunity License v1.0",
        notes=(
            "Distilled checkpoint: 16 steps and no classifier-free guidance. "
            "Editing takes up to 10 reference images."
        ),
    ),
    "flux-3-image-klein-base": ImageModelSpec(
        key="flux3-image-klein-base",
        family="flux3",
        model_id="flux-3-image-klein-base",
        display_name="FLUX 3 Image Klein Base",
        repo_id="black-forest-labs/flux-3-image-klein-base",
        config_method="flux3_image_klein_base",
        defaults=ImageDefaults(steps=50, guidance=3.5, width=1024, height=1024),
        aliases=_FLUX3_BASE_ALIASES,
        supports_edit=True,
        supports_img2img=True,
        max_references=10,
        license_name="FLUX Kommunity License v1.0",
        notes=(
            "Undistilled checkpoint for fine-tuning and research: 50 steps "
            "with guidance 3.5."
        ),
    ),
    "krea-2-turbo": ImageModelSpec(
        key="krea-2",
        family="krea2",
        model_id="krea-2-turbo",
        display_name="Krea 2 Turbo",
        repo_id="krea/Krea-2-Turbo",
        config_method="krea2",
        defaults=ImageDefaults(
            steps=8,
            guidance=1.0,
            width=1024,
            height=1024,
            negative_prompt=True,
            image_strength=0.65,
        ),
        aliases=_KREA2_ALIASES,
        supports_img2img=True,
        license_name="Krea 2 Community License",
        notes=(
            "Timestep-distilled checkpoint: 8 steps with guidance 1.0. "
            "Text-to-image and strength-based img2img; no instruction editing."
        ),
    ),
    "qwen-image-2.1-turbo": ImageModelSpec(
        key="qwen-image-2.1-turbo",
        family="qwen21",
        model_id="qwen-image-2.1-turbo",
        display_name="Qwen-Image-2.1-Turbo",
        repo_id="Qwen/Qwen-Image-2.1-Turbo",
        config_method="qwen_image_21_turbo",
        defaults=ImageDefaults(
            steps=8,
            guidance=1.0,
            width=1024,
            height=1024,
            negative_prompt=True,
            image_strength=1.0,
        ),
        aliases=_QWEN21_TURBO_ALIASES,
        supports_edit=True,
        supports_img2img=True,
        max_references=10,
        outputs_alpha=True,
        license_name="Qwen Research License Agreement",
        notes=(
            "8-step checkpoint of Qwen-Image-2.1 with multi-reference editing "
            "and RGBA output. Use 1024 px or larger."
        ),
    ),
    "qwen-image-2.1": ImageModelSpec(
        key="qwen-image-2.1",
        family="qwen21",
        model_id="qwen-image-2.1",
        display_name="Qwen-Image-2.1",
        repo_id="Qwen/Qwen-Image-2.1",
        config_method="qwen_image_21",
        defaults=ImageDefaults(
            steps=40,
            guidance=1.0,
            width=1024,
            height=1024,
            negative_prompt=True,
            image_strength=1.0,
        ),
        aliases=_QWEN21_ALIASES,
        supports_edit=True,
        supports_img2img=True,
        max_references=10,
        outputs_alpha=True,
        license_name="Qwen Research License Agreement",
        notes=(
            "Undistilled checkpoint: 40 steps without guidance. Same editing "
            "and RGBA capabilities as Turbo, roughly five times slower."
        ),
    ),
}

_SPLIT_RE = re.compile(r"[\s_.]+")


def normalize_model_name(value: str) -> str:
    """Normalise a directory/repo/class name for substring matching.

    ``"Qwen-Image-2.1-Turbo"``, ``"qwen_image_2_1_turbo"`` and
    ``"QwenImage21Pipeline"`` all collapse to strings that contain the
    registry aliases (``qwen-image-2-1-turbo`` / ``qwenimage21``).
    """
    return _SPLIT_RE.sub("-", value.strip().lower()).replace("/", "-")


def normalize_key(value: str) -> str:
    """Normalise an mflux registry key (``qwen-image-2.1-turbo``)."""
    return value.strip().lower().replace("_", "-")


#: model_id -> spec, plus alias lookups built once at import time.
_SPEC_BY_ALIAS: dict[str, ImageModelSpec] = {}
for _spec in IMAGE_MODELS.values():
    for _alias in _spec.aliases:
        _SPEC_BY_ALIAS[normalize_key(_alias)] = _spec
_ALIASES_BY_LENGTH: tuple[tuple[str, ImageModelSpec], ...] = tuple(
    sorted(_SPEC_BY_ALIAS.items(), key=lambda item: len(item[0]), reverse=True)
)

def match_image_spec(text: str | None) -> ImageModelSpec | None:
    """Return the registry spec named by ``text``, or None.

    Matching is deliberately narrow: only the families registered above are
    recognised, so an unrelated diffusion checkpoint in the model directory is
    not advertised as servable. Longest alias wins, so
    ``flux-3-image-klein-base`` is not mistaken for the distilled checkpoint.
    """
    if not text:
        return None
    normalized = normalize_model_name(text)
    for alias, spec in _ALIASES_BY_LENGTH:
        if alias in normalized:
            return spec
    return None


def get_spec(model_id_or_name: str) -> ImageModelSpec | None:
    """Look a model up by canonical id, mflux key, or alias."""
    if not model_id_or_name:
        return None
    if model_id_or_name in IMAGE_MODELS:
        return IMAGE_MODELS[model_id_or_name]
    return match_image_spec(model_id_or_name)


def list_specs() -> tuple[ImageModelSpec, ...]:
    """Return every registered spec, in registry order."""
    return tuple(IMAGE_MODELS.values())


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _diffusers_hints(model_path: Path) -> list[str]:
    """Collect identifying strings from a diffusers ``model_index.json``."""
    index = _read_json(model_path / DIFFUSERS_INDEX_FILE)
    if not index:
        return []
    hints: list[str] = []
    for key in ("_name_or_path", "_class_name"):
        value = index.get(key)
        if isinstance(value, str):
            hints.append(value)
    # Component entries look like ["diffusers", "Flux3Transformer2DModel"].
    # Their class names carry the family even when _class_name is generic.
    for value in index.values():
        if isinstance(value, list) and len(value) == 2 and isinstance(value[1], str):
            hints.append(value[1])
    return hints


def _mflux_hints(model_path: Path) -> list[str]:
    """Collect identifying strings from an ``mflux-save`` checkpoint."""
    info = _read_json(model_path / MFLUX_METADATA_FILE)
    if not info:
        return []
    hints: list[str] = []
    for key in ("base_model", "model_name"):
        value = info.get(key)
        if isinstance(value, str):
            hints.append(value)
    upstream = info.get("upstream")
    if isinstance(upstream, dict):
        repo_id = upstream.get("repo_id")
        if isinstance(repo_id, str):
            hints.append(repo_id)
    return hints


def _config_hints(model_path: Path) -> list[str]:
    """Collect identifying strings from a top-level ``config.json``.

    Component checkpoints (the text encoder, for example) also carry
    ``config.json`` files, but they never identify the whole pipeline, so only
    the top level is read.
    """
    config = _read_json(model_path / "config.json")
    if not config:
        return []
    hints: list[str] = []
    for key in ("model_type", "_name_or_path"):
        value = config.get(key)
        if isinstance(value, str):
            hints.append(value)
    architectures = config.get("architectures")
    if isinstance(architectures, list):
        hints.extend(a for a in architectures if isinstance(a, str))
    return hints


#: Subdirectories a diffusers/mflux checkpoint keeps its weights in.
_WEIGHT_COMPONENTS = (
    "transformer",
    "transformer_2",
    "unet",
    "vae",
    "vae_decoder",
    "vae_encoder",
    "text_encoder",
    "text_encoder_2",
)


def _has_weight_files(model_path: Path) -> bool:
    """True when ``model_path`` itself holds weights (not just sub-folders).

    Used to keep an organization folder such as ``krea-2/`` — which contains a
    checkpoint directory but no weights of its own — from being mistaken for a
    model because its name happens to match a family alias.
    """
    if any(model_path.glob("*.safetensors")):
        return True
    for component in _WEIGHT_COMPONENTS:
        component_dir = model_path / component
        if component_dir.is_dir() and any(component_dir.glob("*.safetensors")):
            return True
    return False


def detect_image_spec(model_path: str | Path) -> ImageModelSpec | None:
    """Identify the image checkpoint stored in ``model_path``, if any.

    Signals are checked in order of authority:

    1. ``mflux_model.json`` (an ``mflux-save`` checkpoint records its base
       model and upstream repo there),
    2. diffusers ``model_index.json`` (pipeline and component class names),
    3. a top-level ``config.json``,
    4. the directory name (covers ``...-mflux-q8`` saves and Hub snapshots).

    Returns ``None`` for anything that is not a registered family.
    """
    path = Path(model_path)
    if not path.is_dir():
        return None
    # An HF Hub cache repo root ("models--org--name") is a container for
    # snapshots, not a checkpoint, even though mflux's repo name appears in
    # its own directory name.
    if path.name.startswith("models--") and (path / "snapshots").is_dir():
        return None

    for hints in (
        _mflux_hints(path),
        _diffusers_hints(path),
        _config_hints(path),
    ):
        for hint in hints:
            spec = match_image_spec(hint)
            if spec is not None:
                return spec

    # Fall back to the directory name, but only for a directory that actually
    # holds weights: an organization folder named after the family is not a
    # model, and its children still need to be discovered.
    if not _has_weight_files(path):
        return None
    return match_image_spec(path.name)


def looks_like_image_model_dir(model_path: str | Path) -> bool:
    """True when discovery should register ``model_path`` as an image model.

    Only registered families qualify, and only when their metadata or name is
    present. This keeps unrelated diffusion checkpoints out of the model list
    instead of advertising them as servable.
    """
    return detect_image_spec(model_path) is not None


def is_mflux_checkpoint(model_path: str | Path) -> bool:
    """True when the directory holds ``mflux``-format weights.

    ``mflux-save`` writes one index per component; Hub conversions of the same
    layout also carry them. Upstream diffusers snapshots use
    ``diffusion_pytorch_model.safetensors`` instead.
    """
    path = Path(model_path)
    if not path.is_dir():
        return False
    if (path / MFLUX_METADATA_FILE).is_file():
        return True
    return any(path.glob("*/model.safetensors.index.json"))


def parse_image_size(size: str | None) -> tuple[int | None, int | None]:
    """Parse an OpenAI-style ``size`` value.

    ``"auto"``/``None`` means "use the family default" and returns
    ``(None, None)``. Otherwise the value must be ``"<width>x<height>"`` with
    positive integers; shape validation against the family happens later.
    """
    if size is None:
        return (None, None)
    value = size.strip().lower()
    if value in ("", "auto"):
        return (None, None)
    parts = value.replace("×", "x").split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid size '{size}': expected 'auto' or '<width>x<height>' "
            "(for example '1024x1024')."
        )
    try:
        width, height = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(
            f"Invalid size '{size}': width and height must be integers."
        ) from error
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size '{size}': width and height must be positive.")
    return (width, height)


def resolve_dimensions(
    spec: ImageModelSpec,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    """Apply the family default for missing dimensions, then validate.

    Both dimensions must be within the family's range and a multiple of its
    latent grid (16 px for every family oMLX serves today).
    """
    resolved_width = width if width is not None else spec.defaults.width
    resolved_height = height if height is not None else spec.defaults.height
    if (width is None) != (height is None):
        raise ValueError("Pass both width and height, or neither.")
    for label, value in (("width", resolved_width), ("height", resolved_height)):
        if value % spec.size_multiple:
            raise ValueError(
                f"{label}={value} is not a multiple of {spec.size_multiple}, "
                f"which {spec.display_name} requires."
            )
        if value < spec.min_size or value > spec.max_size:
            raise ValueError(
                f"{label}={value} is outside the supported range "
                f"{spec.min_size}-{spec.max_size} for {spec.display_name}."
            )
    return resolved_width, resolved_height


def resolve_defaults(
    spec: ImageModelSpec,
    steps: int | None,
    guidance: float | None,
) -> tuple[int, float | None]:
    """Apply vendor defaults for sampling parameters the caller omitted."""
    resolved_steps = spec.defaults.steps if steps is None else steps
    resolved_guidance = spec.defaults.guidance if guidance is None else guidance
    if resolved_steps < 1:
        raise ValueError("steps must be at least 1.")
    if spec.defaults.guidance is None:
        # Distilled FLUX 3 has no guidance branch. Silently accepting a value
        # the runtime ignores would hide a user error, so reject it.
        if guidance is not None and guidance != 0:
            raise ValueError(
                f"{spec.display_name} does not use classifier-free guidance; "
                "omit 'guidance' (or pass 0)."
            )
        return resolved_steps, None
    if resolved_guidance is not None and resolved_guidance < 0:
        raise ValueError("guidance must be non-negative.")
    return resolved_steps, resolved_guidance
