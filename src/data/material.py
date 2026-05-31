"""Material and texture-layer data structures used throughout the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch


class BCFormat(str, Enum):
    """Block-compression target format for a texture layer."""

    BC1 = "BC1"  # 3-channel RGB, 8 bytes per 4x4 block
    BC4 = "BC4"  # 1-channel, 8 bytes per 4x4 block


class Role(str, Enum):
    """Semantic role of a texture layer within a PBR material."""

    COLOR = "color"
    NORMAL = "normal"
    ROUGHNESS = "roughness"
    METALNESS = "metalness"
    DISPLACEMENT = "displacement"
    AO = "ao"
    OPACITY = "opacity"


# Canonical mapping from semantic role to target BC format.
# Normal maps go through BC1 (RGB) since BC5 is out of scope for this seminar.
ROLE_TO_FORMAT: dict[Role, BCFormat] = {
    Role.COLOR: BCFormat.BC1,
    Role.NORMAL: BCFormat.BC1,
    Role.ROUGHNESS: BCFormat.BC4,
    Role.METALNESS: BCFormat.BC4,
    Role.DISPLACEMENT: BCFormat.BC4,
    Role.AO: BCFormat.BC4,
    Role.OPACITY: BCFormat.BC4,
}


@dataclass
class TextureLayer:
    """A single texture layer within a material.

    `data` is stored as a float tensor in [0, 1] with shape (C, H, W). C is 3 for
    BC1 layers and 1 for BC4 layers.
    """

    name: str
    role: Role
    bc_format: BCFormat
    data: torch.Tensor

    @property
    def channels(self) -> int:
        return self.data.shape[0]

    @property
    def resolution(self) -> tuple[int, int]:
        return self.data.shape[1], self.data.shape[2]


@dataclass
class Material:
    """A PBR material: a named collection of texture layers at a common resolution."""

    name: str
    layers: list[TextureLayer] = field(default_factory=list)

    @property
    def resolution(self) -> int:
        if not self.layers:
            raise ValueError(f"Material '{self.name}' has no layers")
        h, w = self.layers[0].resolution
        if h != w:
            raise ValueError(f"Non-square texture in material '{self.name}': {h}x{w}")
        for layer in self.layers[1:]:
            if layer.resolution != (h, w):
                raise ValueError(
                    f"Inconsistent resolutions in material '{self.name}': "
                    f"{layer.name} is {layer.resolution} vs {(h, w)}"
                )
        return h

    @property
    def bc1_layers(self) -> list[TextureLayer]:
        return [layer for layer in self.layers if layer.bc_format == BCFormat.BC1]

    @property
    def bc4_layers(self) -> list[TextureLayer]:
        return [layer for layer in self.layers if layer.bc_format == BCFormat.BC4]

    def __len__(self) -> int:
        return len(self.layers)


# ambientCG naming uses suffixes like "_Color", "_NormalGL", "_Roughness", etc.
AMBIENTCG_SUFFIX_TO_ROLE: dict[str, Role] = {
    "color": Role.COLOR,
    "diffuse": Role.COLOR,
    "albedo": Role.COLOR,
    "basecolor": Role.COLOR,
    "normalgl": Role.NORMAL,
    "normaldx": Role.NORMAL,
    "normal": Role.NORMAL,
    "roughness": Role.ROUGHNESS,
    "metalness": Role.METALNESS,
    "metallic": Role.METALNESS,
    "displacement": Role.DISPLACEMENT,
    "height": Role.DISPLACEMENT,
    "ambientocclusion": Role.AO,
    "ao": Role.AO,
    "opacity": Role.OPACITY,
}


def role_from_filename(path: Path) -> Role | None:
    """Heuristic mapping from a texture filename to its semantic role.

    Recognizes ambientCG conventions (e.g. ``MetalPlates013_1K-PNG_Color.png``)
    and Poly Haven conventions (e.g. ``brick_wall_diff_1k.png``).
    """
    stem = path.stem.lower()
    # Strip resolution suffixes commonly present in Poly Haven filenames.
    for suffix in ("_1k", "_2k", "_4k", "_8k", "-1k", "-2k", "-4k", "-8k"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    tokens = stem.replace("-", "_").split("_")
    # Try longest-token-first matching against the lookup table.
    for token in reversed(tokens):
        if token in AMBIENTCG_SUFFIX_TO_ROLE:
            return AMBIENTCG_SUFFIX_TO_ROLE[token]
    # Poly Haven uses short codes like diff, nor, rough, disp, ao.
    short_map = {
        "diff": Role.COLOR,
        "col": Role.COLOR,
        "nor": Role.NORMAL,
        "rough": Role.ROUGHNESS,
        "metal": Role.METALNESS,
        "disp": Role.DISPLACEMENT,
        "height": Role.DISPLACEMENT,
        "ao": Role.AO,
        "arm": None,  # AO+Roughness+Metalness packed; needs splitting upstream
    }
    for token in reversed(tokens):
        if token in short_map and short_map[token] is not None:
            return short_map[token]
    return None
