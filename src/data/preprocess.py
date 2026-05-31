"""Load PBR material textures from disk and convert them into ``Material`` objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .material import (
    ROLE_TO_FORMAT,
    BCFormat,
    Material,
    Role,
    TextureLayer,
    role_from_filename,
)


PNG_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif")


def _load_image(path: Path) -> np.ndarray:
    """Load an image as an HxWxC uint8 numpy array."""
    img = Image.open(path)
    if img.mode in ("I", "I;16", "L;16"):
        img = img.convert("I")
        arr = np.array(img, dtype=np.int32)
        arr = (arr / 257).clip(0, 255).astype(np.uint8)
        return arr[:, :, None]
    if img.mode == "RGBA":
        img = img.convert("RGB")
    if img.mode == "L":
        return np.array(img, dtype=np.uint8)[:, :, None]
    img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def _resize(arr: np.ndarray, target: int) -> np.ndarray:
    """Resize an image array to ``target x target`` using Pillow's LANCZOS filter."""
    h, w = arr.shape[:2]
    if h == target and w == target:
        return arr
    if arr.shape[2] == 1:
        img = Image.fromarray(arr[:, :, 0], mode="L")
        img = img.resize((target, target), resample=Image.LANCZOS)
        return np.array(img, dtype=np.uint8)[:, :, None]
    img = Image.fromarray(arr, mode="RGB")
    img = img.resize((target, target), resample=Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _to_layer(
    name: str, role: Role, arr: np.ndarray, bc_format: BCFormat
) -> TextureLayer:
    """Convert an HxWxC uint8 array to a ``TextureLayer`` tensor in [0, 1]."""
    if bc_format == BCFormat.BC4:
        if arr.shape[2] >= 3:
            arr = arr[:, :, :1]
        chw = arr.transpose(2, 0, 1)
    else:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        chw = arr[:, :, :3].transpose(2, 0, 1)
    tensor = torch.from_numpy(chw.astype(np.float32) / 255.0)
    return TextureLayer(name=name, role=role, bc_format=bc_format, data=tensor)


def load_material_from_dir(
    directory: Path,
    *,
    resolution: int = 512,
    allowed_roles: set[Role] | None = None,
) -> Material:
    """Load a material from a directory of PBR texture PNGs.

    Files are auto-classified by filename using :func:`role_from_filename`.
    All textures are resized to ``resolution x resolution``. ``resolution`` must
    be a multiple of 4 because BC1/BC4 operate on 4x4 texel blocks.
    """
    if resolution % 4 != 0:
        raise ValueError("resolution must be a multiple of 4 (BC block size)")

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Material directory not found: {directory}")

    files: list[Path] = []
    for pattern in PNG_GLOBS:
        files.extend(sorted(directory.glob(pattern)))

    # If both NormalGL and NormalDX are present we prefer the OpenGL convention,
    # which is what most open renderers (and pyrender) expect.
    has_normal_gl = any("normalgl" in f.stem.lower() for f in files)
    if has_normal_gl:
        files = [f for f in files if "normaldx" not in f.stem.lower()]

    layers: list[TextureLayer] = []
    seen_roles: set[Role] = set()
    for file in files:
        role = role_from_filename(file)
        if role is None:
            continue
        if allowed_roles is not None and role not in allowed_roles:
            continue
        if role in seen_roles:
            continue
        bc_format = ROLE_TO_FORMAT[role]
        arr = _load_image(file)
        arr = _resize(arr, resolution)
        layer = _to_layer(name=file.stem, role=role, arr=arr, bc_format=bc_format)
        layers.append(layer)
        seen_roles.add(role)

    if not layers:
        raise ValueError(
            f"No recognizable PBR textures found in {directory}. "
            f"Expected ambientCG- or PolyHaven-style filenames."
        )

    layers.sort(key=lambda layer: list(Role).index(layer.role))
    return Material(name=directory.name, layers=layers)


def stack_bc1(material: Material) -> torch.Tensor:
    """Stack all BC1 layers of a material into a single (3*N, H, W) tensor."""
    layers = material.bc1_layers
    if not layers:
        return torch.zeros(0, material.resolution, material.resolution)
    return torch.cat([layer.data for layer in layers], dim=0)


def stack_bc4(material: Material) -> torch.Tensor:
    """Stack all BC4 layers of a material into a single (N, H, W) tensor."""
    layers = material.bc4_layers
    if not layers:
        return torch.zeros(0, material.resolution, material.resolution)
    return torch.cat([layer.data for layer in layers], dim=0)


def make_synthetic_material(
    name: str = "Synthetic", resolution: int = 64, seed: int = 0
) -> Material:
    """Create a small synthetic material for tests (no network required)."""
    rng = np.random.default_rng(seed)

    def gradient(h: int, w: int, c: int) -> np.ndarray:
        ys = np.linspace(0, 1, h, dtype=np.float32)
        xs = np.linspace(0, 1, w, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        stack = []
        for i in range(c):
            phase = 0.5 * i
            stack.append(0.5 + 0.5 * np.sin(2 * np.pi * (grid_x + grid_y + phase)))
        arr = np.stack(stack, axis=-1)
        noise = 0.05 * rng.standard_normal(arr.shape).astype(np.float32)
        return np.clip((arr + noise) * 255.0, 0, 255).astype(np.uint8)

    color = gradient(resolution, resolution, 3)
    normal = gradient(resolution, resolution, 3)
    rough = gradient(resolution, resolution, 1)
    disp = gradient(resolution, resolution, 1)

    layers = [
        _to_layer("synth_color", Role.COLOR, color, BCFormat.BC1),
        _to_layer("synth_normal", Role.NORMAL, normal, BCFormat.BC1),
        _to_layer("synth_roughness", Role.ROUGHNESS, rough, BCFormat.BC4),
        _to_layer("synth_displacement", Role.DISPLACEMENT, disp, BCFormat.BC4),
    ]
    return Material(name=name, layers=layers)
