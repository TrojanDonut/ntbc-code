"""Tests for data preprocessing and material loading."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.material import BCFormat, Role, role_from_filename
from src.data.preprocess import load_material_from_dir, make_synthetic_material
from pathlib import Path


def test_synthetic_material_layout():
    m = make_synthetic_material(resolution=32)
    assert m.resolution == 32
    assert len(m.bc1_layers) == 2  # color, normal
    assert len(m.bc4_layers) == 2  # roughness, displacement
    for layer in m.layers:
        assert layer.data.dtype.is_floating_point
        assert layer.data.min().item() >= 0.0
        assert layer.data.max().item() <= 1.0
        if layer.bc_format == BCFormat.BC1:
            assert layer.data.shape[0] == 3
        else:
            assert layer.data.shape[0] == 1


def test_role_from_filename():
    cases = {
        "MetalPlates013_1K-PNG_Color.png": Role.COLOR,
        "Wood062_1K-PNG_NormalGL.png": Role.NORMAL,
        "Rocks023_1K-PNG_Roughness.png": Role.ROUGHNESS,
        "Bricks075A_1K-PNG_Displacement.png": Role.DISPLACEMENT,
        "Carpet015_1K-PNG_AmbientOcclusion.png": Role.AO,
        "brick_wall_diff_1k.png": Role.COLOR,
        "brick_wall_nor_gl_1k.png": Role.NORMAL,
        "brick_wall_rough_1k.png": Role.ROUGHNESS,
    }
    for fname, expected in cases.items():
        assert role_from_filename(Path(fname)) == expected, fname


def test_load_material_from_dir(tmp_path):
    rng = np.random.default_rng(0)
    base = tmp_path / "FakeMat"
    base.mkdir()
    rgb = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    gray = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    Image.fromarray(rgb).save(base / "FakeMat_1K-PNG_Color.png")
    Image.fromarray(rgb).save(base / "FakeMat_1K-PNG_NormalGL.png")
    Image.fromarray(gray, mode="L").save(base / "FakeMat_1K-PNG_Roughness.png")
    Image.fromarray(gray, mode="L").save(base / "FakeMat_1K-PNG_Displacement.png")

    mat = load_material_from_dir(base, resolution=32)
    assert mat.resolution == 32
    assert {layer.role for layer in mat.layers} == {
        Role.COLOR, Role.NORMAL, Role.ROUGHNESS, Role.DISPLACEMENT
    }
    color = [layer for layer in mat.layers if layer.role == Role.COLOR][0]
    assert color.data.shape == (3, 32, 32)
    rough = [layer for layer in mat.layers if layer.role == Role.ROUGHNESS][0]
    assert rough.data.shape == (1, 32, 32)


def test_invalid_resolution_rejected(tmp_path):
    base = tmp_path / "BadMat"
    base.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
        base / "BadMat_1K-PNG_Color.png"
    )
    with pytest.raises(ValueError, match="multiple of 4"):
        load_material_from_dir(base, resolution=13)
