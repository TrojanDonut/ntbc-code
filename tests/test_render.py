"""Smoke tests for the sphere renderer."""

from __future__ import annotations

import numpy as np

from src.data.material import Role
from src.render.sphere import RenderConfig, render_sphere


def test_render_sphere_outputs_image():
    layers = {
        Role.COLOR: np.full((32, 32, 3), 200, dtype=np.uint8),
        Role.ROUGHNESS: np.full((32, 32, 1), 100, dtype=np.uint8),
        Role.METALNESS: np.full((32, 32, 1), 0, dtype=np.uint8),
    }
    img = render_sphere(layers, RenderConfig(size=64))
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
    # Center should be lit, corners should be background.
    assert img[32, 32].sum() > img[1, 1].sum()


def test_render_sphere_normal_map_changes_shading():
    rng = np.random.default_rng(0)
    base = {
        Role.COLOR: np.full((32, 32, 3), 200, dtype=np.uint8),
        Role.ROUGHNESS: np.full((32, 32, 1), 80, dtype=np.uint8),
    }
    flat_normal = np.full((32, 32, 3), [128, 128, 255], dtype=np.uint8)
    noisy_normal = np.clip(
        flat_normal.astype(np.int16) + rng.integers(-40, 40, flat_normal.shape, dtype=np.int16),
        0, 255,
    ).astype(np.uint8)
    img_flat = render_sphere({**base, Role.NORMAL: flat_normal}, RenderConfig(size=64))
    img_noisy = render_sphere({**base, Role.NORMAL: noisy_normal}, RenderConfig(size=64))
    assert not np.array_equal(img_flat, img_noisy)
