"""Sanity tests for BC1/BC4 reference encoder, decoder, and torch ops."""

from __future__ import annotations

import numpy as np
import torch

from src.codecs.bc import (
    bc1_decode,
    bc1_encode,
    bc1_encode_decode,
    bc1_storage_bytes,
    bc4_decode,
    bc4_encode,
    bc4_encode_decode,
    bc4_storage_bytes,
    build_bc1_palette,
    build_bc4_palette,
    hard_assign_bc1,
    hard_assign_bc4,
    n_blocks,
    quantize_r8_ste,
    quantize_rgb565_ste,
)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    mse = ((a - b) ** 2).mean()
    if mse <= 0:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def test_bc1_flat_block_lossless():
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = bc1_encode_decode(img)
    assert out.shape == img.shape
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 8  # RGB565 quantization noise


def test_bc1_random_roundtrip_reasonable_quality():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    out = bc1_encode_decode(img)
    assert out.shape == img.shape
    # Uniform-random RGB is the worst case for BC1 -- 16 independent colors
    # squeezed into a 4-entry palette with RGB565 endpoints. We only require the
    # reconstruction to be non-trivially better than random (>=10 dB).
    assert _psnr(img, out) > 10.0


def test_bc1_smooth_image_high_quality():
    ys, xs = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64), indexing="ij")
    img = np.stack([
        (ys * 255).astype(np.uint8),
        (xs * 255).astype(np.uint8),
        ((ys * xs) * 255).astype(np.uint8),
    ], axis=-1)
    out = bc1_encode_decode(img)
    assert _psnr(img, out) > 30.0


def test_bc1_storage_bytes_matches_spec():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    enc = bc1_encode(img)
    assert enc.storage_bytes == bc1_storage_bytes(n_blocks(128, 128))
    assert enc.storage_bytes == 8 * (128 // 4) * (128 // 4)


def test_bc4_flat_block_lossless():
    img = np.full((4, 4), 123, dtype=np.uint8)
    out = bc4_encode_decode(img)
    assert out.shape == (4, 4, 1)
    assert np.abs(out[..., 0].astype(int) - img.astype(int)).max() == 0


def test_bc4_smooth_signal_high_quality():
    ys = np.linspace(0, 1, 64, dtype=np.float32)
    img = (ys[:, None] * np.linspace(0, 1, 64, dtype=np.float32)[None, :] * 255).astype(np.uint8)
    out = bc4_encode_decode(img)
    assert out.shape == (64, 64, 1)
    assert _psnr(img[..., None], out) > 35.0


def test_bc4_storage_bytes_matches_spec():
    img = np.zeros((128, 128), dtype=np.uint8)
    enc = bc4_encode(img)
    assert enc.storage_bytes == bc4_storage_bytes(n_blocks(128, 128))


def test_quantize_rgb565_ste_matches_numpy_path():
    rng = np.random.default_rng(0)
    rgb = torch.from_numpy(rng.random((10, 3), dtype=np.float32))
    q = quantize_rgb565_ste(rgb)
    assert q.shape == rgb.shape
    # Differences should be small (max ~1/32 of range).
    assert (q - rgb).abs().max().item() < 0.06
    # Gradient identity (straight-through).
    rgb_var = rgb.clone().requires_grad_(True)
    q = quantize_rgb565_ste(rgb_var)
    q.sum().backward()
    assert torch.allclose(rgb_var.grad, torch.ones_like(rgb_var.grad))


def test_quantize_r8_ste_grad_identity():
    x = torch.linspace(0, 1, 16).requires_grad_(True)
    y = quantize_r8_ste(x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x.grad))


def test_build_bc1_palette_endpoints_match():
    e0 = torch.tensor([[1.0, 0.0, 0.0]])
    e1 = torch.tensor([[0.0, 0.0, 1.0]])
    p = build_bc1_palette(e0, e1)
    assert p.shape == (1, 4, 3)
    assert torch.allclose(p[0, 0], e0[0])
    assert torch.allclose(p[0, 1], e1[0])
    assert torch.allclose(p[0, 2], (2 * e0 + e1) / 3)
    assert torch.allclose(p[0, 3], (e0 + 2 * e1) / 3)


def test_hard_assign_bc1_picks_closest():
    e0 = torch.tensor([[1.0, 0.0, 0.0]])
    e1 = torch.tensor([[0.0, 0.0, 1.0]])
    palette = build_bc1_palette(e0, e1)
    # 16 texels: 4 of each palette color.
    indices_expected = torch.tensor([0, 1, 2, 3] * 4)
    colors = palette[0, indices_expected]  # (16, 3)
    palette_b = palette.expand(16, 4, 3)
    idx, decoded = hard_assign_bc1(palette_b, colors)
    assert torch.equal(idx, indices_expected)
    assert torch.allclose(decoded, colors)


def test_build_bc4_palette_correct():
    r0 = torch.tensor([1.0])
    r1 = torch.tensor([0.0])
    p = build_bc4_palette(r0, r1)
    assert p.shape == (1, 8)
    expected = torch.tensor([
        1.0, 0.0,
        6.0 / 7.0, 5.0 / 7.0, 4.0 / 7.0, 3.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0,
    ])
    assert torch.allclose(p[0], expected, atol=1e-6)


def test_hard_assign_bc4_picks_closest():
    r0 = torch.tensor([1.0])
    r1 = torch.tensor([0.0])
    palette = build_bc4_palette(r0, r1)
    palette_b = palette.expand(8, 8)
    values = palette[0]
    idx, decoded = hard_assign_bc4(palette_b, values)
    assert torch.equal(idx, torch.arange(8))
    assert torch.allclose(decoded, values)
