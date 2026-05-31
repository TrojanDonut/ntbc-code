"""Smoke tests for the NTBC model and coord helpers."""

from __future__ import annotations

import torch

from src.data.preprocess import make_synthetic_material
from src.models.coords import (
    block_st_grid,
    sample_block_batch,
    texel_to_block_indices,
    texel_uv_grid,
)
from src.models.hash_grid import MultiResGridConfig
from src.models.ntbc import NTBC, NTBCConfig


def test_texel_to_block_consistency():
    h = w = 32
    t2b = texel_to_block_indices(h, w, block=4)
    assert t2b.shape == (h * w,)
    assert t2b.unique().numel() == (h // 4) * (w // 4)
    # The block index of texel (0, 0) is 0; of texel (4, 0) is 1 row down (Wb=8).
    assert t2b[0].item() == 0
    assert t2b[4 * w].item() == (h // 4)  # we made block_y change after 4 full rows


def test_uv_and_st_grids():
    h = w = 32
    uv = texel_uv_grid(h, w)
    st = block_st_grid(h, w, block=4)
    assert uv.shape == (h * w, 2)
    assert st.shape == ((h // 4) * (w // 4), 2)
    assert uv.min() > 0 and uv.max() < 1


def test_sample_block_batch_shapes():
    uv, st, t2b = sample_block_batch(64, 64, n_blocks=8)
    assert st.shape == (8, 2)
    assert uv.shape == (8 * 16, 2)
    assert t2b.shape == (8 * 16,)
    assert t2b.min().item() == 0
    assert t2b.max().item() <= 7
    # Every block id appears exactly 16 times.
    counts = torch.bincount(t2b, minlength=8)
    assert (counts == 16).all()


def test_ntbc_forward_shapes_and_grads():
    torch.manual_seed(0)
    mat = make_synthetic_material(resolution=32)
    cfg = NTBCConfig(
        color_grid=MultiResGridConfig(n_levels=3, coarsest=4, finest=32, n_features=2),
        endpoint_grid=MultiResGridConfig(n_levels=3, coarsest=4, finest=8, n_features=2),
        mlp_hidden=16,
        mlp_layers=2,
    )
    model = NTBC(mat, cfg)
    uv, st, t2b = sample_block_batch(32, 32, n_blocks=4)
    out = model(uv, st, t2b)

    assert out.color_pred.shape == (4 * 16, 3 * 2 + 1 * 2)
    assert out.bc1_decoded.shape == (4 * 16, 3 * 2)
    assert out.bc4_decoded.shape == (4 * 16, 1 * 2)
    assert out.bc1_indices.shape == (4 * 16, 2)
    assert out.bc4_indices.shape == (4 * 16, 2)

    loss = (out.bc1_decoded.pow(2).mean() + out.bc4_decoded.pow(2).mean()
            + out.color_pred.pow(2).mean())
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None, "param had no gradient"
            assert torch.isfinite(p.grad).all()


def test_ntbc_storage_bytes_decomposes():
    mat = make_synthetic_material(resolution=32)
    cfg = NTBCConfig(
        color_grid=MultiResGridConfig(n_levels=3, coarsest=4, finest=32, n_features=2),
        endpoint_grid=MultiResGridConfig(n_levels=3, coarsest=4, finest=8, n_features=2),
        mlp_hidden=16,
        mlp_layers=2,
    )
    model = NTBC(mat, cfg)
    total = model.storage_bytes()
    assert total > 0
