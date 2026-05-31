"""Tests for the multi-resolution feature grid and MLP."""

from __future__ import annotations

import torch

from src.models.hash_grid import MultiResFeatureGrid, MultiResGridConfig, geometric_levels
from src.models.mlp import TinyMLP


def test_geometric_levels_endpoints():
    L = geometric_levels(16, 512, 6)
    assert L[0] == 16
    assert L[-1] == 512
    assert L == sorted(L)


def test_grid_output_shape():
    grid = MultiResFeatureGrid(MultiResGridConfig(n_levels=4, coarsest=8, finest=64, n_features=2))
    uv = torch.rand(100, 2)
    out = grid(uv)
    assert out.shape == (100, 4 * 2)
    assert torch.isfinite(out).all()


def test_grid_qat_keeps_gradients():
    grid = MultiResFeatureGrid(MultiResGridConfig(n_levels=3, coarsest=8, finest=32, n_features=2))
    grid.enable_qat(True)
    uv = torch.rand(64, 2, requires_grad=False)
    out = grid(uv)
    loss = out.pow(2).mean()
    loss.backward()
    for p in grid.features.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_grid_storage_bytes_includes_scales():
    grid = MultiResFeatureGrid(MultiResGridConfig(n_levels=3, coarsest=8, finest=32, n_features=2))
    expected_features = 2 * (8 * 8 + 16 * 16 + 32 * 32)
    assert grid.storage_bytes() == expected_features + 4 * 3


def test_mlp_forward():
    mlp = TinyMLP(in_dim=12, out_dim=6, hidden_dim=64, n_hidden=3)
    x = torch.randn(32, 12)
    y = mlp(x)
    assert y.shape == (32, 6)
    assert (y >= 0).all() and (y <= 1).all()
    n_params = sum(p.numel() for p in mlp.parameters())
    expected = (
        12 * 64 + 64
        + 64 * 64 + 64
        + 64 * 64 + 64
        + 64 * 6 + 6
    )
    assert n_params == expected
