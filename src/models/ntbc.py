"""Neural Texture Block Compression (NTBC) model.

Mirrors Sec. 3.2 of Fujieda & Harada (2024). Two MLPs predict, for a single
material:

- The **color network** maps a texel coord ``(u, v)`` to the predicted
  uncompressed color ``c_hat``. Inputs are encoded with a multi-resolution
  feature grid.
- The **endpoint network** maps a 4x4 block coord ``(s, t)`` to a pair of BC
  endpoints ``(e0_hat, e1_hat)``.

At inference time we build the BC1/BC4 palette from the predicted endpoints,
compute the per-texel argmin index against ``c_hat``, and reconstruct decoded
colors. The whole material's BC1 and BC4 layers share a single color network
and a single endpoint network -- the joint multi-texture training prescribed
by the NTBC paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..codecs.bc.torch_ops import (
    build_bc1_palette,
    build_bc4_palette,
    hard_assign_bc1,
    hard_assign_bc4,
    quantize_r8_ste,
    quantize_rgb565_ste,
)
from .hash_grid import MultiResFeatureGrid, MultiResGridConfig
from .mlp import TinyMLP

if TYPE_CHECKING:
    from ..data.material import Material


@dataclass
class NTBCConfig:
    """Hyperparameters for an NTBC model instance."""

    color_grid: MultiResGridConfig = field(
        default_factory=lambda: MultiResGridConfig(
            n_levels=6, coarsest=16, finest=512, n_features=2
        )
    )
    endpoint_grid: MultiResGridConfig = field(
        default_factory=lambda: MultiResGridConfig(
            n_levels=5, coarsest=8, finest=128, n_features=2
        )
    )
    mlp_hidden: int = 64
    mlp_layers: int = 3


@dataclass
class NTBCOutput:
    """Container holding all tensors produced by a single NTBC forward pass."""

    # Per-texel predictions and reconstructions.
    color_pred: torch.Tensor          # (N, color_out_dim) c_hat in [0, 1]
    bc1_decoded: torch.Tensor | None  # (N, 3 * n_bc1) decoded colors
    bc4_decoded: torch.Tensor | None  # (N, 1 * n_bc4) decoded values
    # Per-block predictions.
    bc1_endpoints: torch.Tensor | None  # (M, 2, 3 * n_bc1) post-quantization endpoints
    bc4_endpoints: torch.Tensor | None  # (M, 2, 1 * n_bc4)
    # Per-texel hard-assigned palette indices.
    bc1_indices: torch.Tensor | None    # (N, n_bc1) int64 in {0..3}
    bc4_indices: torch.Tensor | None    # (N, n_bc4) int64 in {0..7}


class NTBC(nn.Module):
    """Joint-material NTBC network for a single material's BC1 + BC4 layers."""

    def __init__(self, material: "Material", config: NTBCConfig | None = None) -> None:
        super().__init__()
        self.config = config or NTBCConfig()
        self.n_bc1 = len(material.bc1_layers)
        self.n_bc4 = len(material.bc4_layers)
        self.resolution = material.resolution

        self.color_grid = MultiResFeatureGrid(self.config.color_grid)
        self.endpoint_grid = MultiResFeatureGrid(self.config.endpoint_grid)
        self.color_out_dim = 3 * self.n_bc1 + 1 * self.n_bc4
        self.endpoint_out_dim = 6 * self.n_bc1 + 2 * self.n_bc4
        self.color_mlp = TinyMLP(
            in_dim=self.color_grid.output_dim,
            out_dim=self.color_out_dim,
            hidden_dim=self.config.mlp_hidden,
            n_hidden=self.config.mlp_layers,
        )
        self.endpoint_mlp = TinyMLP(
            in_dim=self.endpoint_grid.output_dim,
            out_dim=self.endpoint_out_dim,
            hidden_dim=self.config.mlp_hidden,
            n_hidden=self.config.mlp_layers,
        )

    # -- inner predictions ---------------------------------------------------

    def predict_colors(self, uv: torch.Tensor) -> torch.Tensor:
        """``uv``: ``(N, 2)`` in ``[0, 1]``. Returns ``(N, color_out_dim)``."""
        return self.color_mlp(self.color_grid(uv))

    def predict_endpoints(self, st: torch.Tensor) -> torch.Tensor:
        """``st``: ``(M, 2)`` in ``[0, 1]``. Returns ``(M, endpoint_out_dim)``."""
        return self.endpoint_mlp(self.endpoint_grid(st))

    # -- structured slicing of the joint outputs ----------------------------

    def split_colors(self, colors: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Split a joint color tensor into BC1 (3-ch per layer) and BC4 (1-ch per layer)."""
        bc1 = colors[..., : 3 * self.n_bc1] if self.n_bc1 else None
        bc4 = colors[..., 3 * self.n_bc1 :] if self.n_bc4 else None
        return bc1, bc4

    def split_endpoints(
        self, endpoints: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Split a joint endpoint tensor into ``(bc1_endpoints, bc4_endpoints)``.

        Layout per block:
        ``[e0_bc1_layer0_rgb, e1_bc1_layer0_rgb, ..., e0_bc4_layer0, e1_bc4_layer0, ...]``.
        Output tensors have shape ``(M, 2, channels)`` -- explicit (e0, e1).
        """
        n_bc1_endpoints = 6 * self.n_bc1
        bc1_part = endpoints[..., :n_bc1_endpoints]
        bc4_part = endpoints[..., n_bc1_endpoints:]
        bc1 = bc1_part.reshape(*bc1_part.shape[:-1], self.n_bc1, 2, 3) if self.n_bc1 else None
        bc4 = bc4_part.reshape(*bc4_part.shape[:-1], self.n_bc4, 2, 1) if self.n_bc4 else None
        return bc1, bc4

    # -- joint forward ------------------------------------------------------

    def forward(
        self,
        uv: torch.Tensor,
        st: torch.Tensor,
        texel_to_block: torch.Tensor,
        *,
        return_decoded: bool = True,
    ) -> NTBCOutput:
        """Joint forward pass.

        Parameters
        ----------
        uv: ``(N, 2)`` texel UV coords in [0, 1].
        st: ``(M, 2)`` block ST coords in [0, 1].
        texel_to_block: ``(N,)`` int64 mapping each texel to its block index in ``st``.
        return_decoded: If False, skip palette construction / argmin (auxiliary-loss-only mode).
        """
        colors = self.predict_colors(uv)
        endpoints_raw = self.predict_endpoints(st)

        bc1_color, bc4_color = self.split_colors(colors)
        bc1_endpoints_raw, bc4_endpoints_raw = self.split_endpoints(endpoints_raw)

        # Quantize endpoints to RGB565 (BC1) / 8-bit (BC4) using STE so the
        # endpoint network sees realistic palette construction during training.
        if bc1_endpoints_raw is not None:
            # Reshape to apply RGB565 to the trailing 3 channels per endpoint.
            bc1_endpoints = quantize_rgb565_ste(bc1_endpoints_raw)  # (M, n_bc1, 2, 3)
        else:
            bc1_endpoints = None
        if bc4_endpoints_raw is not None:
            bc4_endpoints = quantize_r8_ste(bc4_endpoints_raw)  # (M, n_bc4, 2, 1)
        else:
            bc4_endpoints = None

        bc1_decoded = bc1_indices = None
        if return_decoded and self.n_bc1 > 0 and bc1_endpoints is not None and bc1_color is not None:
            # Build per-block palette: (M, n_bc1, 4, 3).
            e0 = bc1_endpoints[..., 0, :]  # (M, n_bc1, 3)
            e1 = bc1_endpoints[..., 1, :]
            palette = build_bc1_palette(e0, e1)  # (M, n_bc1, 4, 3)
            # Gather per-texel palette via texel_to_block index.
            palette_per_texel = palette[texel_to_block]  # (N, n_bc1, 4, 3)
            # Reshape colors to (N, n_bc1, 3).
            color_per_layer = bc1_color.reshape(-1, self.n_bc1, 3)
            bc1_indices, bc1_decoded_per_layer = hard_assign_bc1(
                palette_per_texel, color_per_layer
            )  # indices (N, n_bc1), decoded (N, n_bc1, 3)
            bc1_decoded = bc1_decoded_per_layer.reshape(-1, self.n_bc1 * 3)

        bc4_decoded = bc4_indices = None
        if return_decoded and self.n_bc4 > 0 and bc4_endpoints is not None and bc4_color is not None:
            r0 = bc4_endpoints[..., 0, 0]  # (M, n_bc4)
            r1 = bc4_endpoints[..., 1, 0]
            palette = build_bc4_palette(r0, r1)  # (M, n_bc4, 8)
            palette_per_texel = palette[texel_to_block]  # (N, n_bc4, 8)
            value_per_layer = bc4_color.reshape(-1, self.n_bc4)  # (N, n_bc4)
            bc4_indices, bc4_decoded_per_layer = hard_assign_bc4(
                palette_per_texel, value_per_layer
            )  # indices (N, n_bc4), decoded (N, n_bc4)
            bc4_decoded = bc4_decoded_per_layer

        return NTBCOutput(
            color_pred=colors,
            bc1_decoded=bc1_decoded,
            bc4_decoded=bc4_decoded,
            bc1_endpoints=bc1_endpoints,
            bc4_endpoints=bc4_endpoints,
            bc1_indices=bc1_indices,
            bc4_indices=bc4_indices,
        )

    # -- storage accounting -------------------------------------------------

    def storage_bytes(self, mlp_fp_bytes: int = 2, grid_fp_bytes: int = 1) -> int:
        """Total inference-time storage for the model.

        Defaults assume the MLP weights are stored in fp16 (the NTBC paper uses
        half-precision MLPs) and the feature grids in int8 after QAT.
        """
        n_color_mlp = sum(p.numel() for p in self.color_mlp.parameters())
        n_endpoint_mlp = sum(p.numel() for p in self.endpoint_mlp.parameters())
        return (
            (n_color_mlp + n_endpoint_mlp) * mlp_fp_bytes
            + self.color_grid.storage_bytes(grid_fp_bytes)
            + self.endpoint_grid.storage_bytes(grid_fp_bytes)
        )
