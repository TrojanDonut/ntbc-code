"""Torch-side BC1/BC4 palette operations used inside the NTBC training loop.

NTBC's training pass needs to (a) build the BC1/BC4 palette from network-predicted
endpoints, (b) for each texel compute the argmin over the palette against the
network-predicted color, and (c) reconstruct decoded colors so we can take an L2
loss against the reference uncompressed texture. All of this is done in torch
so gradients can flow back to the endpoint network and grid features.
"""

from __future__ import annotations

import torch


BC1_INTERP_TS = torch.tensor([0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0])
BC4_INTERP_TS = torch.tensor(
    [0.0, 1.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0]
)


def quantize_rgb565_ste(rgb: torch.Tensor) -> torch.Tensor:
    """RGB565 quantize+dequantize with a straight-through estimator.

    ``rgb`` is in [0, 1]. Forward: emulate the hardware RGB565 quantization-and-
    expansion; backward: identity. This lets the endpoint network see realistic
    quantization noise during training while still receiving usable gradients.
    """
    r5 = torch.round(rgb[..., 0] * 31.0).clamp(0, 31)
    g6 = torch.round(rgb[..., 1] * 63.0).clamp(0, 63)
    b5 = torch.round(rgb[..., 2] * 31.0).clamp(0, 31)
    # Bit-shift expansion that GPUs use to decode RGB565 -> RGB888.
    r8 = ((r5.to(torch.int32) << 3) | (r5.to(torch.int32) >> 2)).to(torch.float32) / 255.0
    g8 = ((g6.to(torch.int32) << 2) | (g6.to(torch.int32) >> 4)).to(torch.float32) / 255.0
    b8 = ((b5.to(torch.int32) << 3) | (b5.to(torch.int32) >> 2)).to(torch.float32) / 255.0
    quant = torch.stack([r8, g8, b8], dim=-1)
    return rgb + (quant - rgb).detach()


def quantize_r8_ste(x: torch.Tensor) -> torch.Tensor:
    """8-bit unsigned quantize+dequantize with a straight-through estimator."""
    q = torch.round(x.clamp(0, 1) * 255.0) / 255.0
    return x + (q - x).detach()


def build_bc1_palette(e0: torch.Tensor, e1: torch.Tensor) -> torch.Tensor:
    """Construct a 4-color BC1 palette from per-block endpoints.

    ``e0``, ``e1``: ``(..., 3)``. Returns ``(..., 4, 3)``.
    """
    c2 = (2.0 * e0 + e1) / 3.0
    c3 = (e0 + 2.0 * e1) / 3.0
    return torch.stack([e0, e1, c2, c3], dim=-2)


def build_bc4_palette(r0: torch.Tensor, r1: torch.Tensor) -> torch.Tensor:
    """Construct an 8-color BC4 palette from per-block endpoints.

    ``r0``, ``r1``: ``(...,)``. Returns ``(..., 8)``.
    """
    base = torch.stack([r0, r1], dim=-1)
    interps = torch.stack(
        [
            (6.0 * r0 + 1.0 * r1) / 7.0,
            (5.0 * r0 + 2.0 * r1) / 7.0,
            (4.0 * r0 + 3.0 * r1) / 7.0,
            (3.0 * r0 + 4.0 * r1) / 7.0,
            (2.0 * r0 + 5.0 * r1) / 7.0,
            (1.0 * r0 + 6.0 * r1) / 7.0,
        ],
        dim=-1,
    )
    return torch.cat([base, interps], dim=-1)


def hard_assign_bc1(
    palette: torch.Tensor, color: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the palette entry closest to ``color`` per texel (hard argmin).

    ``palette``: ``(..., 4, 3)``; ``color``: ``(..., 3)`` (same leading shape).
    Returns ``(indices, decoded)`` where ``indices`` is int64 with values in
    ``{0..3}`` and ``decoded`` is the gathered palette colors.

    The gradient flows through the palette entries only, never through the
    argmin (standard NTBC training behavior).
    """
    diff = color.unsqueeze(-2) - palette
    d2 = diff.pow(2).sum(dim=-1)
    idx = d2.argmin(dim=-1)
    gathered = torch.gather(
        palette, dim=-2, index=idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, 3)
    ).squeeze(-2)
    return idx, gathered


def hard_assign_bc4(
    palette: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the palette entry closest to ``value`` per texel (hard argmin).

    ``palette``: ``(..., 8)``; ``value``: ``(...,)``.
    """
    diff = value.unsqueeze(-1) - palette
    d2 = diff.pow(2)
    idx = d2.argmin(dim=-1)
    gathered = torch.gather(palette, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    return idx, gathered
