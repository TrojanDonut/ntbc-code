"""Helpers to build UV (per-texel) and ST (per-block) coordinate grids and the
texel-to-block lookup used by NTBC.

All coordinates use the half-pixel convention (centers of texels/blocks) and are
in ``[0, 1]``. This matches the NTBC paper's normalized indexing.
"""

from __future__ import annotations

import torch


def texel_uv_grid(h: int, w: int) -> torch.Tensor:
    """``(H*W, 2)`` tensor of texel-center UV coordinates row-major."""
    ys = (torch.arange(h, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(-1, 2)


def block_st_grid(h: int, w: int, block: int = 4) -> torch.Tensor:
    """``(Hb*Wb, 2)`` tensor of block-center ST coordinates row-major."""
    if h % block or w % block:
        raise ValueError(f"image dims ({h}x{w}) must be multiples of {block}")
    hb, wb = h // block, w // block
    ss = (torch.arange(hb, dtype=torch.float32) + 0.5) / hb
    ts = (torch.arange(wb, dtype=torch.float32) + 0.5) / wb
    gs, gt = torch.meshgrid(ss, ts, indexing="ij")
    return torch.stack([gt, gs], dim=-1).reshape(-1, 2)


def texel_to_block_indices(h: int, w: int, block: int = 4) -> torch.Tensor:
    """``(H*W,)`` int64 tensor giving the block index each texel belongs to.

    Block indices are row-major over ``(Hb, Wb)`` and match :func:`block_st_grid`.
    """
    if h % block or w % block:
        raise ValueError(f"image dims ({h}x{w}) must be multiples of {block}")
    wb = w // block
    ys = torch.arange(h, dtype=torch.int64)
    xs = torch.arange(w, dtype=torch.int64)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    block_y = gy // block
    block_x = gx // block
    return (block_y * wb + block_x).reshape(-1)


def sample_block_batch(
    h: int, w: int, n_blocks: int, *, block: int = 4, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``n_blocks`` random 4x4 blocks and return per-texel + per-block coords.

    Returns ``(uv, st, texel_to_block)`` where:

    - ``uv`` is ``(n_blocks * block * block, 2)`` texel-center UV coords.
    - ``st`` is ``(n_blocks, 2)`` block-center ST coords.
    - ``texel_to_block`` is ``(n_blocks * block * block,)`` mapping each texel to
      its corresponding row in ``st``.
    """
    hb, wb = h // block, w // block
    block_ids = torch.randint(
        0, hb * wb, (n_blocks,), generator=generator, dtype=torch.int64
    )
    by = block_ids // wb
    bx = block_ids % wb
    st = torch.stack([(bx + 0.5) / wb, (by + 0.5) / hb], dim=-1).float()

    dy, dx = torch.meshgrid(
        torch.arange(block), torch.arange(block), indexing="ij"
    )
    dy = dy.reshape(-1)
    dx = dx.reshape(-1)
    # Broadcast block (B,) with intra-block offsets (16,) -> (B, 16).
    ty = by[:, None] * block + dy[None, :]
    tx = bx[:, None] * block + dx[None, :]
    uv = torch.stack([(tx.float() + 0.5) / w, (ty.float() + 0.5) / h], dim=-1)
    uv = uv.reshape(-1, 2)
    texel_to_block = torch.arange(n_blocks).repeat_interleave(block * block)
    return uv, st, texel_to_block
