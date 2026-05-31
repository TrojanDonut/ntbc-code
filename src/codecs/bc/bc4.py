"""Reference BC4 encoder and decoder (unsigned 8-bit single-channel).

The encoder uses min/max endpoints with a few Lloyd refinement passes; the
8-color "red0 > red1" palette mode is always used so the index space is the
full 8-entry palette (0..7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import blockify, to_uint8, unblockify


@dataclass
class BC4Encoded:
    """Structured representation of a BC4-compressed texture."""

    endpoints: np.ndarray   # (n_blocks, 2) uint8 -- (red0, red1)
    indices: np.ndarray     # (n_blocks, 16) uint8 in {0,...,7}
    height: int
    width: int

    @property
    def n_blocks(self) -> int:
        return self.endpoints.shape[0]

    @property
    def storage_bytes(self) -> int:
        return 8 * self.n_blocks


def _build_palette_bc4(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """8-color BC4 palette (red0 > red1 mode). Returns (n_blocks, 8) float32."""
    r0 = r0.astype(np.float32)
    r1 = r1.astype(np.float32)
    palette = np.stack(
        [
            r0,
            r1,
            (6.0 * r0 + 1.0 * r1) / 7.0,
            (5.0 * r0 + 2.0 * r1) / 7.0,
            (4.0 * r0 + 3.0 * r1) / 7.0,
            (3.0 * r0 + 4.0 * r1) / 7.0,
            (2.0 * r0 + 5.0 * r1) / 7.0,
            (1.0 * r0 + 6.0 * r1) / 7.0,
        ],
        axis=1,
    )
    return palette


def _assign_indices_bc4(palette: np.ndarray, texels: np.ndarray) -> np.ndarray:
    """``palette``: (N, 8); ``texels``: (N, 16). Returns (N, 16) uint8."""
    diff = texels[:, :, None] - palette[:, None, :]
    d2 = diff * diff
    return np.argmin(d2, axis=-1).astype(np.uint8)


def _refine_bc4(
    r0: np.ndarray, r1: np.ndarray, texels: np.ndarray, n_iters: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd-style refinement of BC4 endpoints."""
    # Interpolation t for indices 0..7 along the r0 -> r1 axis.
    t_table = np.array(
        [0.0, 1.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0],
        dtype=np.float32,
    )
    for _ in range(n_iters):
        palette = _build_palette_bc4(r0, r1)
        idx = _assign_indices_bc4(palette, texels)
        t = t_table[idx]  # (N, 16)
        w0 = 1.0 - t
        w1 = t
        a00 = (w0 * w0).sum(axis=1)
        a01 = (w0 * w1).sum(axis=1)
        a11 = (w1 * w1).sum(axis=1)
        det = a00 * a11 - a01 * a01
        det_safe = np.where(np.abs(det) < 1e-6, 1.0, det)
        c = texels.astype(np.float32)
        b0 = (w0 * c).sum(axis=1)
        b1 = (w1 * c).sum(axis=1)
        new_r0 = (a11 * b0 - a01 * b1) / det_safe
        new_r1 = (a00 * b1 - a01 * b0) / det_safe
        degenerate = np.abs(det) < 1e-6
        new_r0 = np.where(degenerate, r0.astype(np.float32), new_r0)
        new_r1 = np.where(degenerate, r1.astype(np.float32), new_r1)
        r0 = np.clip(new_r0, 0, 255).astype(np.uint8)
        r1 = np.clip(new_r1, 0, 255).astype(np.uint8)
    return r0, r1


def _enforce_8color_mode_bc4(r0: np.ndarray, r1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure red0 > red1 so the 8-interpolated-color palette mode is selected."""
    swap = r0 < r1
    if swap.any():
        tmp = r0.copy()
        r0 = np.where(swap, r1, r0)
        r1 = np.where(swap, tmp, r1)
    return r0, r1


def bc4_encode(image: np.ndarray, *, refine_steps: int = 2) -> BC4Encoded:
    """Encode an HxW (or HxWx1) single-channel image as BC4."""
    img = to_uint8(image)
    if img.ndim == 3:
        if img.shape[2] != 1:
            raise ValueError(f"BC4 expects single-channel, got {img.shape}")
        img = img[:, :, 0]
    h, w = img.shape
    blocks = blockify(img[..., None], block=4).reshape(-1, 16)
    r0 = blocks.max(axis=1)
    r1 = blocks.min(axis=1)
    if refine_steps > 0:
        r0, r1 = _refine_bc4(r0, r1, blocks, n_iters=refine_steps)
    r0, r1 = _enforce_8color_mode_bc4(r0, r1)
    palette = _build_palette_bc4(r0, r1)
    indices = _assign_indices_bc4(palette, blocks)
    endpoints = np.stack([r0, r1], axis=1)
    return BC4Encoded(endpoints=endpoints, indices=indices, height=h, width=w)


def bc4_decode(encoded: BC4Encoded) -> np.ndarray:
    """Decode a :class:`BC4Encoded` back to an HxWx1 uint8 image."""
    r0 = encoded.endpoints[:, 0]
    r1 = encoded.endpoints[:, 1]
    palette = _build_palette_bc4(r0, r1)  # (N, 8)
    flat = np.take_along_axis(palette, encoded.indices.astype(np.int64), axis=1)  # (N, 16)
    blocks = flat.reshape(-1, 4, 4, 1).clip(0, 255).astype(np.uint8)
    return unblockify(blocks, encoded.height, encoded.width, block=4)


def bc4_encode_decode(image: np.ndarray, *, refine_steps: int = 2) -> np.ndarray:
    return bc4_decode(bc4_encode(image, refine_steps=refine_steps))
