"""Reference BC1 encoder and decoder.

The encoder uses min/max bounding-box endpoint selection in RGB565 space, then
performs a small number of Lloyd-style refinement passes (re-fit endpoints to
the centroid of texels assigned to each end of the palette). This mirrors the
classic real-time BC1 encoders and is the same family of algorithm used as the
seminar baseline (the NTBC paper used AMD Compressonator with two refine steps).

All array layout assumptions:

- Input image: ``(H, W, 3)`` uint8 or float in [0, 1].
- Block layout: row-major over 4x4 blocks (block_y, block_x).
- ``endpoints`` arrays: ``(n_blocks, 2, 3)`` uint8 in RGB888 (already dequantized
  from the encoded RGB565, so they match what a GPU would see).
- ``indices`` arrays: ``(n_blocks, 16)`` uint8 with values in {0, 1, 2, 3}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import blockify, rgb888_to_rgb565, to_uint8, unblockify


@dataclass
class BC1Encoded:
    """Structured representation of a BC1-compressed texture."""

    endpoints: np.ndarray   # (n_blocks, 2, 3) uint8, RGB888 after RGB565 quantization
    indices: np.ndarray     # (n_blocks, 16) uint8 in {0, 1, 2, 3}
    height: int
    width: int

    @property
    def n_blocks(self) -> int:
        return self.endpoints.shape[0]

    @property
    def storage_bytes(self) -> int:
        return 8 * self.n_blocks


def _build_palette(e0: np.ndarray, e1: np.ndarray) -> np.ndarray:
    """4-color palette from a pair of RGB endpoints, always assuming color0 > color1.

    Returns shape ``(n_blocks, 4, 3)`` float32.
    """
    e0 = e0.astype(np.float32)
    e1 = e1.astype(np.float32)
    c2 = (2.0 * e0 + e1) / 3.0
    c3 = (e0 + 2.0 * e1) / 3.0
    return np.stack([e0, e1, c2, c3], axis=1)


def _assign_indices(palette: np.ndarray, texels: np.ndarray) -> np.ndarray:
    """For each texel, pick the palette entry with the smallest squared distance.

    ``palette``: (n_blocks, 4, 3); ``texels``: (n_blocks, 16, 3). Returns
    ``(n_blocks, 16)`` uint8 indices.
    """
    diff = texels[:, :, None, :] - palette[:, None, :, :]
    d2 = (diff * diff).sum(axis=-1)
    return np.argmin(d2, axis=-1).astype(np.uint8)


def _initial_endpoints(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min/max bounding-box endpoints per block. ``blocks``: (N, 16, 3) uint8."""
    e0 = blocks.max(axis=1)
    e1 = blocks.min(axis=1)
    return e0, e1


def _refine(
    e0: np.ndarray, e1: np.ndarray, texels: np.ndarray, n_iters: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lloyd-style refinement: alternate assignment and endpoint relocation.

    Updates endpoints by least-squares fitting along the assignment-derived
    palette parameter t in {0, 1, 2/3, 1/3}. This is a standard cheap refinement
    pass used in real-time BC1 encoders.
    """
    for _ in range(n_iters):
        # Encode endpoints through RGB565 so the assignment is performed against
        # the same palette the decoder will see.
        _, e0_q = rgb888_to_rgb565(e0)
        _, e1_q = rgb888_to_rgb565(e1)
        palette = _build_palette(e0_q, e1_q)
        idx = _assign_indices(palette, texels)
        # Per-index weight in the e0->e1 interpolation: t = 0, 1, 1/3, 2/3.
        t_table = np.array([0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0], dtype=np.float32)
        t = t_table[idx]  # (n_blocks, 16)
        # Least-squares fit per channel for c = (1 - t) * e0 + t * e1 across the 16 texels.
        # Solve a 2x2 system per block.
        w0 = 1.0 - t
        w1 = t
        a00 = (w0 * w0).sum(axis=1)
        a01 = (w0 * w1).sum(axis=1)
        a11 = (w1 * w1).sum(axis=1)
        det = a00 * a11 - a01 * a01
        det_safe = np.where(np.abs(det) < 1e-6, 1.0, det)
        c = texels.astype(np.float32)
        b0 = (w0[..., None] * c).sum(axis=1)  # (n_blocks, 3)
        b1 = (w1[..., None] * c).sum(axis=1)
        new_e0 = (a11[:, None] * b0 - a01[:, None] * b1) / det_safe[:, None]
        new_e1 = (a00[:, None] * b1 - a01[:, None] * b0) / det_safe[:, None]
        # Fall back to original endpoints when the system is degenerate (a flat block).
        degenerate = np.abs(det) < 1e-6
        new_e0[degenerate] = e0[degenerate]
        new_e1[degenerate] = e1[degenerate]
        e0 = np.clip(new_e0, 0, 255).astype(np.uint8)
        e1 = np.clip(new_e1, 0, 255).astype(np.uint8)
    return e0, e1, _assign_indices(_build_palette(*rgb888_to_rgb565_pair(e0, e1)), texels)


def rgb888_to_rgb565_pair(e0: np.ndarray, e1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, e0_q = rgb888_to_rgb565(e0)
    _, e1_q = rgb888_to_rgb565(e1)
    return e0_q, e1_q


def _enforce_4color_mode(e0: np.ndarray, e1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make sure color0 > color1 in RGB565 packed order so the 4-color palette is used."""
    p0, _ = rgb888_to_rgb565(e0)
    p1, _ = rgb888_to_rgb565(e1)
    swap = p0 < p1
    if swap.any():
        tmp = e0.copy()
        e0 = np.where(swap[:, None], e1, e0)
        e1 = np.where(swap[:, None], tmp, e1)
        # If they are equal we leave them alone; the resulting block is a flat
        # color and the palette collapses to one entry which is fine for decoding.
    return e0, e1


def bc1_encode(image: np.ndarray, *, refine_steps: int = 2) -> BC1Encoded:
    """Encode an HxWx3 image as BC1.

    ``image`` may be either uint8 in [0, 255] or float in [0, 1]. The output
    endpoints are dequantized RGB888 (i.e. the same values a GPU decoder would
    reconstruct from the stored RGB565 codewords).
    """
    img = to_uint8(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"BC1 expects an HxWx3 image, got {img.shape}")
    h, w = img.shape[:2]
    blocks = blockify(img, block=4).reshape(-1, 16, 3)
    e0, e1 = _initial_endpoints(blocks)
    if refine_steps > 0:
        e0, e1, indices = _refine(e0, e1, blocks, n_iters=refine_steps)
    else:
        e0_q, e1_q = rgb888_to_rgb565_pair(e0, e1)
        indices = _assign_indices(_build_palette(e0_q, e1_q), blocks)
    e0, e1 = _enforce_4color_mode(e0, e1)
    # Snap endpoints to their RGB565-dequantized values so the stored
    # representation matches what a GPU decoder would see.
    _, e0_q = rgb888_to_rgb565(e0)
    _, e1_q = rgb888_to_rgb565(e1)
    endpoints = np.stack([e0_q, e1_q], axis=1)
    palette = _build_palette(e0_q, e1_q)
    indices = _assign_indices(palette, blocks)
    return BC1Encoded(endpoints=endpoints, indices=indices, height=h, width=w)


def bc1_decode(encoded: BC1Encoded) -> np.ndarray:
    """Decode a :class:`BC1Encoded` back to an HxWx3 uint8 image."""
    e0 = encoded.endpoints[:, 0]
    e1 = encoded.endpoints[:, 1]
    palette = _build_palette(e0, e1)
    nb = encoded.n_blocks
    flat = np.take_along_axis(palette, encoded.indices[:, :, None], axis=1)
    blocks = flat.reshape(nb, 4, 4, 3).clip(0, 255).astype(np.uint8)
    return unblockify(blocks, encoded.height, encoded.width, block=4)


def bc1_encode_decode(image: np.ndarray, *, refine_steps: int = 2) -> np.ndarray:
    """Convenience: encode then decode an image, returning the reconstruction."""
    return bc1_decode(bc1_encode(image, refine_steps=refine_steps))
