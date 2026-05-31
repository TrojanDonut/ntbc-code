"""Shared utilities for BC1 / BC4 encoders and decoders."""

from __future__ import annotations

import numpy as np


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a float [0, 1] array to uint8 with rounding, or pass through uint8."""
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_float01(arr: np.ndarray) -> np.ndarray:
    """Convert a uint8 array to float32 in [0, 1]."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def blockify(img: np.ndarray, block: int = 4) -> np.ndarray:
    """Split an HxWxC array into a (nblocks, block, block, C) array of 4x4 blocks.

    The block layout is row-major in (block_y, block_x).
    """
    h, w = img.shape[:2]
    if h % block or w % block:
        raise ValueError(
            f"image dims ({h}x{w}) must be multiples of block size {block}"
        )
    c = img.shape[2] if img.ndim == 3 else 1
    nh, nw = h // block, w // block
    img = img.reshape(nh, block, nw, block, c)
    img = img.transpose(0, 2, 1, 3, 4).reshape(nh * nw, block, block, c)
    return img


def unblockify(blocks: np.ndarray, h: int, w: int, block: int = 4) -> np.ndarray:
    """Inverse of :func:`blockify`."""
    nh, nw = h // block, w // block
    c = blocks.shape[-1]
    arr = blocks.reshape(nh, nw, block, block, c)
    arr = arr.transpose(0, 2, 1, 3, 4).reshape(h, w, c)
    return arr


def rgb888_to_rgb565(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize an (..., 3) uint8 RGB array to RGB565.

    Returns ``(packed, rgb8)`` where ``packed`` is a uint16 with the bit layout
    ``rrrrr ggggggg bbbbb`` and ``rgb8`` is the dequantized RGB888 reconstruction
    using the standard bit-shift expansion that GPUs use to decode BC1.
    """
    rgb = rgb.astype(np.uint16)
    r = ((rgb[..., 0].astype(np.float32) * 31.0 / 255.0).round()).clip(0, 31).astype(np.uint16)
    g = ((rgb[..., 1].astype(np.float32) * 63.0 / 255.0).round()).clip(0, 63).astype(np.uint16)
    b = ((rgb[..., 2].astype(np.float32) * 31.0 / 255.0).round()).clip(0, 31).astype(np.uint16)
    packed = (r << 11) | (g << 5) | b
    r8 = ((r << 3) | (r >> 2)).astype(np.uint8)
    g8 = ((g << 2) | (g >> 4)).astype(np.uint8)
    b8 = ((b << 3) | (b >> 2)).astype(np.uint8)
    rgb8 = np.stack([r8, g8, b8], axis=-1)
    return packed, rgb8


def bc1_storage_bytes(n_blocks: int) -> int:
    """BC1 uses 8 bytes per 4x4 block."""
    return 8 * n_blocks


def bc4_storage_bytes(n_blocks: int) -> int:
    """BC4 uses 8 bytes per 4x4 block (same as BC1)."""
    return 8 * n_blocks


def n_blocks(h: int, w: int, block: int = 4) -> int:
    return (h // block) * (w // block)
