"""BC1 / BC4 reference encoders and decoders."""

from .bc1 import BC1Encoded, bc1_decode, bc1_encode, bc1_encode_decode
from .bc4 import BC4Encoded, bc4_decode, bc4_encode, bc4_encode_decode
from .compressonator import compressonator_roundtrip, has_compressonator
from .torch_ops import (
    build_bc1_palette,
    build_bc4_palette,
    hard_assign_bc1,
    hard_assign_bc4,
    quantize_r8_ste,
    quantize_rgb565_ste,
)
from .utils import bc1_storage_bytes, bc4_storage_bytes, n_blocks

__all__ = [
    "BC1Encoded",
    "BC4Encoded",
    "bc1_decode",
    "bc1_encode",
    "bc1_encode_decode",
    "bc1_storage_bytes",
    "bc4_decode",
    "bc4_encode",
    "bc4_encode_decode",
    "bc4_storage_bytes",
    "build_bc1_palette",
    "build_bc4_palette",
    "compressonator_roundtrip",
    "hard_assign_bc1",
    "hard_assign_bc4",
    "has_compressonator",
    "n_blocks",
    "quantize_r8_ste",
    "quantize_rgb565_ste",
]
