"""Run a trained NTBC model and the BC baseline on a material to produce
fully-reconstructed HxWxC uint8 textures per layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..codecs.bc.bc1 import bc1_encode_decode
from ..codecs.bc.bc4 import bc4_encode_decode
from ..data.material import BCFormat, Material, Role, TextureLayer
from ..models.coords import block_st_grid, texel_to_block_indices, texel_uv_grid
from ..models.ntbc import NTBC


@dataclass
class LayerReconstruction:
    """Reference + reconstruction for a single texture layer."""

    name: str
    role: Role
    bc_format: BCFormat
    reference: np.ndarray   # HxWxC uint8
    baseline: np.ndarray    # HxWxC uint8 from BC1/BC4 reference encoder
    ntbc: np.ndarray        # HxWxC uint8 from the NTBC model


def _layer_to_uint8(layer: TextureLayer) -> np.ndarray:
    arr = (layer.data.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def baseline_reconstruct(layer: TextureLayer, *, refine_steps: int = 2) -> np.ndarray:
    """Round-trip a layer through the pure-Python BC1/BC4 reference encoder."""
    ref = _layer_to_uint8(layer)
    if layer.bc_format == BCFormat.BC1:
        return bc1_encode_decode(ref, refine_steps=refine_steps)
    return bc4_encode_decode(ref, refine_steps=refine_steps)


@torch.no_grad()
def ntbc_reconstruct(
    model: NTBC, material: Material, *, chunk: int = 16384
) -> dict[str, np.ndarray]:
    """Decode every layer of a material using the trained NTBC model.

    Returns a dict mapping ``layer.name`` to ``HxWxC`` uint8 arrays. Decoding is
    chunked over texels to keep peak memory bounded on CPU.
    """
    model.eval()
    device = next(model.parameters()).device
    h = w = material.resolution

    uv = texel_uv_grid(h, w).to(device)
    st = block_st_grid(h, w, block=4).to(device)
    t2b = texel_to_block_indices(h, w, block=4).to(device)

    n_texels = uv.shape[0]
    bc1_decoded_chunks: list[torch.Tensor] = []
    bc4_decoded_chunks: list[torch.Tensor] = []

    for start in range(0, n_texels, chunk):
        end = min(start + chunk, n_texels)
        out = model(uv[start:end], st, t2b[start:end])
        if out.bc1_decoded is not None:
            bc1_decoded_chunks.append(out.bc1_decoded.detach().cpu())
        if out.bc4_decoded is not None:
            bc4_decoded_chunks.append(out.bc4_decoded.detach().cpu())

    result: dict[str, np.ndarray] = {}

    if bc1_decoded_chunks:
        bc1_full = torch.cat(bc1_decoded_chunks, dim=0)  # (H*W, n_bc1 * 3)
        bc1_full = bc1_full.reshape(h, w, model.n_bc1, 3)
        for i, layer in enumerate(material.bc1_layers):
            arr = (bc1_full[..., i, :].clamp(0, 1) * 255.0 + 0.5).clip(0, 255).numpy().astype(np.uint8)
            result[layer.name] = arr

    if bc4_decoded_chunks:
        bc4_full = torch.cat(bc4_decoded_chunks, dim=0)  # (H*W, n_bc4)
        bc4_full = bc4_full.reshape(h, w, model.n_bc4)
        for i, layer in enumerate(material.bc4_layers):
            arr = (bc4_full[..., i].clamp(0, 1) * 255.0 + 0.5).clip(0, 255).numpy().astype(np.uint8)
            result[layer.name] = arr[..., None]

    return result


def reconstruct_all(model: NTBC, material: Material) -> list[LayerReconstruction]:
    """Reference, baseline-BC, and NTBC reconstructions for every layer."""
    ntbc_outputs = ntbc_reconstruct(model, material)
    out: list[LayerReconstruction] = []
    for layer in material.layers:
        ref = _layer_to_uint8(layer)
        out.append(
            LayerReconstruction(
                name=layer.name,
                role=layer.role,
                bc_format=layer.bc_format,
                reference=ref,
                baseline=baseline_reconstruct(layer),
                ntbc=ntbc_outputs[layer.name],
            )
        )
    return out
