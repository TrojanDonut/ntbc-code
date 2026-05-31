"""Multi-resolution dense feature grid with optional 8-bit quantization-aware
training.

This is a pure-PyTorch substitute for the tiny-cuda-nn ``HashGrid`` used by the
NTBC paper (which requires NVIDIA-only CUDA kernels). For the texture sizes we
target on CPU / weak GPUs (<=512^2), a dense multi-resolution grid is small
enough that we do not need spatial hashing -- storage is comparable to the
hashed variant at the paper's 4k scale.

Quantization-aware training (QAT) emulates the 8-bit integer storage used at
inference time. When QAT is enabled, each level's feature tensor is
fake-quantized with a per-level symmetric scale before bilinear sampling. The
backward pass uses a straight-through estimator so gradients can flow through
the rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def geometric_levels(coarsest: int, finest: int, n_levels: int) -> list[int]:
    """Return ``n_levels`` integer grid resolutions geometrically spaced between
    ``coarsest`` and ``finest`` (both inclusive).

    The NTBC paper uses a geometric progression for its multi-resolution grids.
    """
    if n_levels < 2:
        raise ValueError("need at least 2 levels")
    factor = (finest / coarsest) ** (1.0 / (n_levels - 1))
    levels: list[int] = []
    for i in range(n_levels):
        levels.append(max(2, int(round(coarsest * (factor ** i)))))
    levels[-1] = finest  # snap the last level exactly to the requested finest
    levels[0] = coarsest
    return levels


def _fake_quantize_int8(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 fake-quantization with a straight-through estimator.

    ``scale`` maps the range [-127, 127] back to the float domain. The forward
    pass rounds to the nearest int8 code and dequantizes; the backward pass is
    the identity.
    """
    quantized = torch.round((t / scale).clamp(-127, 127)) * scale
    return t + (quantized - t).detach()


@dataclass
class MultiResGridConfig:
    n_levels: int = 6
    coarsest: int = 16
    finest: int = 512
    n_features: int = 2
    init_std: float = 1e-3


class MultiResFeatureGrid(nn.Module):
    """Multi-resolution dense feature grid over the unit square.

    Looks up bilinear features at ``n_levels`` resolutions and concatenates them.
    Inputs are ``(N, 2)`` coordinates in ``[0, 1]``. Output is
    ``(N, n_levels * n_features)``.
    """

    def __init__(self, config: MultiResGridConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = MultiResGridConfig(**kwargs)
        self.config = config
        self.levels: list[int] = geometric_levels(
            config.coarsest, config.finest, config.n_levels
        )
        self.features = nn.ParameterList()
        self.qat_scales = nn.ParameterList()
        for L in self.levels:
            param = nn.Parameter(torch.randn(1, config.n_features, L, L) * config.init_std)
            self.features.append(param)
            # Symmetric int8 scale per level; initialized large so QAT is a no-op
            # until enabled. Calibration happens via enable_qat().
            self.qat_scales.append(nn.Parameter(torch.tensor(0.05), requires_grad=False))
        self._qat: bool = False

    @property
    def output_dim(self) -> int:
        return self.config.n_features * self.config.n_levels

    def storage_bytes(self, fp_bytes_per_element: int = 1) -> int:
        """Total storage required for the grid features (in bytes).

        With ``fp_bytes_per_element=1`` this counts the 8-bit-quantized inference
        size used in the seminar report.
        """
        n = sum(self.config.n_features * (L * L) for L in self.levels)
        extra = 4 * len(self.levels)  # one fp32 scale per level
        return n * fp_bytes_per_element + extra

    def enable_qat(self, on: bool = True) -> None:
        """Toggle 8-bit fake-quantization during forward passes."""
        self._qat = on
        if on:
            with torch.no_grad():
                for param, scale in zip(self.features, self.qat_scales):
                    max_abs = param.detach().abs().max().clamp_min(1e-6)
                    scale.copy_(max_abs / 127.0)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Sample concatenated features at the given UV coordinates.

        ``uv``: ``(N, 2)`` in ``[0, 1]``. Returns ``(N, output_dim)``.
        """
        if uv.dim() != 2 or uv.size(-1) != 2:
            raise ValueError(f"uv must be (N, 2), got {tuple(uv.shape)}")
        # grid_sample expects (N, H_out, W_out, 2) and coords in [-1, 1].
        # We use H_out = N, W_out = 1 so the per-coordinate output is a "column".
        coords = uv * 2.0 - 1.0
        grid = coords.view(1, -1, 1, 2)  # batch=1, N points, single col
        outputs: list[torch.Tensor] = []
        for feat, scale in zip(self.features, self.qat_scales):
            f = feat
            if self._qat:
                f = _fake_quantize_int8(f, scale)
            # grid_sample defaults: bilinear interpolation, zero padding, align_corners=False.
            sampled = F.grid_sample(
                f, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            # sampled: (1, C, N, 1) -> (N, C)
            sampled = sampled.squeeze(-1).squeeze(0).transpose(0, 1)
            outputs.append(sampled)
        return torch.cat(outputs, dim=-1)
