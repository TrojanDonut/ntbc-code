"""Small MLP head used by NTBC's color and endpoint networks.

Per Sec. 3.3 of the NTBC paper: 3 hidden layers x 64 neurons with SELU
activations, sigmoid on the output to land in ``[0, 1]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.SELU(inplace=True))
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                # LeCun-style init suits SELU (preserves self-normalizing property).
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def storage_bytes(self, fp_bytes_per_element: int = 2) -> int:
        """Total weight storage in bytes (defaults to fp16 storage on disk)."""
        return sum(p.numel() for p in self.parameters()) * fp_bytes_per_element
