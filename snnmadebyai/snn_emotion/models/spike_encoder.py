from __future__ import annotations

import torch
from torch import nn

from config import T_SPIKE


class RateSpikeEncoder(nn.Module):
    def __init__(self, timesteps: int = T_SPIKE):
        super().__init__()
        self.timesteps = timesteps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if features.dim() != 3:
            raise ValueError("RateSpikeEncoder expects input shaped [B, C, F] or [C, F].")

        probabilities = torch.sigmoid(features)
        sample_shape = (self.timesteps,) + tuple(probabilities.shape)
        random_values = torch.rand(sample_shape, device=probabilities.device, dtype=probabilities.dtype)
        spikes = (random_values < probabilities.unsqueeze(0)).to(probabilities.dtype)
        return spikes


def encode_rate(features: torch.Tensor, timesteps: int = T_SPIKE) -> torch.Tensor:
    return RateSpikeEncoder(timesteps=timesteps)(features)
