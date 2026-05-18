from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from config import FEATURE_LENGTH, HIDDEN, TAU_D, TAU_R
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class AlphaLIFNode(neuron.ParametricLIFNode):
    def __init__(
        self,
        init_tau: float = TAU_D,
        tau_r: float = TAU_R,
        tau_d: float = TAU_D,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan(),
        detach_reset: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        store_v_seq: bool = False,
    ):
        super().__init__(
            init_tau=init_tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )
        self.tau_r = float(tau_r)
        self.tau_d = float(tau_d)
        self._alpha_rise: Optional[torch.Tensor] = None
        self._alpha_decay: Optional[torch.Tensor] = None

    def reset(self):
        self._alpha_rise = None
        self._alpha_decay = None
        if hasattr(super(), "reset"):
            super().reset()

    def _alpha_filter(self, x: torch.Tensor) -> torch.Tensor:
        if self._alpha_rise is None or self._alpha_rise.shape != x.shape or self._alpha_rise.device != x.device:
            self._alpha_rise = torch.zeros_like(x)
            self._alpha_decay = torch.zeros_like(x)

        self._alpha_rise = self._alpha_rise + (x - self._alpha_rise) / self.tau_r
        self._alpha_decay = self._alpha_decay + (x - self._alpha_decay) / self.tau_d
        return self._alpha_rise - self._alpha_decay

    def neuronal_charge(self, x: torch.Tensor):
        alpha_current = self._alpha_filter(x)
        return super().neuronal_charge(alpha_current)


class ConvSNN(nn.Module):
    def __init__(self, input_channels: int, feature_length: int = FEATURE_LENGTH):
        super().__init__()
        pooled_length = feature_length // 2
        hidden_in_features = 256 * pooled_length

        self.feature_length = feature_length
        self.input_channels = input_channels

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.neuron1 = AlphaLIFNode()

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.neuron2 = AlphaLIFNode()

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.neuron3 = AlphaLIFNode()
        self.pool = layer.AvgPool1d(2)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.neuron4 = AlphaLIFNode()

        self.flatten = layer.Flatten()
        self.fc = nn.Linear(hidden_in_features, HIDDEN)
        self.hidden_neuron = AlphaLIFNode(init_tau=TAU_D, tau_r=TAU_R, tau_d=TAU_D)

        self.affect_head = nn.Sequential(
            nn.Linear(HIDDEN, 2),
            nn.Tanh(),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(HIDDEN, 7),
            nn.Sigmoid(),
        )

    def _forward_timestep(self, x_t: torch.Tensor) -> torch.Tensor:
        x_t = self.conv1(x_t)
        x_t = self.bn1(x_t)
        x_t = self.neuron1(x_t)

        x_t = self.conv2(x_t)
        x_t = self.bn2(x_t)
        x_t = self.neuron2(x_t)

        x_t = self.conv3(x_t)
        x_t = self.bn3(x_t)
        x_t = self.neuron3(x_t)
        x_t = self.pool(x_t)

        x_t = self.conv4(x_t)
        x_t = self.bn4(x_t)
        x_t = self.neuron4(x_t)

        x_t = self.flatten(x_t)
        x_t = self.fc(x_t)
        _ = self.hidden_neuron(x_t)
        return self.hidden_neuron.v

    def forward(self, spike_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        functional.reset_net(self)

        if spike_tensor.dim() == 3:
            spike_tensor = spike_tensor.unsqueeze(1)
        if spike_tensor.dim() != 4:
            raise ValueError("ConvSNN expects input shaped [T, B, C, F].")

        time_steps = spike_tensor.shape[0]
        membrane_trace = []
        for time_index in range(time_steps):
            hidden_membrane = self._forward_timestep(spike_tensor[time_index])
            membrane_trace.append(hidden_membrane)

        hidden_mean = torch.stack(membrane_trace, dim=0).mean(dim=0)
        affect_prediction = torch.clamp(self.affect_head(hidden_mean), -1.0, 1.0)
        emotion_prediction = self.emotion_head(hidden_mean) * 100.0
        return affect_prediction, emotion_prediction
