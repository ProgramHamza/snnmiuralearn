from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import snntorch as snn
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 360, 640),
        hidden_sizes: Tuple[int, int] = (64, 32),
        output_size: int = 2,
        num_steps: int = 8,
        beta: float = 0.95,
        learn_beta: bool = True,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_steps = num_steps

        num_inputs = int(input_shape[0] * input_shape[1] * input_shape[2])

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.lif1 = snn.Leaky(beta=beta, learn_beta=learn_beta)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lif2 = snn.Leaky(beta=beta, learn_beta=learn_beta)
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.lif3 = snn.Leaky(beta=beta, learn_beta=learn_beta)

    def _beta_value(self, layer) -> float:
        beta = layer.beta
        if torch.is_tensor(beta):
            return float(beta.detach().mean().item())
        return float(beta)

    @property
    def beta_values(self) -> Dict[str, float]:
        return {
            "layer1": self._beta_value(self.lif1),
            "layer2": self._beta_value(self.lif2),
            "layer3": self._beta_value(self.lif3),
        }

    def forward(self, x, state: Optional[Dict[str, torch.Tensor]] = None, num_steps: Optional[int] = None):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.flatten(x)
        steps = self.num_steps if num_steps is None else num_steps

        if state and "mem_state" in state:
            mem_state = state["mem_state"]
            mem1 = mem_state.get("layer1", self.lif1.init_leaky())
            mem2 = mem_state.get("layer2", self.lif2.init_leaky())
            mem3 = mem_state.get("layer3", self.lif3.init_leaky())
        else:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()

        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        mem3_rec = []

        for _ in range(steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        spk1_stack = torch.stack(spk1_rec, dim=0)
        spk2_stack = torch.stack(spk2_rec, dim=0)
        spk3_stack = torch.stack(spk3_rec, dim=0)
        mem3_stack = torch.stack(mem3_rec, dim=0)

        pred = mem3_stack.mean(dim=0)
        next_state = {
            "mem_state": {
                "layer1": mem1,
                "layer2": mem2,
                "layer3": mem3,
            },
            "spikes": {
                "layer1": spk1_stack,
                "layer2": spk2_stack,
                "layer3": spk3_stack,
            },
            "membrane": {"output": mem3_stack},
            "spike_rates": {
                "layer1": float(spk1_stack.detach().mean().item()),
                "layer2": float(spk2_stack.detach().mean().item()),
                "layer3": float(spk3_stack.detach().mean().item()),
            },
            "beta_values": self.beta_values,
        }
        return pred, next_state


def save_model(
    model: Net,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    history: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> Path:
    checkpoint = {
        "model_class": model.__class__.__name__,
        "model_kwargs": {
            "input_shape": model.input_shape,
            "hidden_sizes": model.hidden_sizes,
            "output_size": model.output_size,
            "num_steps": model.num_steps,
        },
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if history is not None:
        checkpoint["history"] = history
    if extra is not None:
        checkpoint["extra"] = extra

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return output_path


def load_model(
    path: str | Path,
    map_location: str | torch.device | None = None,
    model_kwargs: Optional[dict] = None,
) -> tuple[Net, dict]:
    checkpoint = torch.load(path, map_location=map_location)
    kwargs = checkpoint.get("model_kwargs", {})
    if model_kwargs is not None:
        kwargs.update(model_kwargs)
    model = Net(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    if map_location is not None:
        model = model.to(map_location)
    return model, checkpoint
