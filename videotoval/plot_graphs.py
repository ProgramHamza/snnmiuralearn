from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def ensure_plot_dir(plot_dir: str | Path) -> Path:
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    return plot_path


def _save_figure(fig: plt.Figure, plot_dir: str | Path, filename: str) -> Path:
    plot_path = ensure_plot_dir(plot_dir)
    output_path = plot_path / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_loss_curves(history: Mapping[str, Sequence[float]], plot_dir: str | Path) -> Path:
    epochs = np.arange(1, len(history.get("train_total_loss", [])) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    panels = [
        ("train_mse_valence", "val_mse_valence", "Valence MSE"),
        ("train_mse_arousal", "val_mse_arousal", "Arousal MSE"),
        ("train_ccc_valence", "val_ccc_valence", "Valence CCC Loss"),
        ("train_ccc_arousal", "val_ccc_arousal", "Arousal CCC Loss"),
    ]

    for ax, (train_key, val_key, title) in zip(axes.flat, panels):
        ax.plot(epochs, history.get(train_key, []), label="train")
        ax.plot(epochs, history.get(val_key, []), label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle("Train vs Val Loss Components")
    return _save_figure(fig, plot_dir, "loss_components.png")


def plot_total_loss(history: Mapping[str, Sequence[float]], plot_dir: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = np.arange(1, len(history.get("train_total_loss", [])) + 1)
    ax.plot(epochs, history.get("train_total_loss", []), label="train total")
    ax.plot(epochs, history.get("val_total_loss", []), label="val total")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_figure(fig, plot_dir, "total_loss.png")


def plot_learning_rate(history: Mapping[str, Sequence[float]], plot_dir: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = np.arange(1, len(history.get("learning_rates", [])) + 1)
    ax.plot(epochs, history.get("learning_rates", []), color="tab:orange")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, plot_dir, "learning_rate.png")


def plot_gradient_norms(history: Mapping[str, Mapping[str, Sequence[float]]], plot_dir: str | Path) -> Path:
    gradient_norms = history.get("gradient_norms", {})
    fig, ax = plt.subplots(figsize=(10, 5))
    for layer_name, values in gradient_norms.items():
        ax.plot(np.arange(1, len(values) + 1), values, label=layer_name)
    ax.set_title("Gradient Norm per Layer")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient norm")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_figure(fig, plot_dir, "gradient_norms.png")


def plot_spike_rates(history: Mapping[str, Mapping[str, Sequence[float]]], plot_dir: str | Path) -> Path:
    spike_rates = history.get("spike_rates", {})
    fig, ax = plt.subplots(figsize=(10, 5))
    for layer_name, values in spike_rates.items():
        ax.plot(np.arange(1, len(values) + 1), values, label=layer_name)
    ax.axhspan(0.10, 0.30, color="green", alpha=0.12, label="healthy range")
    ax.set_title("Spike Rate per Layer")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average firing rate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_figure(fig, plot_dir, "spike_rates.png")


def plot_beta_values(history: Mapping[str, Mapping[str, Sequence[float]]], plot_dir: str | Path) -> Path:
    beta_values = history.get("beta_values", {})
    fig, ax = plt.subplots(figsize=(10, 5))
    for layer_name, values in beta_values.items():
        ax.plot(np.arange(1, len(values) + 1), values, label=layer_name)
    ax.set_title("Learned Beta Values")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Beta")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_figure(fig, plot_dir, "beta_values.png")


def plot_scatter(predictions: np.ndarray, targets: np.ndarray, plot_dir: str | Path, split_name: str = "val") -> Path:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Valence", "Arousal"]

    for idx, ax in enumerate(axes):
        ax.scatter(targets[:, idx], predictions[:, idx], s=12, alpha=0.7)
        low = min(targets[:, idx].min(), predictions[:, idx].min())
        high = max(targets[:, idx].max(), predictions[:, idx].max())
        ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        ax.set_title(f"{split_name.capitalize()} {labels[idx]}: Predicted vs Ground Truth")
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.25)

    return _save_figure(fig, plot_dir, f"{split_name}_scatter.png")


def plot_timeseries(times: Sequence[float], predictions: np.ndarray, targets: np.ndarray, plot_dir: str | Path, sample_name: str = "sample") -> Path:
    times = np.asarray(times)
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    labels = ["Valence", "Arousal"]

    for idx, ax in enumerate(axes):
        ax.plot(times, targets[:, idx], label="ground truth", linewidth=2)
        ax.plot(times, predictions[:, idx], label="prediction", linewidth=1.8)
        ax.set_ylabel(labels[idx])
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Time (sec)")
    fig.suptitle(f"Predicted vs Ground Truth Timeseries: {sample_name}")
    return _save_figure(fig, plot_dir, f"{sample_name}_timeseries.png")


def plot_membrane_trace(times: Sequence[float], membrane_trace: np.ndarray, plot_dir: str | Path, sample_name: str = "sample") -> Path:
    times = np.asarray(times)
    membrane_trace = np.asarray(membrane_trace)
    if membrane_trace.ndim == 1:
        membrane_trace = membrane_trace[:, None]

    fig, ax = plt.subplots(figsize=(12, 5))
    n_neurons = membrane_trace.shape[1]
    for idx in range(min(n_neurons, 4)):
        ax.plot(times, membrane_trace[:, idx], label=f"neuron {idx}")
    ax.set_title(f"Membrane Potential Trace: {sample_name}")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Membrane potential")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_figure(fig, plot_dir, f"{sample_name}_membrane_trace.png")


def plot_prediction_timeseries(times: Sequence[float], predictions: np.ndarray, plot_dir: str | Path, sample_name: str = "sample") -> Path:
    times = np.asarray(times)
    predictions = np.asarray(predictions)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    labels = ["Valence", "Arousal"]

    for idx, ax in enumerate(axes):
        ax.plot(times, predictions[:, idx], label="prediction", linewidth=2)
        ax.set_ylabel(labels[idx])
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Time (sec)")
    fig.suptitle(f"Prediction Timeseries: {sample_name}")
    return _save_figure(fig, plot_dir, f"{sample_name}_prediction_timeseries.png")
