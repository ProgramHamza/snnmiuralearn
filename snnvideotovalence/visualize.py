"""Visualization helpers for emotion predictions and training metrics."""

from pathlib import Path

import matplotlib.pyplot as plt

from config import OUTPUT_DIR, ensure_dirs


def emotion_quadrant(valence, arousal):
    """Return the named valence-arousal quadrant."""
    if valence >= 0 and arousal >= 0:
        return "EXCITED"
    if valence < 0 and arousal >= 0:
        return "FEARFUL"
    if valence >= 0 and arousal < 0:
        return "CALM"
    return "BORED/SAD"


def plot_va_point(valence, arousal, save_path):
    """Plot one valence-arousal point on a 2D emotion plane."""
    ensure_dirs()
    save_path = Path(save_path)
    try:
        state = emotion_quadrant(valence, arousal)
        magnitude = (float(valence) ** 2 + float(arousal) ** 2) ** 0.5
        color = "green" if valence >= 0 else "red"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)
        ax.scatter([valence], [arousal], s=220, c=color, edgecolors="black", zorder=3)
        ax.text(0.55, 0.82, "EXCITED", ha="center", va="center", fontsize=11)
        ax.text(-0.55, 0.82, "FEARFUL", ha="center", va="center", fontsize=11)
        ax.text(0.55, -0.82, "CALM", ha="center", va="center", fontsize=11)
        ax.text(-0.55, -0.82, "BORED/SAD", ha="center", va="center", fontsize=11)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title(f"{state} emotion state, magnitude {magnitude:.2f}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        raise RuntimeError(f"Could not save valence-arousal plot to {save_path}: {exc}") from exc


def plot_training_curves(train_losses, val_losses, save_path, val_r_v=None, val_r_a=None):
    """Plot loss curves and Pearson r curves over epochs."""
    ensure_dirs()
    save_path = Path(save_path)
    try:
        epochs = list(range(1, len(train_losses) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(epochs, train_losses, label="train_loss")
        axes[0].plot(epochs, val_losses, label="val_loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        val_r_v = val_r_v or []
        val_r_a = val_r_a or []
        if val_r_v:
            axes[1].plot(epochs[: len(val_r_v)], val_r_v, label="valence r")
        if val_r_a:
            axes[1].plot(epochs[: len(val_r_a)], val_r_a, label="arousal r")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Pearson r")
        axes[1].set_ylim(-1, 1)
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        raise RuntimeError(f"Could not save training curves to {save_path}: {exc}") from exc

