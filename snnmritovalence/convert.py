"""Convert the trained ANN to an SNN with SpikingJelly ann2snn."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import (
    ANN_CHECKPOINT,
    BATCH_SIZE,
    CASE_ROOT,
    CLIP_DURATION,
    DEVICE,
    FPS,
    IMG_SIZE,
    N_MFCC,
    OUTPUT_DIR,
    SNN_CHECKPOINT,
    SR,
    VAL_SPLIT,
    ensure_dirs,
)
from dataset import CASEVideoDataset, collate_fn
from model import EmotionANN
from train import pearsonr_safe


class PackedInputs:
    """Container for two-input batches that still supports .to(device)."""

    def __init__(self, frames, mfcc):
        """Store frame and MFCC tensors."""
        self.frames = frames
        self.mfcc = mfcc

    def to(self, device):
        """Move both tensors to a device and return a new packed input."""
        return PackedInputs(self.frames.to(device), self.mfcc.to(device))


def load_ann():
    """Load the trained ANN checkpoint."""
    if not Path(ANN_CHECKPOINT).exists():
        raise FileNotFoundError(f"ANN checkpoint not found: {ANN_CHECKPOINT}")
    try:
        checkpoint = torch.load(ANN_CHECKPOINT, map_location=DEVICE)
        model = EmotionANN(N_MFCC).to(DEVICE)
        state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as exc:
        raise RuntimeError(f"Could not load ANN checkpoint {ANN_CHECKPOINT}: {exc}") from exc


def calibration_collate(batch):
    """Collate calibration data as a packed two-input tuple for ann2snn."""
    collated = collate_fn(batch)
    return PackedInputs(collated["frames"], collated["mfcc"])


def model_outputs(model, loader):
    """Collect model predictions and targets from a dataloader."""
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(DEVICE)
            mfcc = batch["mfcc"].to(DEVICE)
            out = model(frames, mfcc)
            preds.append(out.detach().cpu().numpy())
            targets.append(torch.stack([batch["valence"], batch["arousal"]], dim=1).numpy())
    if not preds:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(preds), np.concatenate(targets)


def plot_ann_snn_comparison(ann_preds, snn_preds, save_path):
    """Save a scatter comparison of ANN and SNN predictions."""
    save_path = Path(save_path)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = ["Valence", "Arousal"]
        for i, ax in enumerate(axes):
            ax.scatter(ann_preds[:, i], snn_preds[:, i], alpha=0.5)
            ax.plot([-1, 1], [-1, 1], color="black", linewidth=1)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xlabel(f"ANN {labels[i]}")
            ax.set_ylabel(f"SNN {labels[i]}")
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        raise RuntimeError(f"Could not save comparison plot to {save_path}: {exc}") from exc


def convert_ann_to_snn(ann, calibration_loader):
    """Run SpikingJelly ANN-to-SNN conversion."""
    try:
        from spikingjelly.activation_based import ann2snn
    except ImportError as exc:
        raise ImportError("SpikingJelly is required: pip install spikingjelly") from exc

    try:
        converter = ann2snn.Converter(mode="max", dataloader=calibration_loader)
        return converter(ann).to(DEVICE)
    except TypeError:
        converter = ann2snn.Converter(dataloader=calibration_loader, mode="max")
        return converter(ann).to(DEVICE)
    except Exception as exc:
        raise RuntimeError(f"SpikingJelly conversion failed: {exc}") from exc


def main():
    """CLI entry point for ANN-to-SNN conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    ensure_dirs()

    train_ds = CASEVideoDataset(CASE_ROOT, CLIP_DURATION, FPS, IMG_SIZE, N_MFCC, SR, "train", VAL_SPLIT)
    val_ds = CASEVideoDataset(CASE_ROOT, CLIP_DURATION, FPS, IMG_SIZE, N_MFCC, SR, "val", VAL_SPLIT)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("No clips available for conversion calibration/validation.")

    n_cal = max(1, int(round(len(train_ds) * 0.1)))
    calibration_loader = DataLoader(Subset(train_ds, range(n_cal)), batch_size=args.batch_size, shuffle=False, collate_fn=calibration_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    ann = load_ann()
    snn = convert_ann_to_snn(ann, calibration_loader)
    try:
        torch.save(snn, SNN_CHECKPOINT)
    except Exception as exc:
        raise RuntimeError(f"Could not save SNN checkpoint {SNN_CHECKPOINT}: {exc}") from exc

    ann_preds, targets = model_outputs(ann, val_loader)
    snn_preds, _ = model_outputs(snn, val_loader)
    mse = float(np.mean((ann_preds - snn_preds) ** 2))
    r_v = pearsonr_safe(snn_preds[:, 0], targets[:, 0])
    r_a = pearsonr_safe(snn_preds[:, 1], targets[:, 1])
    print(f"ANN vs SNN output MSE: {mse:.5f} (target < 0.05)")
    print(f"SNN val Pearson r valence={r_v:.3f}, arousal={r_a:.3f}")
    plot_ann_snn_comparison(ann_preds, snn_preds, OUTPUT_DIR / "ann_vs_snn_comparison.png")


if __name__ == "__main__":
    main()
