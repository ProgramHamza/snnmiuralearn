"""Train the ANN emotion regressor on CASE clips."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import (
    ANN_CHECKPOINT,
    BATCH_SIZE,
    CASE_ROOT,
    CLIP_DURATION,
    DEVICE,
    EPOCHS,
    FPS,
    IMG_SIZE,
    LR,
    N_MFCC,
    OUTPUT_DIR,
    SR,
    VAL_SPLIT,
    ensure_dirs,
)
from dataset import CASEVideoDataset, collate_fn
from model import EmotionANN
from visualize import plot_training_curves


def pearsonr_safe(x, y):
    """Compute Pearson r, returning 0.0 for degenerate inputs."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.size < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_epoch(model, loader, optimizer=None):
    """Run one train or evaluation epoch and return loss plus metrics."""
    training = optimizer is not None
    model.train(training)
    mse = nn.MSELoss()
    losses, preds, targets = [], [], []
    for batch in loader:
        frames = batch["frames"].to(DEVICE)
        mfcc = batch["mfcc"].to(DEVICE)
        valence = batch["valence"].to(DEVICE)
        arousal = batch["arousal"].to(DEVICE)
        with torch.set_grad_enabled(training):
            out = model(frames, mfcc)
            loss = mse(out[:, 0], valence) + mse(out[:, 1], arousal)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        preds.append(out.detach().cpu().numpy())
        targets.append(torch.stack([valence, arousal], dim=1).detach().cpu().numpy())
    if not losses:
        return 0.0, 0.0, 0.0
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return float(np.mean(losses)), pearsonr_safe(preds[:, 0], targets[:, 0]), pearsonr_safe(preds[:, 1], targets[:, 1])


def main():
    """CLI entry point for ANN training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()
    ensure_dirs()

    train_ds = CASEVideoDataset(CASE_ROOT, CLIP_DURATION, FPS, IMG_SIZE, N_MFCC, SR, "train", VAL_SPLIT)
    val_ds = CASEVideoDataset(CASE_ROOT, CLIP_DURATION, FPS, IMG_SIZE, N_MFCC, SR, "val", VAL_SPLIT)
    print(f"Train clips: {len(train_ds)} | Val clips: {len(val_ds)}")
    if len(train_ds):
        sample = train_ds[0]
        print(f"Sample frames {tuple(sample['frames'].shape)}, mfcc {tuple(sample['mfcc'].shape)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("No train/val clips available. CASE annotations are present, but matching stimulus videos are missing.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = EmotionANN(N_MFCC).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss, best_rv, best_ra, patience, stale = float("inf"), 0.0, 0.0, 7, 0
    train_losses, val_losses, val_rv, val_ra = [], [], [], []
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, r_v, r_a = run_epoch(model, val_loader)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_rv.append(r_v)
        val_ra.append(r_a)
        print(f"epoch {epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_r_v={r_v:.3f} val_r_a={r_a:.3f}")
        if val_loss < best_loss:
            best_loss, best_rv, best_ra, stale = val_loss, r_v, r_a, 0
            try:
                torch.save({"model_state": model.state_dict(), "config": {"n_mfcc": N_MFCC}}, ANN_CHECKPOINT)
            except Exception as exc:
                raise RuntimeError(f"Could not save checkpoint {ANN_CHECKPOINT}: {exc}") from exc
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break
    plot_training_curves(train_losses, val_losses, OUTPUT_DIR / "training_curves.png", val_rv, val_ra)
    print(f"Best val loss: {best_loss:.5f}; best Pearson r valence={best_rv:.3f}, arousal={best_ra:.3f}")


if __name__ == "__main__":
    main()

