"""Validate ANN or SNN checkpoints on CASE or a supplied video."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CASE_ROOT, CLIP_DURATION, DEVICE, FPS, IMG_SIZE, N_MFCC, OUTPUT_DIR, SR, VAL_SPLIT, ensure_dirs
from dataset import CASEVideoDataset, collate_fn
from infer import load_model
from preprocess import load_frame_diff, load_mfcc
from train import pearsonr_safe
from visualize import emotion_quadrant


QUADRANTS = ["EXCITED", "FEARFUL", "CALM", "BORED/SAD"]


def quadrant_index(valence, arousal):
    """Map valence and arousal to a quadrant index."""
    return QUADRANTS.index(emotion_quadrant(float(valence), float(arousal)))


def confusion_matrix(preds, targets):
    """Build a 4x4 quadrant confusion matrix."""
    matrix = np.zeros((4, 4), dtype=np.int64)
    for pred, target in zip(preds, targets):
        matrix[quadrant_index(target[0], target[1]), quadrant_index(pred[0], pred[1])] += 1
    return matrix


def validate_case(mode):
    """Run validation on the CASE validation split and print metrics."""
    model = load_model(mode)
    ds = CASEVideoDataset(CASE_ROOT, CLIP_DURATION, FPS, IMG_SIZE, N_MFCC, SR, "val", VAL_SPLIT)
    if len(ds) == 0:
        raise RuntimeError("No CASE validation clips available. Add matching stimulus videos under CASE_ROOT.")
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch["frames"].to(DEVICE), batch["mfcc"].to(DEVICE)).detach().cpu().numpy()
            tgt = torch.stack([batch["valence"], batch["arousal"]], dim=1).numpy()
            preds.append(out)
            targets.append(tgt)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mse = np.mean((preds - targets) ** 2, axis=0)
    mae = np.mean(np.abs(preds - targets), axis=0)
    print(f"MSE valence={mse[0]:.5f}, arousal={mse[1]:.5f}")
    print(f"MAE valence={mae[0]:.5f}, arousal={mae[1]:.5f}")
    print(f"Pearson r valence={pearsonr_safe(preds[:, 0], targets[:, 0]):.3f}, arousal={pearsonr_safe(preds[:, 1], targets[:, 1]):.3f}")
    print("Quadrant confusion matrix rows=true cols=pred; order EXCITED, FEARFUL, CALM, BORED/SAD")
    print(confusion_matrix(preds, targets))


def video_duration_seconds(video_path):
    """Return video duration in seconds using OpenCV."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required: pip install opencv-python") from exc
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(frames / fps) if fps else 0.0


def validate_video(mode, video_path):
    """Run model over a full video in fixed-size clips and plot a timeline."""
    model = load_model(mode)
    video_path = Path(video_path)
    duration = video_duration_seconds(video_path)
    starts = np.arange(0.0, max(0.0, duration - CLIP_DURATION + 1e-6), CLIP_DURATION)
    if starts.size == 0:
        starts = np.array([0.0])
    preds = []
    model.eval()
    with torch.no_grad():
        for start in starts:
            frames = load_frame_diff(video_path, float(start), CLIP_DURATION, FPS, IMG_SIZE)
            mfcc = load_mfcc(video_path, float(start), CLIP_DURATION, SR, N_MFCC, n_frames=frames.shape[0])
            frames_t = torch.as_tensor(frames, dtype=torch.float32).mean(dim=0, keepdim=True).to(DEVICE)
            mfcc_t = torch.as_tensor(mfcc, dtype=torch.float32).mean(dim=0, keepdim=True).to(DEVICE)
            preds.append(model(frames_t, mfcc_t).detach().cpu().numpy()[0])
    preds = np.asarray(preds)
    for start, pred in zip(starts, preds):
        print(f"{start:8.2f}s valence={pred[0]: .3f} arousal={pred[1]: .3f}")
    save_timeline(starts, preds, OUTPUT_DIR / "validation_timeline.png")


def save_timeline(starts, preds, save_path):
    """Save valence and arousal timelines."""
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(starts, preds[:, 0], label="valence")
        ax.plot(starts, preds[:, 1], label="arousal")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Prediction")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        raise RuntimeError(f"Could not save validation timeline to {save_path}: {exc}") from exc


def main():
    """CLI entry point for validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ann", "snn"], default="ann")
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()
    ensure_dirs()
    if args.video:
        validate_video(args.mode, args.video)
    else:
        validate_case(args.mode)


if __name__ == "__main__":
    main()

