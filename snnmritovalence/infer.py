"""Run inference on a single video clip."""

import argparse
from pathlib import Path

import torch

from config import ANN_CHECKPOINT, CLIP_DURATION, DEVICE, FPS, IMG_SIZE, N_MFCC, OUTPUT_DIR, SNN_CHECKPOINT, SR, ensure_dirs
from model import EmotionANN
from preprocess import load_frame_diff, load_mfcc
from visualize import emotion_quadrant, plot_va_point


def load_model(mode):
    """Load the ANN or converted SNN checkpoint."""
    ckpt_path = ANN_CHECKPOINT if mode == "ann" else SNN_CHECKPOINT
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        try:
            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        if mode == "ann" or isinstance(checkpoint, dict):
            model = EmotionANN(N_MFCC).to(DEVICE)
            state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state)
        else:
            model = checkpoint.to(DEVICE)
        model.eval()
        return model
    except Exception as exc:
        raise RuntimeError(f"Could not load {mode} model from {ckpt_path}: {exc}") from exc


def predict_video(video, mode="ann", start=0.0, duration=5.0):
    """Predict valence and arousal for one video clip."""
    model = load_model(mode)
    frames = load_frame_diff(video, start, duration, FPS, IMG_SIZE)
    mfcc = load_mfcc(video, start, duration, SR, N_MFCC, n_frames=frames.shape[0])
    frames_t = torch.as_tensor(frames, dtype=torch.float32).mean(dim=0, keepdim=True).to(DEVICE)
    mfcc_t = torch.as_tensor(mfcc, dtype=torch.float32).mean(dim=0, keepdim=True).to(DEVICE)
    with torch.no_grad():
        out = model(frames_t, mfcc_t).detach().cpu().squeeze(0)
    return float(out[0]), float(out[1])


def main():
    """CLI entry point for single-video inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--mode", choices=["ann", "snn"], default="ann")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=CLIP_DURATION)
    args = parser.parse_args()
    ensure_dirs()
    valence, arousal = predict_video(args.video, args.mode, args.start, args.duration)
    state = emotion_quadrant(valence, arousal).replace("/SAD", "")
    print(f"Valence: {valence:.2f}  (range -1 to 1, positive = pleasant)")
    print(f"Arousal: {arousal:.2f}  (range -1 to 1, positive = excited)")
    print(f"Emotion quadrant: {state}")
    plot_va_point(valence, arousal, OUTPUT_DIR / "infer_result.png")


if __name__ == "__main__":
    main()
