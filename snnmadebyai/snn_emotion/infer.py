from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from spikingjelly.activation_based import functional

from config import BEST_CHECKPOINT_PATH, CLIP_DURATION, EMOTION_NAMES, FEATURE_LENGTH, T_SPIKE
from models.encoder import preprocess_to_spikes
from models.snn_classifier import ConvSNN
from visualize import render_emotion_result


def main():
    parser = argparse.ArgumentParser(description="Run CASE emotion inference on a single video file.")
    parser.add_argument("--video", required=True, help="Path to a video file.")
    parser.add_argument("--checkpoint", default=str(BEST_CHECKPOINT_PATH), help="Path to the trained checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    video_path = Path(args.video).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    spikes = preprocess_to_spikes(video_path, duration=CLIP_DURATION, target_length=FEATURE_LENGTH, timesteps=T_SPIKE)
    if spikes.dim() == 4:
        spikes = spikes.squeeze(1)
    input_channels = spikes.shape[1]
    feature_length = spikes.shape[2]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_channels = int(checkpoint.get("input_channels", input_channels))
    feature_length = int(checkpoint.get("feature_length", feature_length))
    model = ConvSNN(input_channels=input_channels, feature_length=feature_length).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    spikes = spikes.unsqueeze(1).to(device)
    functional.reset_net(model)
    with torch.no_grad():
        affect_pred, emotion_pred = model(spikes)

    affect_pred = affect_pred.squeeze(0).detach().cpu().tolist()
    emotion_pred = emotion_pred.squeeze(0).detach().cpu().tolist()

    result = {
        "valence": float(affect_pred[0]),
        "arousal": float(affect_pred[1]),
    }
    for name, value in zip(EMOTION_NAMES, emotion_pred):
        result[name] = float(value)

    print(json.dumps(result, indent=2))
    render_emotion_result(result)


if __name__ == "__main__":
    main()
