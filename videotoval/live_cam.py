from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from .model import load_model
    from .video_processor import VideoProcessor
except ImportError:
    from model import load_model
    from video_processor import VideoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained video-to-VA model on a live camera feed and save the result.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved model checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("live_runs"), help="Directory for recorded footage and predictions")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for cv2.VideoCapture")
    parser.add_argument("--duration-sec", type=float, default=10.0, help="Capture duration in seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum number of frames to capture")
    parser.add_argument("--frame-step", type=int, default=3, help="Process every Nth captured frame")
    parser.add_argument("--num-steps", type=int, default=None, help="Optional override for the model's internal SNN steps")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=360, help="Capture height")
    parser.add_argument("--fps", type=float, default=30.0, help="Capture FPS for the saved video")
    return parser.parse_args()


def overlay_prediction(frame, prediction, spike_rate, frame_idx):
    valence, arousal = prediction
    text_lines = [
        f"frame: {frame_idx}",
        f"valence: {valence:.3f}",
        f"arousal: {arousal:.3f}",
        f"spike_rate: {spike_rate:.3f}",
    ]
    y = 28
    for line in text_lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += 30
    return frame


def run_live_camera():
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model, checkpoint = load_model(args.checkpoint, map_location=device)
    if args.num_steps is not None:
        model.num_steps = args.num_steps
    model.eval()

    processor = VideoProcessor()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_video_path = output_dir / "live_capture_raw.mp4"
    annotated_video_path = output_dir / "live_capture_annotated.mp4"
    predictions_path = output_dir / "live_predictions.csv"

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    writer_raw = cv2.VideoWriter(
        str(raw_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    writer_annotated = cv2.VideoWriter(
        str(annotated_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )

    predictions_rows = []
    state = None
    frame_idx = 0
    captured_frames = 0
    start_time = time()

    try:
        while True:
            if args.max_frames is not None and captured_frames >= args.max_frames:
                break
            if args.duration_sec is not None and (time() - start_time) >= args.duration_sec:
                break

            ret, frame = capture.read()
            if not ret:
                break

            writer_raw.write(frame)
            captured_frames += 1

            if frame_idx % args.frame_step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                image_tensor = processor.transform(pil_frame).unsqueeze(0).to(device)
                spikes = processor.spike_tensor(image_tensor.squeeze(0).cpu(), num_steps=model.num_steps)
                spike_rate = float(spikes.float().mean().item())

                with torch.no_grad():
                    prediction, state = model(image_tensor, state=state)

                prediction_np = prediction.squeeze(0).detach().cpu().numpy()
                predictions_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "time_sec": frame_idx / args.fps,
                        "valence": float(prediction_np[0]),
                        "arousal": float(prediction_np[1]),
                        "spike_rate": spike_rate,
                    }
                )
                frame = overlay_prediction(frame, prediction_np, spike_rate, frame_idx)

            writer_annotated.write(frame)
            frame_idx += 1
    finally:
        capture.release()
        writer_raw.release()
        writer_annotated.release()

    if predictions_rows:
        import pandas as pd

        pd.DataFrame(predictions_rows).to_csv(predictions_path, index=False)

    print(f"Saved raw capture to {raw_video_path.resolve()}")
    print(f"Saved annotated capture to {annotated_video_path.resolve()}")
    print(f"Saved predictions to {predictions_path.resolve()}")
    print(f"Loaded checkpoint keys: {sorted(checkpoint.keys())}")


if __name__ == "__main__":
    run_live_camera()
