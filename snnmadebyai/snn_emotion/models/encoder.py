from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Optional, Sequence

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import hog

from config import (
    CLIP_DURATION,
    FEATURE_LENGTH,
    FRAME_SIZE,
    HOG_PIXELS_PER_CELL,
    N_HOG_ORIENTATIONS,
    N_MELS,
    SR,
    T_SPIKE,
    VIDEO_FPS,
)
from models.spike_encoder import RateSpikeEncoder


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _extract_audio(video_path: Path, start_time: float, duration: float) -> np.ndarray:
    try:
        audio, _ = librosa.load(
            str(video_path),
            sr=SR,
            mono=True,
            offset=max(0.0, start_time),
            duration=duration,
        )
    except Exception:
        audio = np.zeros(int(SR * duration), dtype=np.float32)

    if audio.size == 0:
        audio = np.zeros(int(SR * duration), dtype=np.float32)
    return audio.astype(np.float32, copy=False)


def extract_audio_features(
    video_path: str | Path,
    start_time: float = 0.0,
    duration: float = CLIP_DURATION,
    target_length: int = FEATURE_LENGTH,
) -> torch.Tensor:
    video_path = _resolve_path(video_path)
    audio = _extract_audio(video_path, start_time, duration)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=N_MELS,
        hop_length=512,
        n_fft=1024,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    tensor = torch.from_numpy(log_mel).float().unsqueeze(0)
    if tensor.shape[-1] != target_length:
        tensor = F.interpolate(tensor, size=target_length, mode="linear", align_corners=False)
    return tensor


def _sample_video_frames(
    video_path: Path,
    start_time: float,
    duration: float,
    target_fps: float = VIDEO_FPS,
) -> Sequence[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_time) * 1000.0)
    frames: list[np.ndarray] = []
    next_sample_time = start_time
    end_time = start_time + duration

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = target_fps

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        current_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        current_time = current_ms / 1000.0 if current_ms > 0 else next_sample_time
        if current_time < start_time:
            continue
        if current_time > end_time:
            break

        if current_time >= next_sample_time:
            frames.append(frame)
            next_sample_time += 1.0 / target_fps

        if current_time >= end_time:
            break

        if fps > target_fps:
            capture.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000.0 + 1000.0 / target_fps)

    capture.release()
    return frames


def _frame_to_hog(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    features = hog(
        gray,
        orientations=N_HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=(1, 1),
        feature_vector=True,
    )
    return features.astype(np.float32, copy=False)


def extract_video_features(
    video_path: str | Path,
    start_time: float = 0.0,
    duration: float = CLIP_DURATION,
    target_length: int = FEATURE_LENGTH,
) -> torch.Tensor:
    video_path = _resolve_path(video_path)
    frames = _sample_video_frames(video_path, start_time, duration)

    if not frames:
        dummy = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]), dtype=np.uint8)
        hog_dim = _frame_to_hog(cv2.cvtColor(dummy, cv2.COLOR_GRAY2BGR)).shape[0]
        features = np.zeros((hog_dim, 1), dtype=np.float32)
        tensor = torch.from_numpy(features).unsqueeze(0)
        return F.interpolate(tensor, size=target_length, mode="linear", align_corners=False)

    hog_features = [_frame_to_hog(frame) for frame in frames]
    stacked = np.stack(hog_features, axis=1)
    tensor = torch.from_numpy(stacked).float().unsqueeze(0)
    if tensor.shape[-1] != target_length:
        tensor = F.interpolate(tensor, size=target_length, mode="linear", align_corners=False)
    return tensor


def fuse_modalities(
    audio_features: torch.Tensor,
    video_features: torch.Tensor,
    target_length: int = FEATURE_LENGTH,
) -> torch.Tensor:
    if audio_features.dim() != 3 or video_features.dim() != 3:
        raise ValueError("Modal features must have shape [B, C, F].")

    if audio_features.shape[-1] != target_length:
        audio_features = F.interpolate(audio_features, size=target_length, mode="linear", align_corners=False)
    if video_features.shape[-1] != target_length:
        video_features = F.interpolate(video_features, size=target_length, mode="linear", align_corners=False)
    return torch.cat([audio_features, video_features], dim=1)


def preprocess_video(
    video_path: str | Path,
    start_time: float = 0.0,
    duration: float = CLIP_DURATION,
    target_length: int = FEATURE_LENGTH,
) -> torch.Tensor:
    audio_features = extract_audio_features(video_path, start_time=start_time, duration=duration, target_length=target_length)
    video_features = extract_video_features(video_path, start_time=start_time, duration=duration, target_length=target_length)
    return fuse_modalities(audio_features, video_features, target_length=target_length)


def preprocess_to_spikes(
    video_path: str | Path,
    start_time: float = 0.0,
    duration: float = CLIP_DURATION,
    target_length: int = FEATURE_LENGTH,
    timesteps: int = T_SPIKE,
) -> torch.Tensor:
    features = preprocess_video(video_path, start_time=start_time, duration=duration, target_length=target_length)
    encoder = RateSpikeEncoder(timesteps=timesteps)
    return encoder(features).squeeze(1)


def infer_participant_id(path: str | Path) -> str:
    candidate = Path(path)
    tokens = [candidate.stem, candidate.parent.name, candidate.parent.parent.name if candidate.parent.parent else ""]
    patterns = [
        r"(?:participant|subject|speaker|sub|p)[_\- ]*(\d+)",
        r"(\d{2,})",
    ]
    for token in tokens:
        normalized = token.lower()
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return match.group(1)
    return candidate.parent.name or candidate.stem


def read_annotation_clip(
    csv_path: str | Path,
    start_time: float,
    duration: float,
    annotations_per_second: float = 2.0,
) -> tuple[float, float]:
    csv_path = _resolve_path(csv_path)
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            rows = [{(key or "").lower(): value for key, value in row.items()} for row in reader]

    if not rows:
        numeric = np.loadtxt(str(csv_path), delimiter=",", ndmin=2)
        if numeric.shape[1] >= 3:
            valence_values = numeric[:, 1]
            arousal_values = numeric[:, 2]
        elif numeric.shape[1] == 2:
            valence_values = numeric[:, 0]
            arousal_values = numeric[:, 1]
        else:
            valence_values = numeric[:, 0]
            arousal_values = numeric[:, 0]
        return float(np.mean(valence_values)), float(np.mean(arousal_values))

    time_key = next((name for name in rows[0].keys() if "time" in name or "timestamp" in name), None)
    valence_key = next((name for name in rows[0].keys() if "val" in name), None)
    arousal_key = next((name for name in rows[0].keys() if "ar" in name), None)

    def _row_time(row: dict[str, str], index: int) -> float:
        if time_key is None:
            return index / annotations_per_second
        return _safe_float(row.get(time_key, index / annotations_per_second), index / annotations_per_second)

    selected_valence: list[float] = []
    selected_arousal: list[float] = []
    end_time = start_time + duration

    for index, row in enumerate(rows):
        current_time = _row_time(row, index)
        if current_time < start_time or current_time > end_time:
            continue
        selected_valence.append(_safe_float(row.get(valence_key or "valence", 0.0), 0.0))
        selected_arousal.append(_safe_float(row.get(arousal_key or "arousal", 0.0), 0.0))

    if not selected_valence:
        for row in rows:
            selected_valence.append(_safe_float(row.get(valence_key or "valence", 0.0), 0.0))
            selected_arousal.append(_safe_float(row.get(arousal_key or "arousal", 0.0), 0.0))

    return float(np.mean(selected_valence)), float(np.mean(selected_arousal))


def valence_arousal_to_emotions(valence: float, arousal: float) -> torch.Tensor:
    v = float(valence)
    a = float(arousal)
    raw_scores = np.array(
        [
            100.0 * (1.0 / (1.0 + math.exp(-3.0 * v))) * (1.0 / (1.0 + math.exp(-3.0 * a))),
            100.0 * (1.0 / (1.0 + math.exp(2.0 * v))) * (1.0 / (1.0 + math.exp(3.0 * a))),
            100.0 * (1.0 / (1.0 + math.exp(3.0 * v))) * (1.0 / (1.0 + math.exp(2.0 * a))),
            100.0 * (1.0 / (1.0 + math.exp(2.0 * v))) * (1.0 / (1.0 + math.exp(-3.0 * a))),
            100.0 * (1.0 / (1.0 + math.exp(2.0 * v))) * (1.0 / (1.0 + math.exp(-2.0 * a))),
            100.0 * (1.0 / (1.0 + math.exp(-3.0 * a))) * max(0.0, 1.0 - abs(v)),
            100.0 * (1.0 / (1.0 + math.exp(3.0 * abs(v)))) * (1.0 / (1.0 + math.exp(3.0 * abs(a)))),
        ],
        dtype=np.float32,
    )

    raw_scores = np.clip(raw_scores, 0.0, 100.0)
    total = float(raw_scores.sum())
    if total > 0.0:
        raw_scores = raw_scores * (100.0 / total)
    return torch.from_numpy(raw_scores)
