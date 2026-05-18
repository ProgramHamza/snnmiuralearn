from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import CLIP_DURATION, FEATURE_LENGTH, RANDOM_SEED, TRAIN_SPLIT, T_SPIKE
from models.encoder import (
    extract_audio_features,
    extract_video_features,
    fuse_modalities,
    infer_participant_id,
    read_annotation_clip,
    valence_arousal_to_emotions,
)
from models.spike_encoder import RateSpikeEncoder


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}


@dataclass(frozen=True)
class CASERecord:
    video_path: Path
    annotation_path: Path
    participant_id: str


def _normalize_participant_id(participant_id: str) -> str:
    return re.sub(r"\s+", "", participant_id.strip().lower())


def _candidate_video_paths(annotation_path: Path) -> list[Path]:
    stem = annotation_path.stem
    clean_stem = re.sub(r"(_annotations?|_annotation|_ann)$", "", stem, flags=re.IGNORECASE)
    candidates = []
    for extension in VIDEO_EXTENSIONS:
        candidates.append(annotation_path.with_name(clean_stem + extension))
        candidates.append(annotation_path.with_name(stem + extension))
    return candidates


def _find_video_for_annotation(annotation_path: Path) -> Optional[Path]:
    for candidate in _candidate_video_paths(annotation_path):
        if candidate.exists():
            return candidate
    for extension in VIDEO_EXTENSIONS:
        matches = list(annotation_path.parent.rglob(annotation_path.stem + extension))
        if matches:
            return matches[0]
    return None


def _scan_case_records(data_root: Path) -> list[CASERecord]:
    records: list[CASERecord] = []
    for annotation_path in data_root.rglob("*.csv"):
        video_path = _find_video_for_annotation(annotation_path)
        if video_path is None:
            continue
        records.append(
            CASERecord(
                video_path=video_path,
                annotation_path=annotation_path,
                participant_id=infer_participant_id(annotation_path),
            )
        )
    return records


def _split_participants(records: list[CASERecord], split: str) -> list[CASERecord]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'.")

    participants = sorted({_normalize_participant_id(record.participant_id) for record in records})
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(participants)
    cutoff = max(1, int(len(participants) * TRAIN_SPLIT)) if participants else 0
    if len(participants) > 1:
        cutoff = min(cutoff, len(participants) - 1)
    train_participants = set(participants[:cutoff])
    val_participants = set(participants[cutoff:])

    selected: list[CASERecord] = []
    for record in records:
        participant = _normalize_participant_id(record.participant_id)
        is_train = participant in train_participants
        if split == "train" and is_train:
            selected.append(record)
        elif split in {"val", "test"} and participant in val_participants:
            selected.append(record)
    return selected


def _select_clip_start(video_path: Path, clip_duration: float, split: str, index: int) -> float:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()

    if not fps or fps <= 0 or math.isnan(fps):
        fps = 30.0
    duration = frame_count / fps if frame_count and frame_count > 0 else clip_duration
    if duration <= clip_duration:
        return 0.0

    if split == "train":
        rng = random.Random(RANDOM_SEED + index)
        return rng.uniform(0.0, duration - clip_duration)
    return max(0.0, (duration - clip_duration) / 2.0)


class CASEDataset(Dataset):
    def __init__(self, data_root, split: str = "train", clip_duration: int = CLIP_DURATION):
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.clip_duration = clip_duration
        self.records = _split_participants(_scan_case_records(self.data_root), split)
        self.target_length = FEATURE_LENGTH
        self.timesteps = T_SPIKE

        if not self.records:
            raise FileNotFoundError(
                f"No CASE video/annotation pairs were found in {self.data_root}."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        start_time = _select_clip_start(record.video_path, self.clip_duration, self.split, index)

        valence, arousal = read_annotation_clip(
            record.annotation_path,
            start_time=start_time,
            duration=self.clip_duration,
        )

        features = fuse_modalities(
            extract_audio_features(record.video_path, start_time=start_time, duration=self.clip_duration, target_length=self.target_length),
            extract_video_features(record.video_path, start_time=start_time, duration=self.clip_duration, target_length=self.target_length),
            target_length=self.target_length,
        )
        spike_tensor = RateSpikeEncoder(timesteps=self.timesteps)(features).squeeze(1)

        return (
            spike_tensor.float(),
            torch.tensor(valence, dtype=torch.float32),
            torch.tensor(arousal, dtype=torch.float32),
        )


def build_emotion_target(valence: torch.Tensor, arousal: torch.Tensor) -> torch.Tensor:
    if valence.dim() == 0:
        valence_value = float(valence.item())
    else:
        valence_value = float(valence.mean().item())
    if arousal.dim() == 0:
        arousal_value = float(arousal.item())
    else:
        arousal_value = float(arousal.mean().item())
    return valence_arousal_to_emotions(valence_value, arousal_value)
