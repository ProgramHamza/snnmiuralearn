"""CASE dataset loader for video/audio valence-arousal training."""

from pathlib import Path
import random
import re
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CASE_ROOT, CLIP_DURATION, FPS, IMG_SIZE, N_MFCC, SR, VAL_SPLIT
from preprocess import load_frame_diff, load_mfcc


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _scale_label(values):
    """Scale CASE labels from [0.5, 9.5] or [1, 9] to [-1, 1]."""
    arr = np.asarray(values, dtype=np.float32)
    if np.nanmin(arr) >= 0.0 and np.nanmax(arr) > 1.0:
        return np.clip((arr - 5.0) / 4.5, -1.0, 1.0)
    return np.clip(arr, -1.0, 1.0)


def _find_column(columns, candidates):
    """Find the first column whose normalized name matches a candidate."""
    norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in norm:
            return norm[key]
    return None


def _video_id_from_path(path):
    """Infer a CASE video ID from a stimulus path."""
    return Path(path).stem.lower().replace("_", "-")


def scan_video_files(case_root):
    """Recursively scan for media files and map inferred stimulus IDs to paths."""
    case_root = Path(case_root)
    videos = {}
    try:
        for path in case_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.setdefault(_video_id_from_path(path), path)
    except OSError as exc:
        warnings.warn(f"Could not scan videos under {case_root}: {exc}")
    return videos


def _annotation_dir(case_root):
    """Choose the best available CASE annotations directory."""
    root = Path(case_root)
    candidates = [
        root / "CASE_full" / "data" / "interpolated" / "annotations",
        root / "data" / "interpolated" / "annotations",
        root / "CASE_full" / "data" / "non-interpolated" / "annotations",
        root / "data" / "non-interpolated" / "annotations",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return root


class CASEVideoDataset(Dataset):
    """Dataset that pairs CASE continuous annotations with matching videos."""

    def __init__(
        self,
        case_root,
        clip_duration=5,
        fps=2,
        img_size=128,
        n_mfcc=40,
        sr=16000,
        split="train",
        val_split=0.2,
        seed=42,
    ):
        """Scan CASE annotations and create clip-level samples split by video ID."""
        self.case_root = Path(case_root)
        self.clip_duration = clip_duration
        self.fps = fps
        self.img_size = img_size
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.split = split
        self.video_files = scan_video_files(self.case_root)
        self.samples = []
        self._build_samples(val_split=val_split, seed=seed)

    def _build_samples(self, val_split, seed):
        """Populate samples as (video_path, start_sec, mean_valence, mean_arousal)."""
        ann_dir = _annotation_dir(self.case_root)
        try:
            csv_files = sorted(p for p in ann_dir.rglob("*.csv") if p.name.lower().startswith("sub"))
        except OSError as exc:
            raise RuntimeError(f"Could not scan annotation CSV files in {ann_dir}: {exc}") from exc

        if not csv_files:
            warnings.warn(f"No annotation CSV files found under {ann_dir}")
            return

        all_video_ids = set()
        parsed = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                warnings.warn(f"Skipping unreadable annotation CSV {csv_path}: {exc}")
                continue
            val_col = _find_column(df.columns, ["valence", "v", "val"])
            aro_col = _find_column(df.columns, ["arousal", "a", "aro"])
            vid_col = _find_column(df.columns, ["video", "video_id", "stimulus", "stimulus_id"])
            time_col = _find_column(df.columns, ["jstime", "time", "timestamp", "t"])
            if not all([val_col, aro_col, vid_col, time_col]):
                warnings.warn(f"Skipping {csv_path}; missing valence/arousal/video/time columns")
                continue
            df = df[[time_col, val_col, aro_col, vid_col]].dropna()
            df.columns = ["time_ms", "valence", "arousal", "video"]
            df["video"] = df["video"].astype(str)
            for video_id in df["video"].unique():
                all_video_ids.add(video_id.lower())
            parsed.append((csv_path, df))

        train_ids, val_ids = self._split_video_ids(sorted(all_video_ids), val_split, seed)
        selected_ids = train_ids if self.split == "train" else val_ids

        skipped_missing = 0
        for _, df in parsed:
            for video_id, group in df.groupby("video"):
                key = video_id.lower()
                if key not in selected_ids:
                    continue
                video_path = self.video_files.get(key)
                if video_path is None:
                    skipped_missing += 1
                    continue
                group = group.sort_values("time_ms").copy()
                group["rel_sec"] = (group["time_ms"] - group["time_ms"].iloc[0]) / 1000.0
                max_sec = float(group["rel_sec"].max())
                start = 0.0
                while start + self.clip_duration <= max_sec:
                    clip = group[(group["rel_sec"] >= start) & (group["rel_sec"] < start + self.clip_duration)]
                    if not clip.empty:
                        val = float(np.mean(_scale_label(clip["valence"].to_numpy())))
                        aro = float(np.mean(_scale_label(clip["arousal"].to_numpy())))
                        self.samples.append((video_path, start, val, aro))
                    start += self.clip_duration
        if skipped_missing:
            warnings.warn(
                f"Skipped {skipped_missing} annotation video groups because matching stimulus files were not found. "
                f"Detected {len(self.video_files)} media files under {self.case_root}."
            )

    @staticmethod
    def _split_video_ids(video_ids, val_split, seed):
        """Split unique video IDs so clips from the same stimulus do not leak."""
        rng = random.Random(seed)
        ids = list(video_ids)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_split))) if ids else 0
        val_ids = set(ids[:n_val])
        train_ids = set(ids[n_val:])
        return train_ids, val_ids

    def __len__(self):
        """Return the number of available clips."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load one clip and return frames, MFCC, valence, and arousal tensors."""
        video_path, start_sec, valence, arousal = self.samples[idx]
        frames = load_frame_diff(video_path, start_sec, self.clip_duration, self.fps, self.img_size)
        mfcc = load_mfcc(video_path, start_sec, self.clip_duration, self.sr, self.n_mfcc, n_frames=frames.shape[0])
        return {
            "frames": torch.as_tensor(frames, dtype=torch.float32),
            "mfcc": torch.as_tensor(mfcc, dtype=torch.float32),
            "valence": torch.tensor(valence, dtype=torch.float32),
            "arousal": torch.tensor(arousal, dtype=torch.float32),
        }


def collate_fn(batch):
    """Collate variable-length clips by averaging frames and MFCC over time."""
    frames = torch.stack([item["frames"].mean(dim=0) for item in batch])
    mfcc = torch.stack([item["mfcc"].mean(dim=0) for item in batch])
    valence = torch.stack([item["valence"] for item in batch])
    arousal = torch.stack([item["arousal"] for item in batch])
    return {"frames": frames, "mfcc": mfcc, "valence": valence, "arousal": arousal}


if __name__ == "__main__":
    ds = CASEVideoDataset(CASE_ROOT, CLIP_DURATION, FPS, IMG_SIZE, N_MFCC, SR, "train", VAL_SPLIT)
    print(f"Dataset size: {len(ds)}")
    print(f"Detected media files: {len(ds.video_files)}")
    if len(ds):
        sample = ds[0]
        print({k: tuple(v.shape) if hasattr(v, "shape") else v for k, v in sample.items()})
    else:
        print("No samples available. Add CASE stimulus videos under the dataset root to train.")

