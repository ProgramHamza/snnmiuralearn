from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
	from .video_processor import VideoProcessor
except ImportError:
	from video_processor import VideoProcessor


@dataclass(frozen=True)
class FrameSample:
	video_path: str
	frame_idx: int
	time_sec: float
	valence: float
	arousal: float


class VideoFrameDataset(Dataset):
	def __init__(
		self,
		data_root: str | Path | None = None,
		split: str = "train",
		frame_step: int = 3,
		image_size: tuple[int, int] = (360, 640),
		video_limit: Optional[int] = None,
		sample_limit: Optional[int] = None,
		transform=None,
	):
		self.processor = VideoProcessor(data_root)
		self.split = split
		self.frame_step = frame_step
		self.image_size = image_size
		self.video_limit = video_limit
		self.sample_limit = sample_limit
		self.transform = transform or transforms.Compose(
			[
				transforms.Resize(image_size),
				transforms.Grayscale(num_output_channels=1),
				transforms.ToTensor(),
				transforms.Normalize((0.5,), (0.5,)),
			]
		)

		self.video_paths = self._discover_video_paths()
		self.samples = self._build_samples()

	def _discover_video_paths(self) -> list[str]:
		if self.split in {"train", "trn"}:
			video_iter = self.processor.iterate_train_videos()
		elif self.split in {"val", "valid", "test"}:
			video_iter = self.processor.iterate_test_videos()
		else:
			raise ValueError("split must be one of: train, val, test")

		video_paths = list(video_iter)
		if self.video_limit is not None:
			video_paths = video_paths[: self.video_limit]
		return video_paths

	def _build_samples(self) -> list[FrameSample]:
		samples: list[FrameSample] = []
		for video_path in self.video_paths:
			label_frame = self.processor.load_label_table(video_path)
			if label_frame is None or label_frame.empty:
				continue

			frame_rows = label_frame[label_frame["frame_idx"] % self.frame_step == 0]
			for row in frame_rows.itertuples(index=False):
				samples.append(
					FrameSample(
						video_path=video_path,
						frame_idx=int(row.frame_idx),
						time_sec=float(row.time_sec),
						valence=float(row.valence),
						arousal=float(row.arousal),
					)
				)
				if self.sample_limit is not None and len(samples) >= self.sample_limit:
					return samples

		return samples

	def __len__(self) -> int:
		return len(self.samples)

	def _read_frame(self, video_path: str, frame_idx: int):
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise FileNotFoundError(f"Could not open video: {video_path}")

		try:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
			ret, frame = cap.read()
			if not ret:
				raise IndexError(f"Could not read frame {frame_idx} from {video_path}")
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			return frame
		finally:
			cap.release()

	def __getitem__(self, index: int):
		sample = self.samples[index]
		frame = self._read_frame(sample.video_path, sample.frame_idx)
		pil_frame = transforms.ToPILImage()(frame)
		image_tensor = self.transform(pil_frame)
		target_tensor = torch.tensor([sample.valence, sample.arousal], dtype=torch.float32)
		return image_tensor, target_tensor

	def sample_metadata(self, index: int) -> FrameSample:
		return self.samples[index]

	def iter_video_samples(self, video_path: str) -> Iterable[FrameSample]:
		for sample in self.samples:
			if sample.video_path == video_path:
				yield sample


class VideoInferenceDataset(Dataset):
	def __init__(
		self,
		data_root: str | Path | None = None,
		split: str = "test",
		frame_step: int = 3,
		image_size: tuple[int, int] = (360, 640),
		video_paths: Optional[list[str]] = None,
		transform=None,
	):
		self.processor = VideoProcessor(data_root)
		self.split = split
		self.frame_step = frame_step
		self.image_size = image_size
		self.transform = transform or transforms.Compose(
			[
				transforms.Resize(image_size),
				transforms.Grayscale(num_output_channels=1),
				transforms.ToTensor(),
				transforms.Normalize((0.5,), (0.5,)),
			]
		)

		if video_paths is None:
			if split in {"train", "trn"}:
				video_paths = list(self.processor.iterate_train_videos())
			else:
				video_paths = list(self.processor.iterate_test_videos())
		self.video_paths = video_paths
		self.samples = self._build_samples()

	def _build_samples(self) -> list[tuple[str, int, float]]:
		samples: list[tuple[str, int, float]] = []
		for video_path in self.video_paths:
			cap = cv2.VideoCapture(video_path)
			if not cap.isOpened():
				continue

			try:
				n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
				fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
				for frame_idx in range(0, n_frames, self.frame_step):
					time_sec = frame_idx / fps if fps > 0 else frame_idx / 30.0
					samples.append((video_path, frame_idx, time_sec))
			finally:
				cap.release()
		return samples

	def __len__(self) -> int:
		return len(self.samples)

	def _read_frame(self, video_path: str, frame_idx: int):
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise FileNotFoundError(f"Could not open video: {video_path}")

		try:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
			ret, frame = cap.read()
			if not ret:
				raise IndexError(f"Could not read frame {frame_idx} from {video_path}")
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			return frame
		finally:
			cap.release()

	def __getitem__(self, index: int):
		video_path, frame_idx, time_sec = self.samples[index]
		frame = self._read_frame(video_path, frame_idx)
		pil_frame = transforms.ToPILImage()(frame)
		image_tensor = self.transform(pil_frame)
		metadata = {
			"video_path": video_path,
			"frame_idx": frame_idx,
			"time_sec": time_sec,
		}
		return image_tensor, metadata

	def iter_video_samples(self, video_path: str):
		for sample_video_path, frame_idx, time_sec in self.samples:
			if sample_video_path == video_path:
				yield sample_video_path, frame_idx, time_sec


def build_dataloaders(
	data_root: str | Path | None = None,
	batch_size: int = 8,
	frame_step: int = 3,
	image_size: tuple[int, int] = (360, 640),
	num_workers: int = 0,
	train_video_limit: Optional[int] = None,
	val_video_limit: Optional[int] = None,
	train_sample_limit: Optional[int] = None,
	val_sample_limit: Optional[int] = None,
):
	train_dataset = VideoFrameDataset(
		data_root=data_root,
		split="train",
		frame_step=frame_step,
		image_size=image_size,
		video_limit=train_video_limit,
		sample_limit=train_sample_limit,
	)
	val_dataset = VideoFrameDataset(
		data_root=data_root,
		split="val",
		frame_step=frame_step,
		image_size=image_size,
		video_limit=val_video_limit,
		sample_limit=val_sample_limit,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=num_workers,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		drop_last=False,
		num_workers=num_workers,
	)
	return train_dataset, val_dataset, train_loader, val_loader


def build_inference_dataset(
	data_root: str | Path | None = None,
	split: str = "test",
	frame_step: int = 3,
	image_size: tuple[int, int] = (360, 640),
	video_paths: Optional[list[str]] = None,
):
	return VideoInferenceDataset(
		data_root=data_root,
		split=split,
		frame_step=frame_step,
		image_size=image_size,
		video_paths=video_paths,
	)
 