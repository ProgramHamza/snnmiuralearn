from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from videotoval.model import Net
from videotoval.plot_graphs import (
	plot_beta_values,
	plot_gradient_norms,
	plot_learning_rate,
	plot_loss_curves,
	plot_membrane_trace,
	plot_scatter,
	plot_spike_rates,
	plot_timeseries,
	plot_total_loss,
)
from videotoval.training_data_load import VideoFrameDataset
from videotoval.video_processor import VideoProcessor


def _write_sample_video(video_path: Path, frame_count: int = 6, size: tuple[int, int] = (32, 32)) -> None:
	video_path.parent.mkdir(parents=True, exist_ok=True)
	writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
	try:
		for frame_idx in range(frame_count):
			frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
			frame[:, :, 0] = frame_idx * 20
			frame[:, :, 1] = 255 - frame_idx * 20
			frame[:, :, 2] = 50 + frame_idx * 10
			writer.write(frame)
	finally:
		writer.release()


def _write_sample_labels(label_path: Path, frame_count: int = 6) -> None:
	frame_idx = np.arange(frame_count)
	labels = pd.DataFrame(
		{
			"frame_idx": frame_idx,
			"time_sec": frame_idx / 30.0,
			"vmPFC": np.linspace(0.1, 0.2, frame_count),
			"insula": np.linspace(0.2, 0.3, frame_count),
			"ACC": np.linspace(0.3, 0.4, frame_count),
			"valence": np.linspace(-0.5, 0.5, frame_count),
			"arousal": np.linspace(0.5, -0.5, frame_count),
		}
	)
	label_path.parent.mkdir(parents=True, exist_ok=True)
	labels.to_parquet(label_path, index=False)


@pytest.fixture()
def sample_root(tmp_path: Path) -> Path:
	root = tmp_path / "saved_mri"
	chunk_dir = root / "Youtube_sample_chunk_000"
	_write_sample_video(chunk_dir / "source_video.mp4")
	_write_sample_labels(chunk_dir / "roi_timeseries.parquet")
	return root


def test_video_processor_falls_back_to_mnt(monkeypatch):
	def fake_exists(self):
		return str(self) == "/mnt/tribe-share/enma/saved_mri"

	monkeypatch.setattr(Path, "exists", fake_exists)
	processor = VideoProcessor()
	assert processor.dir_path == "/mnt/tribe-share/enma/saved_mri"


def test_video_dataset_loads_frames_and_labels(sample_root: Path):
	dataset = VideoFrameDataset(data_root=sample_root, frame_step=3, image_size=(32, 32))
	assert len(dataset) == 2

	frame, target = dataset[0]
	assert frame.shape == (1, 32, 32)
	assert target.shape == (2,)


def test_model_forward_shape(sample_root: Path):
	dataset = VideoFrameDataset(data_root=sample_root, frame_step=3, image_size=(32, 32))
	model = Net(input_shape=(1, 32, 32), num_steps=4)
	frame, _ = dataset[0]
	prediction, state = model(frame.unsqueeze(0))
	assert prediction.shape == (1, 2)
	assert state["membrane"]["output"].shape[-1] == 2


def test_plot_functions_write_files(tmp_path: Path):
	history = {
		"train_total_loss": [1.0, 0.8],
		"val_total_loss": [1.2, 0.9],
		"train_mse_valence": [0.6, 0.5],
		"val_mse_valence": [0.7, 0.6],
		"train_mse_arousal": [0.4, 0.3],
		"val_mse_arousal": [0.5, 0.4],
		"train_ccc_valence": [0.3, 0.25],
		"val_ccc_valence": [0.35, 0.3],
		"train_ccc_arousal": [0.2, 0.15],
		"val_ccc_arousal": [0.25, 0.2],
		"learning_rates": [5e-4, 4e-4],
		"gradient_norms": {"fc1": [1.0, 0.8], "fc2": [0.5, 0.4], "fc3": [0.2, 0.15]},
		"spike_rates": {"layer1": [0.15, 0.16], "layer2": [0.12, 0.11], "layer3": [0.08, 0.09]},
		"beta_values": {"layer1": [0.9, 0.91], "layer2": [0.92, 0.93], "layer3": [0.94, 0.95]},
	}
	preds = np.array([[0.1, 0.2], [0.3, 0.4]])
	targets = np.array([[0.0, 0.1], [0.2, 0.3]])
	times = np.array([0.0, 0.1])
	membrane = np.array([[0.01, 0.02], [0.03, 0.04]])

	assert plot_loss_curves(history, tmp_path).exists()
	assert plot_total_loss(history, tmp_path).exists()
	assert plot_learning_rate(history, tmp_path).exists()
	assert plot_gradient_norms(history, tmp_path).exists()
	assert plot_spike_rates(history, tmp_path).exists()
	assert plot_beta_values(history, tmp_path).exists()
	assert plot_scatter(preds, targets, tmp_path).exists()
	assert plot_timeseries(times, preds, targets, tmp_path).exists()
	assert plot_membrane_trace(times, membrane, tmp_path).exists()
 