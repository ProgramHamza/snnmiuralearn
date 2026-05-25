import pandas as pd
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

from glob import glob

import IPython.display as ipd
from tqdm import tqdm

import subprocess
from torchvision import transforms
from PIL import Image
import torch
import snntorch.spikegen as spikegen
from pathlib import Path


class VideoProcessor:
    def __init__(self, dir_path=None, transform=None):
        self.dir_path = self._resolve_dir_path(dir_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((360, 640)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ])

    def _resolve_dir_path(self, dir_path):
        candidates = []
        if dir_path is not None:
            candidates.append(Path(dir_path))
        candidates.extend([
            Path('/srv/tribe-share/enma/saved_mri'),
            Path('/mnt/tribe-share/enma/saved_mri'),
        ])

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return str(candidates[-1]) if candidates else '/mnt/tribe-share/enma/saved_mri'

    def iterate_train_videos(self, dir_path=None):
        search_root = dir_path or self.dir_path
        video_files = sorted(glob(f"{search_root}/Youtube_*_chunk_*/source_video.mp4"))

        if not video_files:
            video_files = sorted(glob(f"{search_root}/**/source_video.mp4", recursive=True))

        n_videos = len(video_files)
        n_train = int(n_videos * 0.9) if n_videos > 1 else n_videos

        for video_file in video_files[:n_train]:
            yield video_file
    
    def iterate_test_videos(self, dir_path=None):
        search_root = dir_path or self.dir_path
        video_files = sorted(glob(f"{search_root}/Youtube_*_chunk_*/source_video.mp4"))

        if not video_files:
            video_files = sorted(glob(f"{search_root}/**/source_video.mp4", recursive=True))

        n_videos = len(video_files)
        n_test = n_videos - int(n_videos * 0.9) if n_videos > 1 else n_videos

        for video_file in video_files[-n_test:]:
            yield video_file

    def load_video_frames(self, video_path, transform=None, frame_step=3, num_steps=1):
        frame_transform = transform or self.transform
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f'Could not open video: {video_path}')

        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for frame_idx in range(n_frames):
                ret, img = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step != 0:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img)
                processed = frame_transform(pil)
                spikes = self.spike_tensor(processed, num_steps=num_steps)

                yield processed, spikes
        finally:
            cap.release()

    def spike_tensor(self, tensor, num_steps=1):
        return spikegen.rate(tensor, num_steps=num_steps)

    iterate_frames = load_video_frames



