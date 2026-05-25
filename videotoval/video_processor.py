from glob import glob
from pathlib import Path

import cv2 as cv2
import pandas as pd
from PIL import Image
from torchvision import transforms

import snntorch.spikegen as spikegen


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

    def resolve_label_path(self, video_path):
        return str(Path(video_path).with_name('roi_timeseries.parquet'))

    def load_label_table(self, video_path):
        label_path = Path(self.resolve_label_path(video_path))
        if not label_path.exists():
            return None

        label_frame = pd.read_parquet(label_path)
        expected_columns = {'frame_idx', 'time_sec', 'valence', 'arousal'}
        missing_columns = expected_columns.difference(label_frame.columns)
        if missing_columns:
            missing = ', '.join(sorted(missing_columns))
            raise ValueError(f'Missing label columns in {label_path}: {missing}')

        return label_frame.sort_values('frame_idx').reset_index(drop=True)

    def iter_labeled_frames(self, video_path, transform=None, frame_step=3, num_steps=1):
        label_frame = self.load_label_table(video_path)
        if label_frame is None:
            raise FileNotFoundError(f'Could not find roi_timeseries.parquet next to {video_path}')

        frame_transform = transform or self.transform
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f'Could not open video: {video_path}')

        try:
            for _, row in label_frame.iterrows():
                frame_idx = int(row['frame_idx'])
                if frame_idx % frame_step != 0:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, img = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img)
                processed = frame_transform(pil)
                spikes = self.spike_tensor(processed, num_steps=num_steps)
                target = pd.Series({'valence': row['valence'], 'arousal': row['arousal']})

                yield processed, spikes, target, frame_idx, float(row['time_sec'])
        finally:
            cap.release()

    def load_video_frames(self, video_path, transform=None, frame_step=3, num_steps=1):
        yield from self.iter_labeled_frames(
            video_path,
            transform=transform,
            frame_step=frame_step,
            num_steps=num_steps,
        )

    def iter_video_frames(self, video_path, transform=None, frame_step=3, num_steps=1):
        frame_transform = transform or self.transform
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f'Could not open video: {video_path}')

        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

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
                time_sec = frame_idx / fps if fps > 0 else frame_idx / 30.0

                yield processed, spikes, frame_idx, time_sec
        finally:
            cap.release()

    def spike_tensor(self, tensor, num_steps=1):
        return spikegen.rate(tensor, num_steps=num_steps)

    iterate_frames = load_video_frames



