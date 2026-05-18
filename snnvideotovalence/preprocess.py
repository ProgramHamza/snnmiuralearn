"""Video and audio preprocessing utilities."""

from pathlib import Path
import tempfile
import warnings

import numpy as np
import torch


def _expected_frames(duration: float, fps: int) -> int:
    """Return the number of frames expected for a clip duration and FPS."""
    return max(1, int(round(float(duration) * int(fps))))


def load_frame_diff(video_path, start_sec, duration, fps=2, img_size=128):
    """Extract grayscale frame differences normalized to [0, 1]."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required: pip install opencv-python") from exc

    video_path = Path(video_path)
    n_frames = _expected_frames(duration, fps)
    if not video_path.exists():
        warnings.warn(f"Video file not found: {video_path}")
        return np.zeros((n_frames, 1, img_size, img_size), dtype=np.float32)

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            warnings.warn(f"Could not open video file: {video_path}")
            return np.zeros((n_frames, 1, img_size, img_size), dtype=np.float32)

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        for i in range(n_frames):
            target_sec = float(start_sec) + (i / float(fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_sec * source_fps))
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
            frames.append(gray.astype(np.float32))
        cap.release()

        if not frames:
            warnings.warn(f"No frames extracted from {video_path}")
            return np.zeros((n_frames, 1, img_size, img_size), dtype=np.float32)

        while len(frames) < n_frames:
            frames.append(frames[-1].copy())

        diffs = []
        prev = None
        for frame in frames[:n_frames]:
            diff = np.zeros_like(frame, dtype=np.float32) if prev is None else frame - prev
            diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            diffs.append(diff[None, :, :].astype(np.float32))
            prev = frame
        return np.stack(diffs).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to load frame differences from {video_path}: {exc}") from exc


def _load_audio_with_librosa(video_path, start_sec, duration, sr):
    """Load audio from a media file using librosa."""
    import librosa

    return librosa.load(str(video_path), sr=sr, mono=True, offset=float(start_sec), duration=float(duration))[0]


def _load_audio_with_moviepy(video_path, start_sec, duration, sr):
    """Extract a clip's audio using moviepy and return a mono waveform."""
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        from moviepy import VideoFileClip

    with VideoFileClip(str(video_path)) as clip:
        audio = clip.audio
        if audio is None:
            return np.zeros(int(duration * sr), dtype=np.float32)
        sub = audio.subclip(float(start_sec), min(float(start_sec) + float(duration), clip.duration))
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sub.write_audiofile(tmp.name, fps=sr, nbytes=2, codec="pcm_s16le", logger=None)
            import librosa

            y, _ = librosa.load(tmp.name, sr=sr, mono=True)
            return y


def load_mfcc(video_path, start_sec, duration, sr=16000, n_mfcc=40, n_frames=None):
    """Extract normalized MFCC features and interpolate to one vector per frame."""
    video_path = Path(video_path)
    target_frames = int(n_frames) if n_frames is not None else _expected_frames(duration, 2)
    if not video_path.exists():
        warnings.warn(f"Video file not found for MFCC extraction: {video_path}")
        return np.zeros((target_frames, n_mfcc), dtype=np.float32)

    try:
        import librosa
    except ImportError as exc:
        raise ImportError("librosa is required for MFCC extraction: pip install librosa") from exc

    try:
        try:
            y = _load_audio_with_librosa(video_path, start_sec, duration, sr)
        except Exception:
            y = _load_audio_with_moviepy(video_path, start_sec, duration, sr)

        if y.size == 0:
            return np.zeros((target_frames, n_mfcc), dtype=np.float32)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=512).astype(np.float32)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        mfcc = mfcc.T
        if mfcc.shape[0] == target_frames:
            return mfcc.astype(np.float32)
        x_old = np.linspace(0.0, 1.0, mfcc.shape[0])
        x_new = np.linspace(0.0, 1.0, target_frames)
        pooled = np.stack([np.interp(x_new, x_old, mfcc[:, i]) for i in range(n_mfcc)], axis=1)
        return pooled.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract MFCC from {video_path}: {exc}") from exc


def rate_encode(frame_array, T=16):
    """Rate encode frames by Bernoulli sampling repeated over T timesteps."""
    arr = torch.as_tensor(frame_array, dtype=torch.float32).clamp(0.0, 1.0)
    probs = arr.unsqueeze(1).repeat(1, int(T), 1, 1, 1)
    return torch.bernoulli(probs).float()

