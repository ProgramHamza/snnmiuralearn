"""ANN model architecture compatible with SpikingJelly ANN-to-SNN conversion."""

import torch
from torch import nn


class EmotionANN(nn.Module):
    """Two-stream visual/audio ANN that predicts valence and arousal."""

    def __init__(self, n_mfcc=40):
        """Initialize visual, audio, and fusion streams."""
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.audio = nn.Sequential(
            nn.Linear(n_mfcc, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fusion = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            nn.Tanh(),
        )

    def forward(self, frames, mfcc=None):
        """Predict valence and arousal from frame and MFCC tensors."""
        if mfcc is None:
            if isinstance(frames, dict):
                mfcc = frames["mfcc"]
                frames = frames["frames"]
            elif isinstance(frames, (tuple, list)) and len(frames) == 2:
                frames, mfcc = frames
            elif hasattr(frames, "frames") and hasattr(frames, "mfcc"):
                mfcc = frames.mfcc
                frames = frames.frames
            else:
                raise ValueError("EmotionANN requires frames and MFCC inputs.")
        visual = self.visual(frames)
        audio = self.audio(mfcc)
        return self.fusion(torch.cat([visual, audio], dim=1))
