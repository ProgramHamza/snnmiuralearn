from __future__ import annotations

from pathlib import Path


SR = 16000
N_MELS = 64
HOP_LENGTH = 512
CLIP_DURATION = 5
T_SPIKE = 16
TAU_R = 0.5
TAU_D = 2.0
HIDDEN = 512
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 16

VIDEO_FPS = 2.0
FRAME_SIZE = (112, 112)
N_HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)

TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

EMOTION_NAMES = [
    "happiness",
    "boredom",
    "sadness",
    "scared",
    "frustrated",
    "surprised",
    "neutral",
]

EMOTION_DISPLAY_NAMES = {
    "happiness": "Happiness",
    "boredom": "Boredom",
    "sadness": "Sadness",
    "scared": "Scared",
    "frustrated": "Frustrated",
    "surprised": "Surprised",
    "neutral": "Neutral",
}

OUTPUT_DIR = Path("output")
CHECKPOINT_DIR = Path("checkpoints")
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pth"

FEATURE_LENGTH = (CLIP_DURATION * SR + HOP_LENGTH - 1) // HOP_LENGTH
