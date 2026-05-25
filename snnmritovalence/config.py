"""Central configuration for the video/audio valence-arousal project."""

from pathlib import Path

import torch


REQUESTED_ROOT = Path("/snnlearn")
LOCAL_ROOT = Path("/home/aisl/snnlearn")
PROJECT_ROOT = Path(__file__).resolve().parent

CASE_ROOT = Path("/snnlearn/CASEDataset")
if not CASE_ROOT.exists():
    CASE_ROOT = LOCAL_ROOT / "CASEDataset"

OUTPUT_DIR = Path("/snnlearn/snnvideotovalence/outputs")
if not OUTPUT_DIR.parent.exists():
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = Path("/snnlearn/snnvideotovalence/checkpoints")
if not CHECKPOINT_DIR.parent.exists():
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

IMG_SIZE = 128
N_MFCC = 40
SR = 16000
CLIP_DURATION = 5
FPS = 2
T_SNN = 16
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
VAL_SPLIT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANN_CHECKPOINT = CHECKPOINT_DIR / "ann_best.pth"
SNN_CHECKPOINT = CHECKPOINT_DIR / "snn_converted.pth"


def ensure_dirs() -> None:
    """Create output and checkpoint directories if they do not exist."""
    for directory in (OUTPUT_DIR, CHECKPOINT_DIR):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Could not create directory {directory}: {exc}") from exc

