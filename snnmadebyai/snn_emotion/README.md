# CASE Emotion SNN

This project implements a convolutional spiking neural network for emotion estimation from CASE video clips.

## Structure

```text
snn_emotion/
├── data/
│   └── case_dataset.py
├── models/
│   ├── encoder.py
│   ├── spike_encoder.py
│   └── snn_classifier.py
├── train.py
├── infer.py
├── visualize.py
└── config.py
```

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Optional but useful for some video/audio containers:

```bash
pip install soundfile
```

## Data Layout

`CASEDataset` scans the dataset root recursively for `.csv` annotation files and matching video files in the same folder or nearby.

The CASE dataset is not bundled with this project and is not downloaded automatically. You need to obtain the video stimuli and annotation CSVs separately, then point `--data-root` at the local CASE directory.

Expected files:

- a video stimulus file such as `.mp4`, `.avi`, `.mov`, or `.mkv`
- a matching annotation CSV containing continuous valence/arousal values at about 2 Hz

## Training

```bash
python train.py --data-root /path/to/case_dataset
```

Key defaults:

- epochs: 50
- batch size: 16
- optimizer: Adam with lr=1e-3
- scheduler: CosineAnnealingLR
- best checkpoint: `checkpoints/best.pth`

## Inference

```bash
python infer.py --video path/to/video.mp4
```

The script prints a JSON-like dictionary to stdout and saves the visualization to `output/emotion_result.png`.

## Notes

- Audio preprocessing uses librosa with 16 kHz mono audio.
- Video preprocessing samples frames at 2 fps, resizes them to 112x112, converts to grayscale, and extracts HOG descriptors.
- Spike encoding uses rate coding with Bernoulli sampling over `T_SPIKE = 16` steps.
- The SNN uses a custom alpha-operated LIF node built on top of SpikingJelly's `ParametricLIFNode`.
