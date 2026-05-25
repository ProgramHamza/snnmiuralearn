# SNN Video To Valence

This project trains a PyTorch ANN to predict continuous valence and arousal in `[-1, 1]` from video frame differences and audio MFCCs, then converts the ANN to an SNN with SpikingJelly `ann2snn`.

## Setup

```bash
cd /snnlearn/snnvideotovalence
pip install torch torchvision torchaudio spikingjelly librosa moviepy opencv-python scipy pandas matplotlib numpy
```

In this environment `/snnlearn` was not present, so the code falls back to `/home/aisl/snnlearn`. The requested CASE path is still represented with `pathlib.Path` in `config.py`.

## CASE Data Layout

The local CASE dataset was inspected. It contains:

```text
CASEDataset/CASE_full/data/interpolated/annotations/sub_*.csv
CASEDataset/CASE_full/data/non-interpolated/annotations/sub_*.csv
```

The interpolated annotation CSVs have columns:

```text
jstime,valence,arousal,video
```

This copy does not include the copyrighted stimulus videos. The loader scans recursively for `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, and `.m4v` files and matches them to annotation `video` IDs by filename stem. Put videos anywhere under the CASE root, preferably in a `videos/` or `stimuli/` directory, with stems such as `amusing-1.mp4`, `boring-2.mp4`, or `scary-1.mp4`.

Generate a manifest of the required stimulus files and metadata URLs:

```bash
python prepare_videos.py
```

This writes:

```text
case_video_manifest.csv
```

The official CASE README says the videos cannot be shared due to copyright. This project therefore does not download copyrighted videos automatically. After you obtain or recreate clips that you have rights to use, place them under:

```text
/snnlearn/CASEDataset/videos/
```

or, in this container:

```text
/home/aisl/snnlearn/CASEDataset/videos/
```

Use these expected filenames:

```text
amusing-1.mp4
amusing-2.mp4
boring-1.mp4
boring-2.mp4
relaxed-1.mp4
relaxed-2.mp4
scary-1.mp4
scary-2.mp4
startVid.mp4
endVid.mp4
bluVid.mp4
```

## Training

```bash
python train.py
python train.py --epochs 10 --batch_size 8 --lr 0.001
```

Outputs:

```text
checkpoints/ann_best.pth
outputs/training_curves.png
```

## Conversion

```bash
python convert.py
```

Outputs:

```text
checkpoints/snn_converted.pth
outputs/ann_vs_snn_comparison.png
```

The script prints ANN-vs-SNN MSE and SNN validation Pearson correlations.

## Inference

```bash
python infer.py --video myvideo.mp4 --mode snn
python infer.py --video myvideo.mp4 --mode ann --start 0 --duration 5
```

Output:

```text
Valence: X.XX  (range -1 to 1, positive = pleasant)
Arousal: X.XX  (range -1 to 1, positive = excited)
Emotion quadrant: EXCITED / FEARFUL / CALM / BORED
outputs/infer_result.png
```

## Validation

Validate on the CASE validation split:

```bash
python validate.py --mode ann
python validate.py --mode snn
```

Run over a full video as a clip timeline:

```bash
python validate.py --mode ann --video myvideo.mp4
```

Output:

```text
outputs/validation_timeline.png
```

## Smoke Test

```bash
python dataset.py
```

This prints the detected media count and one sample shape if matching stimulus videos are present. Without videos it warns and exits cleanly with zero samples.
