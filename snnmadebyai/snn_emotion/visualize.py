from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import EMOTION_DISPLAY_NAMES, EMOTION_NAMES, OUTPUT_DIR


EMOTION_COLORS = {
    "happiness": "#2ca02c",
    "boredom": "#7f7f7f",
    "sadness": "#1f77b4",
    "scared": "#ff7f0e",
    "frustrated": "#d62728",
    "surprised": "#bcbd22",
    "neutral": "#17becf",
}


def render_emotion_result(result: Mapping[str, float], output_path: str | Path = OUTPUT_DIR / "emotion_result.png") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valence = float(result.get("valence", 0.0))
    arousal = float(result.get("arousal", 0.0))
    emotion_scores = [float(result.get(name, 0.0)) for name in EMOTION_NAMES]

    figure, (axis_va, axis_bar) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    axis_va.set_title("Valence-Arousal Space")
    axis_va.axhline(0.0, color="#9ca3af", linewidth=1)
    axis_va.axvline(0.0, color="#9ca3af", linewidth=1)
    axis_va.set_xlim(-1.05, 1.05)
    axis_va.set_ylim(-1.05, 1.05)
    axis_va.set_xlabel("Valence")
    axis_va.set_ylabel("Arousal")
    axis_va.grid(alpha=0.2)

    quadrants = [
        (-0.75, 0.75, "FEARFUL"),
        (0.55, 0.75, "EXCITED"),
        (0.55, -0.8, "CALM"),
        (-0.8, -0.8, "SAD / BORED"),
    ]
    for x, y, label in quadrants:
        axis_va.text(x, y, label, fontsize=11, weight="bold", alpha=0.8)

    axis_va.scatter([valence], [arousal], s=160, color="#111827", edgecolor="#f97316", linewidth=2.5, zorder=5)
    axis_va.scatter([valence], [arousal], s=40, color="#f97316", zorder=6)

    bar_labels = [EMOTION_DISPLAY_NAMES[name] for name in EMOTION_NAMES]
    bar_colors = [EMOTION_COLORS[name] for name in EMOTION_NAMES]
    axis_bar.barh(bar_labels, emotion_scores, color=bar_colors)
    axis_bar.set_xlim(0, 100)
    axis_bar.set_xlabel("Score")
    axis_bar.set_title("Emotion Distribution")
    axis_bar.grid(axis="x", alpha=0.2)

    for index, score in enumerate(emotion_scores):
        axis_bar.text(score + 1.5, index, f"{score:.1f}", va="center", fontsize=9)

    figure.suptitle("CASE Emotion Prediction", fontsize=16, weight="bold")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    sample = {
        "valence": 0.2,
        "arousal": 0.7,
        "happiness": 58.0,
        "boredom": 7.0,
        "sadness": 5.0,
        "scared": 10.0,
        "frustrated": 12.0,
        "surprised": 8.0,
        "neutral": 0.0,
    }
    render_emotion_result(sample)
