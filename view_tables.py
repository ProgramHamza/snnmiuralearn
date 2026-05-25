#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd


def load_npy_as_frame(path: Path) -> pd.DataFrame:
    data = np.load(path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.dtype.names:
        return pd.DataFrame.from_records(data)

    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            return pd.DataFrame({"value": [data.item()]})
        if data.ndim == 1:
            return pd.DataFrame({"value": data})
        if data.ndim == 2:
            return pd.DataFrame(data)
        reshaped = data.reshape(data.shape[0], -1)
        return pd.DataFrame(reshaped)

    return pd.DataFrame(data)


def find_default(pattern: str) -> Path | None:
    matches = sorted(Path.cwd().rglob(pattern))
    return matches[0] if matches else None


def table_block(title: str, frame: pd.DataFrame) -> str:
    preview = frame.head(200).copy()
    return "\n".join(
        [
            f"<h2>{html.escape(title)}</h2>",
            f"<p>{len(frame):,} rows, {len(frame.columns):,} columns</p>",
            preview.to_html(index=False, border=0, classes="dataframe", escape=False),
        ]
    )


def load_parquet_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="View parquet files and an optional .npy as HTML tables.")
    parser.add_argument("--npy", type=Path, default=None, help="Optional path to brain_preds.npy")
    parser.add_argument(
        "--parquet",
        type=Path,
        nargs="+",
        default=None,
        help="One or more parquet files to view",
    )
    parser.add_argument("--out", type=Path, default=Path("table_view.html"), help="Output HTML file")
    parser.add_argument("--open", action="store_true", help="Open the HTML in a browser")
    args = parser.parse_args()

    npy_path = args.npy or find_default("brainsored.npy")
    parquet_paths = args.parquet or [
        path
        for path in [
            find_default("video_features.parquet"),
            find_default("roi_timeseries.parquet"),
        ]
        if path is not None
    ]

    if not parquet_paths:
        print("Could not find any parquet input files.", file=sys.stderr)
        print("Pass them explicitly with --parquet.", file=sys.stderr)
        return 1

    npy_frame = load_npy_as_frame(npy_path) if npy_path is not None else None
    parquet_frames = [(path, load_parquet_frame(path)) for path in parquet_paths]

    html_text = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<style>",
            "body{font-family:sans-serif;margin:24px;}",
            ".dataframe{border-collapse:collapse;}",
            ".dataframe th,.dataframe td{border:1px solid #ccc;padding:4px 8px;text-align:left;}",
            ".dataframe thead th{background:#f2f2f2;position:sticky;top:0;}",
            "</style></head><body>",
            f"<h1>Table View</h1><p>{html.escape('<br>'.join(str(path) for path in parquet_paths + ([npy_path] if npy_path is not None else [])))}</p>",
            *[
                table_block(path.name, frame)
                for path, frame in parquet_frames
            ],
            table_block(npy_path.name, npy_frame) if npy_frame is not None else "",
            "</body></html>",
        ]
    )

    args.out.write_text(html_text, encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")

    if args.open:
        webbrowser.open(args.out.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())