"""Create a CASE stimulus manifest and check which local video files are missing."""

from pathlib import Path
import csv
import zipfile
import xml.etree.ElementTree as ET

from config import CASE_ROOT, PROJECT_ROOT
from dataset import VIDEO_EXTENSIONS, scan_video_files


MANIFEST_PATH = PROJECT_ROOT / "case_video_manifest.csv"


def _xlsx_rows(path):
    """Read the first worksheet of a simple XLSX file without optional dependencies."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    try:
        with zipfile.ZipFile(path) as archive:
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for si in root.findall("a:si", ns):
                    shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            rows = []
            for row in sheet.findall(".//a:row", ns):
                values = []
                for cell in row.findall("a:c", ns):
                    value = cell.find("a:v", ns)
                    text = "" if value is None else value.text or ""
                    if cell.attrib.get("t") == "s" and text:
                        text = shared[int(text)]
                    values.append(text)
                rows.append(values)
            return rows
    except Exception as exc:
        raise RuntimeError(f"Could not read XLSX metadata {path}: {exc}") from exc


def write_manifest(save_path=MANIFEST_PATH):
    """Write a CSV manifest of CASE video labels, URLs, durations, and local status."""
    metadata_path = Path(CASE_ROOT) / "CASE_full" / "metadata" / "videos.xlsx"
    if not metadata_path.exists():
        metadata_path = Path(CASE_ROOT) / "metadata" / "videos.xlsx"
    rows = _xlsx_rows(metadata_path)
    headers = rows[0]
    data_rows = rows[2:] if len(rows) > 2 and "Start Time" in "\t".join(rows[1]) else rows[1:]
    videos = scan_video_files(CASE_ROOT)
    output_rows = []
    for row in data_rows:
        if not row or not row[0].strip():
            continue
        padded = row + [""] * max(0, len(headers) - len(row))
        record = dict(zip(headers, padded))
        label = record.get("Video-label", "").strip()
        if not label:
            continue
        local_path = videos.get(label.lower(), "")
        output_rows.append(
            {
                "video_label": label,
                "expected_filename": f"{label}.mp4",
                "duration_ms": record.get("Duration (in ms)", ""),
                "source": record.get("Source (Year)", ""),
                "source_url": record.get("Source--URL", ""),
                "video_url": record.get("Video Available at URL", ""),
                "local_path": str(local_path) if local_path else "",
                "status": "present" if local_path else "missing",
            }
        )
    try:
        save_path = Path(save_path)
        with save_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)
    except Exception as exc:
        raise RuntimeError(f"Could not write manifest {save_path}: {exc}") from exc
    return save_path, output_rows


def main():
    """CLI entry point for CASE video manifest generation."""
    save_path, rows = write_manifest()
    missing = [row for row in rows if row["status"] == "missing"]
    print(f"Wrote manifest: {save_path}")
    print(f"Detected local videos: {len(rows) - len(missing)} / {len(rows)}")
    if missing:
        print("Missing expected files:")
        for row in missing:
            if Path(row["video_label"]).suffix.lower() not in VIDEO_EXTENSIONS:
                print(f"  {row['expected_filename']}  source={row['video_url']}")


if __name__ == "__main__":
    main()

