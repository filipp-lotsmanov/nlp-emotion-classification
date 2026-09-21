"""Read finished runs off disk: the CSV, the timeline, the per-video summary.

Stage 9 writes `emotion_analysis_data.csv` and stage 8 writes
`emotion_timeline.png` into `downloads/video-<id>/`. This module is the only
thing that knows that layout, so the API and the frontend never do.

Every emotion label that leaves here has been through
`vea.config.canonicalize_emotion`. The CSV holds display names (`Happiness`),
the JSON files hold raw model output (`enthusiasm`, `joy`), and treating those
as different strings is exactly the defect that made stages 8 and 9 disagree -
see docs/PROVENANCE.md section 13. The frontend gets one vocabulary.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vea.config import EMOTION_CLASSES, canonicalize_emotion, display_emotion

CSV_NAME = "emotion_analysis_data.csv"
TIMELINE_NAME = "emotion_timeline.png"
VIDEO_DIR_PREFIX = "video-"

#: Columns carrying a model's emotion decision, and the name the UI uses.
MODEL_COLUMNS = {
    "emotion_ru": "Russian (rubert-tiny2)",
    "emotion_en_distil": "English (DistilRoBERTa)",
    "emotion_en_deberta": "English (DeBERTa)",
}


@dataclass(frozen=True)
class RunSummary:
    """Enough to list a run without parsing all of its rows."""

    video_id: str
    segments: int
    has_timeline: bool
    duration_seconds: float
    neutral_share: float
    agreement: dict[str, int]
    emotions: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "segments": self.segments,
            "has_timeline": self.has_timeline,
            "duration_seconds": round(self.duration_seconds, 1),
            "neutral_share": round(self.neutral_share, 4),
            "agreement": self.agreement,
            "emotions": self.emotions,
        }


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def video_dir(downloads: Path, video_id: str) -> Path:
    """Path to one run's directory.

    Raises:
        ValueError: on a video id containing a path separator or `..`. The id
            reaches this from an HTTP route, so it is checked rather than
            trusted - without it, `GET /api/runs/..%2F..%2Fetc/timeline.png`
            would read whatever the process can.
    """
    if not video_id or "/" in video_id or "\\" in video_id or ".." in video_id:
        raise ValueError(f"invalid video id: {video_id!r}")
    return Path(downloads) / f"{VIDEO_DIR_PREFIX}{video_id}"


def list_runs(downloads: Path) -> list[str]:
    """Video ids that have a stage 9 CSV, newest first."""
    downloads = Path(downloads)
    if not downloads.is_dir():
        return []
    found = [
        (path.stat().st_mtime, path.name[len(VIDEO_DIR_PREFIX) :])
        for path in downloads.glob(f"{VIDEO_DIR_PREFIX}*")
        if (path / CSV_NAME).is_file()
    ]
    return [video_id for _, video_id in sorted(found, reverse=True)]


def read_segments(downloads: Path, video_id: str) -> list[dict]:
    """Every row of one run's CSV, with canonical emotion labels.

    Each row keeps the model's own label under `<model>_raw` alongside the
    canonical one, because the difference between them is a finding in this
    project rather than noise: the Russian model emits `enthusiasm`, which is
    Joy, and the pipeline used to call it Neutral.

    Raises:
        FileNotFoundError: if the run has no stage 9 output.
    """
    path = video_dir(downloads, video_id) / CSV_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {CSV_NAME} for {video_id}; stage 9 has not run")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, raw in enumerate(csv.DictReader(handle)):
            row: dict = {
                "segment_id": raw.get("segment_id") or index,
                "start_time": _float(raw.get("start_time")) or 0.0,
                "end_time": _float(raw.get("end_time")) or 0.0,
                "duration": _float(raw.get("duration")),
                "text_ru": raw.get("text_ru") or "",
                "text_en": raw.get("text_en") or "",
                "ru_valence": _float(raw.get("ru_valence")),
                "ru_arousal": _float(raw.get("ru_arousal")),
                "en_valence": _float(raw.get("en_valence")),
                "en_arousal": _float(raw.get("en_arousal")),
                "agreement": raw.get("emotion_agreement") or "unknown",
            }
            for column in MODEL_COLUMNS:
                value = raw.get(column) or ""
                row[f"{column}_raw"] = value
                # default=: a label nobody has mapped yet must not 500 the
                # whole run. It shows as itself and stands out in the UI.
                row[column] = canonicalize_emotion(value, default="") if value else ""
            final = raw.get("emotion_final") or ""
            row["emotion_final_raw"] = final
            row["emotion_final"] = canonicalize_emotion(final, default="") if final else ""
            row["confidence"] = _float(raw.get("emotion_en_deberta_confidence"))
            rows.append(row)
    return rows


def summarize(downloads: Path, video_id: str) -> RunSummary:
    """Headline figures for one run, computed from its rows."""
    segments = read_segments(downloads, video_id)
    finals = [row["emotion_final"] for row in segments if row["emotion_final"]]
    neutral = sum(1 for label in finals if label == "neutral")
    return RunSummary(
        video_id=video_id,
        segments=len(segments),
        has_timeline=(video_dir(downloads, video_id) / TIMELINE_NAME).is_file(),
        duration_seconds=max((row["end_time"] for row in segments), default=0.0),
        neutral_share=(neutral / len(finals)) if finals else 0.0,
        agreement=dict(Counter(row["agreement"] for row in segments)),
        emotions=dict(Counter(finals)),
    )


def per_model_counts(downloads: Path, video_id: str) -> dict[str, dict[str, int]]:
    """Label distribution per model, canonicalised.

    This is the table that shows the register mismatch: every model
    Neutral-dominant, and the Russian model finding no Disgust at all while
    both English ones do.
    """
    segments = read_segments(downloads, video_id)
    out: dict[str, dict[str, int]] = {}
    for column, label in MODEL_COLUMNS.items():
        counts = Counter(row[column] for row in segments if row[column])
        out[label] = {name: counts.get(name, 0) for name in EMOTION_CLASSES}
    return out


def palette() -> dict[str, str]:
    """Canonical class -> the hex colour the timeline PNG uses for it.

    Read from `visualize.EMOTION_CONFIG` so the browser and the rendered PNG
    cannot drift apart. Imported lazily: that module pulls in matplotlib, which
    the viewer image does not install.
    """
    from vea.stages.visualize import EMOTION_CONFIG

    return {name: config["color"] for name, config in EMOTION_CONFIG.items()}


def display_names() -> dict[str, str]:
    """Canonical class -> the label shown to a reader."""
    return {name: display_emotion(name) for name in EMOTION_CLASSES}
