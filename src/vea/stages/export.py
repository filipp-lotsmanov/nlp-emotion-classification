"""
Emotion Data Export Module (Stage 8)

Consolidates all emotion analysis results into a CSV file for further analysis.
Runs after visualization.py to create a comprehensive dataset combining:
- Original transcription segments
- Translations
- Russian and English VA predictions
- Russian and English emotion predictions
- Final ensemble emotion

Design rationale:
- CSV format for easy analysis in Excel/Python/R
- One row per segment for simple alignment
- All timestamps and texts included for traceability
- Handles missing data gracefully (not all files may exist)
- Preserves source information for debugging

Alternative approaches considered:
- Multiple CSVs per data type: More complex to analyze together
- JSON output: Less accessible for non-programmers
- Database: Overkill for this use case
- Current approach: Single CSV is simplest and most practical
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vea.config import canonicalize_emotion, display_emotion

# Logging is configured by the CLI (vea.cli) or by the calling application.
# Library modules must not call logging.basicConfig at import time.
logger = logging.getLogger(__name__)


# =============================================================================
# FILE DISCOVERY
# =============================================================================


def _vote_label(label: str) -> str:
    """Canonical class for the ensemble vote.

    The vote compared raw lowercased strings, so Russian `enthusiasm` and
    English `joy` counted as a disagreement despite being the same class. That
    is the whole of the 5-segment gap between this stage's agreement figures and
    stage 8's on the first full run. Unknown labels pass through as themselves,
    which counts as a disagreement - the honest outcome when the label's class
    is genuinely unknown.
    """
    return canonicalize_emotion(label, default=str(label).strip().lower())


def get_display_emotion(emotion: str) -> str:
    """Display name for a CSV cell, e.g. 'joy' and 'enthusiasm' -> 'Happiness'.

    Delegates to vea.config so this stage, the timeline and the ensemble vote
    cannot disagree about what a label means. The local map this replaced fell
    back to `emotion.capitalize()`, which let the Russian model's `enthusiasm`
    through untranslated and put one class in the CSV under two different
    values.
    """
    try:
        return display_emotion(emotion)
    except ValueError:
        # The CSV is the last stage: a new upstream label must not lose the run.
        # It is logged rather than raised, and it is a real problem - see
        # vea.config.EMOTION_ALIASES.
        logger.error(
            "unrecognised emotion label %r from an upstream stage; writing it "
            "through unchanged. Add it to vea.config.EMOTION_ALIASES.",
            emotion,
        )
        return str(emotion).capitalize()


def find_data_files(video_dir: Path) -> Dict[str, Optional[Path]]:
    """
    Find all relevant data files for CSV export.

    Returns dict with keys:
    - segments_ru: Original Russian segments
    - segments_en: Translated segments
    - arousal_ru: Russian valence-arousal predictions
    - arousal_en: English valence-arousal predictions
    - emotion_ru: Russian emotion predictions
    - emotion_en_distil: English DistilRoBERTa emotion predictions
    - emotion_en_deberta: English DeBERTa emotion predictions

    Design choice: Return None for missing files rather than raising errors
    - Allows partial exports when some pipeline stages haven't run
    - Graceful degradation better than complete failure
    - User can see which data is missing from log messages
    """
    video_dir = Path(video_dir)
    files = {}

    # Find Russian segments (base file)
    segments_ru_files = list(video_dir.glob("local_segments_transcription_*.json"))
    segments_ru_files = [f for f in segments_ru_files if "_translated" not in f.name]
    files["segments_ru"] = segments_ru_files[0] if segments_ru_files else None

    # Find translated segments
    segments_en_files = list(video_dir.glob("local_segments_*_translated.json"))
    files["segments_en"] = segments_en_files[0] if segments_en_files else None

    # Find Russian arousal
    arousal_ru_files = list(video_dir.glob("arousal_*_local_*.json"))
    arousal_ru_files = [f for f in arousal_ru_files if "_en_" not in f.name]
    files["arousal_ru"] = arousal_ru_files[0] if arousal_ru_files else None

    # Find English arousal
    arousal_en_files = list(video_dir.glob("arousal_*_en_local_*.json"))
    files["arousal_en"] = arousal_en_files[0] if arousal_en_files else None

    # Find Russian emotion
    emotion_ru_files = list(video_dir.glob("emotion_*_local_*.json"))
    emotion_ru_files = [f for f in emotion_ru_files if "_en_" not in f.name]
    files["emotion_ru"] = emotion_ru_files[0] if emotion_ru_files else None

    # Find English emotions (2 models)
    emotion_distil_files = list(video_dir.glob("emotion_distilroberta_en_local_*.json"))
    files["emotion_en_distil"] = emotion_distil_files[0] if emotion_distil_files else None

    emotion_deberta_files = list(video_dir.glob("emotion_deberta-finetuned_en_local_*.json"))
    files["emotion_en_deberta"] = emotion_deberta_files[0] if emotion_deberta_files else None

    # Log what we found
    logger.info("Data files discovered:")
    for key, path in files.items():
        if path:
            logger.info(f"  {key}: {path.name}")
        else:
            logger.warning(f"  {key}: NOT FOUND")

    return files


# =============================================================================
# DATA LOADING
# =============================================================================


def load_segments_data(file_path: Optional[Path]) -> List[Dict]:
    """
    Load segment data from JSON file.

    Returns list of segment dicts, or empty list if file doesn't exist.
    """
    if not file_path or not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("segments", [])


def load_predictions_data(file_path: Optional[Path]) -> List[Dict]:
    """
    Load prediction data (arousal or emotion) from JSON file.

    Returns list of prediction dicts, or empty list if file doesn't exist.
    """
    if not file_path or not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("predictions", [])


# =============================================================================
# ENSEMBLE CALCULATION (FROM VISUALIZATION.PY)
# =============================================================================


def calculate_final_emotion(
    emotion_ru: Optional[str],
    emotion_en_distil: Optional[str],
    emotion_en_deberta: Optional[str],
    confidence_ru: Optional[float],
    confidence_en_distil: Optional[float],
    confidence_en_deberta: Optional[float],
) -> Tuple[str, str]:
    """
    Calculate final ensemble emotion using same logic as visualization.py.

    Returns:
        Tuple of (final_emotion, agreement_level)

    Agreement levels:
    - 'full': All 3 models agree
    - 'majority': 2 of 3 agree
    - 'confidence': Used highest confidence when all disagree
    - 'single': Only 1 model available

    Design choice: Replicate visualization.py logic
    - Ensures consistency between CSV and visualization
    - Alternative would be loading visualization output, but that's more complex
    - Simple vote counting is reliable and easy to understand
    """
    # Collect available predictions
    predictions = []

    if emotion_ru:
        predictions.append(
            {
                "emotion": _vote_label(emotion_ru),
                "confidence": confidence_ru or 0.0,
                "source": "ru",
            }
        )

    if emotion_en_distil:
        predictions.append(
            {
                "emotion": _vote_label(emotion_en_distil),
                "confidence": confidence_en_distil or 0.0,
                "source": "distil",
            }
        )

    if emotion_en_deberta:
        predictions.append(
            {
                "emotion": _vote_label(emotion_en_deberta),
                "confidence": confidence_en_deberta or 0.0,
                "source": "deberta",
            }
        )

    if not predictions:
        return "unknown", "no_data"

    if len(predictions) == 1:
        return predictions[0]["emotion"], "single"

    # Count votes
    emotion_counts = {}
    for pred in predictions:
        emotion = pred["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    # Get most common emotion
    max_votes = max(emotion_counts.values())
    most_common_emotions = [e for e, count in emotion_counts.items() if count == max_votes]

    # Determine agreement level
    if max_votes == len(predictions) and len(predictions) >= 2:
        # Full agreement
        return most_common_emotions[0], "full"

    elif max_votes >= 2:
        # Majority agreement
        return most_common_emotions[0], "majority"

    else:
        # All disagree - use highest confidence
        best_pred = max(predictions, key=lambda x: x["confidence"])
        return best_pred["emotion"], "confidence"


# =============================================================================
# CSV EXPORT
# =============================================================================


def create_emotion_csv(video_dir: Path, output_filename: str = "emotion_analysis_data.csv") -> Dict:
    """
    Create comprehensive CSV with all emotion analysis data.

    CSV columns:
    1. segment_id: Sequential segment identifier
    2. start_time: Segment start (seconds)
    3. end_time: Segment end (seconds)
    4. duration: Segment duration (seconds)
    5. text_ru: Original Russian text
    6. text_en: English translation
    7. ru_arousal: Russian arousal (0-1)
    8. ru_valence: Russian valence (0-1)
    9. en_arousal: English arousal (0-1)
    10. en_valence: English valence (0-1)
    11. emotion_ru: Russian emotion prediction
    12. emotion_ru_confidence: Russian prediction confidence
    13. emotion_en_distil: English DistilRoBERTa prediction
    14. emotion_en_distil_confidence: DistilRoBERTa confidence
    15. emotion_en_deberta: English DeBERTa prediction
    16. emotion_en_deberta_confidence: DeBERTa confidence
    17. emotion_final: Final ensemble emotion
    18. emotion_agreement: Agreement level (full/majority/confidence/single)

    Design rationale for column order:
    - Identifiers and timestamps first (standard practice)
    - Text data next (what was said)
    - VA predictions grouped by language (continuous values)
    - Emotion predictions grouped by model (categorical values)
    - Final ensemble result last (derived from all above)
    """
    video_dir = Path(video_dir)
    logger.info(f"Creating emotion CSV for {video_dir.name}")

    # Find all data files
    files = find_data_files(video_dir)

    # Check if we have at least the base segments file
    if not files["segments_ru"]:
        logger.error("No Russian segments file found - cannot create CSV")
        return {"status": "failed", "reason": "no_base_segments"}

    # Load all data
    logger.info("Loading data files...")
    segments_ru = load_segments_data(files["segments_ru"])
    segments_en = load_segments_data(files["segments_en"])
    arousal_ru = load_predictions_data(files["arousal_ru"])
    arousal_en = load_predictions_data(files["arousal_en"])
    emotion_ru = load_predictions_data(files["emotion_ru"])
    emotion_en_distil = load_predictions_data(files["emotion_en_distil"])
    emotion_en_deberta = load_predictions_data(files["emotion_en_deberta"])

    logger.info(f"Loaded {len(segments_ru)} segments")

    # Create lookup dictionaries by segment index
    # Design choice: Use index-based matching rather than segment_id
    # - Simpler and more reliable (all files should have same segment order)
    # - segment_id may be missing or inconsistent across files
    # - Index matching is what the pipeline naturally produces

    arousal_ru_dict = {i: pred for i, pred in enumerate(arousal_ru)}
    arousal_en_dict = {i: pred for i, pred in enumerate(arousal_en)}
    emotion_ru_dict = {i: pred for i, pred in enumerate(emotion_ru)}
    emotion_en_distil_dict = {i: pred for i, pred in enumerate(emotion_en_distil)}
    emotion_en_deberta_dict = {i: pred for i, pred in enumerate(emotion_en_deberta)}
    segments_en_dict = {i: seg for i, seg in enumerate(segments_en)}

    # Build CSV rows
    logger.info("Building CSV data...")
    rows = []

    for i, segment_ru in enumerate(segments_ru):
        # Basic segment info
        row = {
            "segment_id": i,
            "start_time": segment_ru.get("start", 0.0),
            "end_time": segment_ru.get("end", 0.0),
            "duration": segment_ru.get("duration", 0.0),
            "text_ru": segment_ru.get("text", "").strip(),
            "text_en": segments_en_dict.get(i, {}).get("text_en", "").strip(),
        }

        # Russian VA
        if i in arousal_ru_dict:
            row["ru_arousal"] = arousal_ru_dict[i].get("arousal")
            row["ru_valence"] = arousal_ru_dict[i].get("valence")
        else:
            row["ru_arousal"] = None
            row["ru_valence"] = None

        # English VA
        if i in arousal_en_dict:
            row["en_arousal"] = arousal_en_dict[i].get("arousal")
            row["en_valence"] = arousal_en_dict[i].get("valence")
        else:
            row["en_arousal"] = None
            row["en_valence"] = None

        # Russian emotion
        if i in emotion_ru_dict:
            row["emotion_ru"] = get_display_emotion(emotion_ru_dict[i].get("predicted_emotion", ""))
            row["emotion_ru_confidence"] = emotion_ru_dict[i].get("confidence")
        else:
            row["emotion_ru"] = ""
            row["emotion_ru_confidence"] = None

        # English emotion (DistilRoBERTa)
        if i in emotion_en_distil_dict:
            row["emotion_en_distil"] = get_display_emotion(
                emotion_en_distil_dict[i].get("predicted_emotion", "")
            )
            row["emotion_en_distil_confidence"] = emotion_en_distil_dict[i].get("confidence")
        else:
            row["emotion_en_distil"] = ""
            row["emotion_en_distil_confidence"] = None

        # English emotion (DeBERTa)
        if i in emotion_en_deberta_dict:
            row["emotion_en_deberta"] = get_display_emotion(
                emotion_en_deberta_dict[i].get("predicted_emotion", "")
            )
            row["emotion_en_deberta_confidence"] = emotion_en_deberta_dict[i].get("confidence")
        else:
            row["emotion_en_deberta"] = ""
            row["emotion_en_deberta_confidence"] = None

        # Calculate final ensemble emotion
        final_emotion, agreement = calculate_final_emotion(
            row.get("emotion_ru"),
            row.get("emotion_en_distil"),
            row.get("emotion_en_deberta"),
            row.get("emotion_ru_confidence"),
            row.get("emotion_en_distil_confidence"),
            row.get("emotion_en_deberta_confidence"),
        )

        row["emotion_final"] = get_display_emotion(final_emotion)
        row["emotion_agreement"] = agreement

        rows.append(row)

    # Write CSV
    logger.info("Writing CSV file...")
    output_path = video_dir / output_filename

    # Column order (explicit for clarity)
    fieldnames = [
        "segment_id",
        "start_time",
        "end_time",
        "duration",
        "text_ru",
        "text_en",
        "ru_arousal",
        "ru_valence",
        "en_arousal",
        "en_valence",
        "emotion_ru",
        "emotion_ru_confidence",
        "emotion_en_distil",
        "emotion_en_distil_confidence",
        "emotion_en_deberta",
        "emotion_en_deberta_confidence",
        "emotion_final",
        "emotion_agreement",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved CSV: {output_path.name}")
    logger.info(f"  Total rows: {len(rows)}")
    logger.info(f"  Columns: {len(fieldnames)}")

    # Calculate statistics
    complete_rows = sum(
        1
        for row in rows
        if all(
            [
                row["text_ru"],
                row["text_en"],
                row["emotion_ru"],
                row["emotion_en_distil"],
                row["emotion_en_deberta"],
            ]
        )
    )

    agreement_counts = {}
    for row in rows:
        level = row["emotion_agreement"]
        agreement_counts[level] = agreement_counts.get(level, 0) + 1

    return {
        "status": "success",
        "output_file": str(output_path),
        "total_rows": len(rows),
        "complete_rows": complete_rows,
        "complete_percentage": round(complete_rows / len(rows) * 100, 1) if rows else 0,
        "agreement_breakdown": agreement_counts,
        "data_sources": {key: str(path.name) if path else "MISSING" for key, path in files.items()},
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def process_all_videos(
    downloads_dir: Path, output_filename: str = "emotion_analysis_data.csv"
) -> List[Dict]:
    """
    Process all video directories and create CSVs.

    Design choice: Process all videos rather than requiring per-video invocation
    - Matches pipeline pattern from other modules
    - Convenient for batch processing
    - Easy to parallelize if needed in future
    """
    downloads_dir = Path(downloads_dir)
    video_dirs = sorted(downloads_dir.glob("video-*"))

    if not video_dirs:
        logger.warning(f"No video directories found in {downloads_dir}")
        return []

    logger.info(f"Found {len(video_dirs)} video(s) to process")
    logger.info("=" * 70)

    results = []

    for video_dir in video_dirs:
        logger.info(f"\nProcessing: {video_dir.name}")
        logger.info("-" * 70)

        try:
            result = create_emotion_csv(video_dir, output_filename)
            results.append({"video_dir": video_dir.name, **result})

            if result["status"] == "success":
                logger.info("Success!")
                logger.info(f"  Output: {result['output_file']}")
                logger.info(f"  Total rows: {result['total_rows']}")
                logger.info(
                    f"  Complete rows: {result['complete_rows']} "
                    f"({result['complete_percentage']:.1f}%)"
                )
                logger.info("  Agreement breakdown:")
                for level, count in result["agreement_breakdown"].items():
                    pct = count / result["total_rows"] * 100
                    logger.info(f"    {level}: {count} ({pct:.1f}%)")
            else:
                logger.error(f"Failed: {result.get('reason', 'unknown')}")

        except Exception as e:
            logger.error(f"Error processing {video_dir.name}: {e}")
            import traceback

            traceback.print_exc()

            results.append({"video_dir": video_dir.name, "status": "error", "error": str(e)})

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    """
    Export emotion analysis data to CSV for each video.
    
    Creates comprehensive CSV with:
    - Original transcription and translation
    - Russian and English VA predictions
    - Multiple emotion model predictions
    - Final ensemble emotion with agreement level
    
    Usage:
        python 1emotion_export.py
    
    Output:
        downloads/video-*/emotion_analysis_data.csv
    """
    import sys

    downloads_dir = Path("downloads")

    if not downloads_dir.exists():
        print("Downloads directory not found")
        sys.exit(1)

    print("=" * 70)
    print("EMOTION DATA EXPORT TO CSV (Stage 8)")
    print("=" * 70)
    print("\nThis module consolidates all emotion analysis results into CSV files.")
    print("\nOutput columns:")
    print("  1. segment_id, start_time, end_time, duration")
    print("  2. text_ru, text_en (original and translated text)")
    print("  3. ru_arousal, ru_valence (Russian VA predictions)")
    print("  4. en_arousal, en_valence (English VA predictions)")
    print("  5. emotion_ru + confidence (Russian emotion)")
    print("  6. emotion_en_distil + confidence (English DistilRoBERTa)")
    print("  7. emotion_en_deberta + confidence (English DeBERTa)")
    print("  8. emotion_final, emotion_agreement (ensemble result)")
    print("=" * 70)

    # Process all videos
    results = process_all_videos(downloads_dir)

    # Summary
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    print(f"Total videos: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if successful:
        print("\nSuccessful exports:")
        for result in successful:
            print(f"  {result['video_dir']}:")
            print(f"    Rows: {result['total_rows']}")
            print(f"    Complete: {result['complete_percentage']:.1f}%")

    if failed:
        print("\nFailed exports:")
        for result in failed:
            print(f"  {result['video_dir']}: {result.get('reason') or result.get('error')}")

    print("\n" + "=" * 70)
    print("CSV files saved to: downloads/video-*/emotion_analysis_data.csv")
    print("=" * 70)
    print("\nUsage tips:")
    print("  - Open CSV in Excel/LibreOffice for manual inspection")
    print("  - Import into pandas for statistical analysis")
    print("  - Use emotion_agreement column to filter high-confidence segments")
    print("  - Compare ru_*/en_* columns to analyze translation effects")
    print("  - Check complete_percentage to identify missing pipeline stages")
