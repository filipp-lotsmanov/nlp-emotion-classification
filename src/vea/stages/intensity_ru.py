"""
Arousal-Valence Classification Module - Continuous Scale Approach

This module predicts emotional arousal and valence for Russian text segments
using multilingual XLM-RoBERTa models, keeping values CONTINUOUS (0-1 scale)
instead of converting to discrete intensity levels.

Design rationale:
- Continuous arousal preserves more information than discrete bins
- Video-specific threshold adapts to content baseline automatically
- Binary filter (emotionally significant Y/N) is simpler than 5-level scale
- Matches academic best practices in affective computing
- Better for smooth timeline visualizations

Alternative approaches considered:
- 5-point intensity scale: Loses information, arbitrary thresholds
- 7-10 point scales: Even worse discretization
- Fixed thresholds: Don't adapt to content type
- Current approach: Research-backed, preserves data, adapts to content

Output format:
- arousal: Continuous [0-1], higher = more activated/excited
- valence: Continuous [0-1], higher = more positive/pleasant
- is_emotionally_significant: Binary flag for emotion classification
- threshold_used: Video-specific cutoff (median arousal)
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Logging is configured by the CLI (vea.cli) or by the calling application.
# Library modules must not call logging.basicConfig at import time.
logger = logging.getLogger(__name__)


class ValenceArousalClassifier:
    """
    Predict valence and arousal for Russian text segments.

    Uses multilingual XLM-RoBERTa models that work directly on Russian text
    without translation. Returns continuous values (0-1 scale) for maximum
    information preservation.

    Why continuous output:
    - Preserves all information from model (0.487 vs 0.493 is meaningful)
    - Enables smooth visualizations (no artificial jumps)
    - Allows video-specific thresholding (adapts to content)
    - Matches how emotion researchers store data (dimensional models)
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading VA model from {model_path}")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        ).to(self.device)

        self.model.eval()
        logger.info("VA model loaded successfully")

    def predict(self, text: str) -> Dict:
        """
        Predict valence and arousal for single text segment.

        Returns continuous values (no discretization):
        - valence: 0 (negative) to 1 (positive)
        - arousal: 0 (calm) to 1 (excited)
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.sigmoid(outputs.logits)

            valence = predictions[0, 0].item()
            arousal = predictions[0, 1].item()

        return {"valence": float(valence), "arousal": float(arousal)}

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Batch prediction for efficiency.

        Processing in batches significantly faster than one-by-one:
        - Single: ~0.05s per segment
        - Batch 32: ~0.002s per segment (25x speedup)
        """
        if not texts:
            return []

        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.sigmoid(outputs.logits)

            for j in range(len(batch_texts)):
                valence = predictions[j, 0].item()
                arousal = predictions[j, 1].item()

                results.append({"valence": float(valence), "arousal": float(arousal)})

        return results


def calculate_emotion_threshold(arousal_values: List[float], method: str = "median") -> float:
    """
    Calculate video-specific threshold for emotion classification.

    Design rationale:
    - Each video has different baseline arousal
    - Documentary: conversational baseline (~0.43)
    - News: urgent baseline (~0.38)
    - Drama: higher baseline (~0.52)
    - Using video's own median adapts automatically

    Methods:
    - 'median': Middle 50% neutral, top 50% classify (default, most balanced)
    - 'mean': Average-based split (sensitive to outliers)
    - 'percentile_60': Top 40% only (more selective)
    - 'mad': Median + median absolute deviation (robust to outliers)

    Why median is default:
    - Robust to outliers (extreme arousal values)
    - Clear semantics: half below, half above
    - Works well across different content types
    - Research-backed (common in affective computing)

    Args:
        arousal_values: List of arousal predictions (0-1)
        method: Threshold calculation method

    Returns:
        Threshold value (0-1 scale)
    """
    if not arousal_values:
        return 0.5  # Fallback if no data

    arr = np.array(arousal_values)

    if method == "median":
        # Most balanced: 50/50 split
        return float(np.median(arr))

    elif method == "mean":
        # Average-based: sensitive to outliers
        return float(np.mean(arr))

    elif method == "percentile_60":
        # More selective: top 40% only
        # Useful for content with high neutral proportion
        return float(np.percentile(arr, 60))

    elif method == "mad":
        # Median Absolute Deviation: very robust
        # Threshold = median + 1 MAD (catches deviations)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return float(median + mad)

    else:
        logger.warning(f"Unknown method '{method}', using median")
        return float(np.median(arr))


def apply_emotion_significance_filter(predictions: List[Dict], threshold: float) -> List[Dict]:
    """
    Flag segments as emotionally significant based on arousal threshold.

    Design choice: Binary filter (significant Y/N) vs multi-level intensity

    Binary approach benefits:
    - Clear semantics: "Should we classify emotion?" Yes/No
    - No arbitrary levels (what's difference between intensity 3 vs 4?)
    - Adapts to content (threshold is video-specific)
    - Simpler logic (one decision, not five)
    - Matches filtering use case (Stage 6B needs this decision)

    Multi-level intensity problems:
    - Arbitrary boundaries (why 0.4 not 0.39?)
    - Information loss (0.48 and 0.52 both become "3")
    - Doesn't adapt to content baseline
    - Stepped visualizations (not smooth)

    Args:
        predictions: List of prediction dicts with arousal/valence
        threshold: Video-specific arousal threshold

    Returns:
        Predictions with added 'is_emotionally_significant' flag
    """
    for pred in predictions:
        if pred.get("has_prediction", False):
            # Binary decision: arousal above threshold?
            pred["is_emotionally_significant"] = pred["arousal"] > threshold
            pred["threshold_used"] = threshold

            # Also store percentile rank (useful for analysis)
            # This helps understand "how significant" without discrete bins
            all_arousal = [p["arousal"] for p in predictions if p.get("has_prediction")]
            percentile = np.sum(np.array(all_arousal) < pred["arousal"]) / len(all_arousal) * 100
            pred["arousal_percentile"] = round(percentile, 1)
        else:
            # Segment without prediction (no speech, etc.)
            pred["is_emotionally_significant"] = False
            pred["threshold_used"] = None
            pred["arousal_percentile"] = None

    return predictions


def generate_arousal_filename(output_dir: Path, model_size: str, segments_base: str) -> Path:
    """
    Generate filename for arousal predictions.

    Pattern: arousal_{model}_local_{segments_base}.json
    Note: Changed from 'intensity' to 'arousal' to reflect continuous output
    """
    filename = f"arousal_{model_size}_local_{segments_base}.json"
    return output_dir / filename


def check_if_processed(video_dir: Path, local_segments_name: str, model_size: str) -> bool:
    """
    Check if arousal predictions already exist for these local segments.

    Changed filename pattern from 'intensity_*.json' to 'arousal_*.json'
    to reflect new continuous output format.
    """
    video_dir = Path(video_dir)

    # Extract base name from local segments filename
    if local_segments_name.startswith("local_segments_"):
        segments_base = local_segments_name.replace("local_segments_", "").replace(".json", "")
    else:
        segments_base = local_segments_name.replace(".json", "")

    # Check for arousal file
    output_path = generate_arousal_filename(video_dir, model_size, segments_base)

    if not output_path.exists():
        return False

    # Verify file has valid content
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check for new format indicators
        metadata = data.get("metadata", {})
        has_continuous_output = metadata.get("output_type") == "continuous_arousal_valence"

        if "predictions" in data and has_continuous_output:
            return True
        else:
            logger.warning(f"Found old format file: {output_path.name}, will reprocess")
            return False

    except Exception as e:
        logger.warning(f"Could not read {output_path.name}: {e}")
        return False


def classify_arousal_valence_from_local_segments(
    video_dir: Path, local_segments_name: str, model_path: str, model_size: str, config: Dict = None
) -> Dict:
    """
    Classify arousal and valence for local segments using continuous output.

    KEY CHANGES from old intensity approach:
    1. Keep arousal/valence continuous (no conversion to 1-5 scale)
    2. Calculate video-specific threshold (adapts to content)
    3. Binary filter: is_emotionally_significant (for Stage 6B filtering)
    4. Store percentile ranks (for analysis without discrete bins)
    5. Updated statistics (arousal distribution, not intensity counts)

    Why continuous output is better:
    - Preserves model's full information (0.487 is different from 0.493)
    - Enables smooth timeline visualizations (no artificial jumps)
    - Adapts to each video's baseline (documentary vs drama)
    - Simpler logic (no arbitrary threshold decisions)
    - Matches academic best practices in emotion research

    Args:
        video_dir: Path to video-N directory
        local_segments_name: Name of local segments file
        model_path: Path to VA model directory
        model_size: Model identifier for output filename
        config: Optional configuration parameters

    Returns:
        Dictionary with arousal/valence predictions
    """
    video_dir = Path(video_dir)
    segments_path = video_dir / local_segments_name

    if not segments_path.exists():
        raise FileNotFoundError(f"Local segments not found: {segments_path}")

    logger.info(f"Processing: {video_dir.name}/{local_segments_name}")

    # Load local segments
    with open(segments_path, "r", encoding="utf-8") as f:
        segments_data = json.load(f)

    segments = segments_data.get("segments", [])

    if not segments:
        logger.warning(f"No segments found in {local_segments_name}")
        return {
            "metadata": {
                "model": model_size,
                "output_type": "continuous_arousal_valence",
                "local_segments_file": local_segments_name,
                "processing_date": datetime.utcnow().isoformat() + "Z",
            },
            "predictions": [],
            "statistics": {"total_segments": 0, "classified_segments": 0},
        }

    # Get configuration
    config = config or {}
    batch_size = config.get("batch_size", 32)
    device = config.get("device", None)
    skip_ads = config.get("skip_advertisements", True)
    threshold_method = config.get("threshold_method", "median")

    # Initialize classifier
    logger.info(f"Loading model: {model_size}")
    start_time = time.time()

    classifier = ValenceArousalClassifier(model_path, device=device)

    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Filter segments for processing
    segments_to_process = []
    segment_indices = []

    for i, segment in enumerate(segments):
        # Skip if no speech
        if not segment.get("has_speech", False):
            continue

        # Skip advertisements (configurable)
        if skip_ads and segment.get("is_advertisement", False):
            continue

        # Check text exists
        text = segment.get("text", "").strip()
        if not text:
            continue

        segments_to_process.append(segment)
        segment_indices.append(i)

    if not segments_to_process:
        logger.warning("No segments to process after filtering")
        logger.info(f"  Total segments: {len(segments)}")
        logger.info(f"  With speech: {sum(1 for s in segments if s.get('has_speech', False))}")
        logger.info(
            f"  Advertisements: {sum(1 for s in segments if s.get('is_advertisement', False))}"
        )

        return {
            "metadata": {
                "model": model_size,
                "output_type": "continuous_arousal_valence",
                "local_segments_file": local_segments_name,
                "processing_date": datetime.utcnow().isoformat() + "Z",
                "threshold_method": threshold_method,
            },
            "predictions": [],
            "statistics": {
                "total_segments": len(segments),
                "segments_with_speech": sum(1 for s in segments if s.get("has_speech", False)),
                "advertisement_segments": sum(
                    1 for s in segments if s.get("is_advertisement", False)
                ),
                "classified_segments": 0,
            },
        }

    logger.info(f"Classifying {len(segments_to_process)} segments (after filtering)...")
    logger.info(f"  Skipped ads: {skip_ads}")

    # Extract texts for batch processing
    texts = [seg["text"] for seg in segments_to_process]

    classify_start = time.time()

    # Batch predict arousal and valence
    va_predictions = classifier.predict_batch(texts, batch_size=batch_size)

    classify_time = time.time() - classify_start
    logger.info(f"Classification complete in {classify_time:.1f}s")
    logger.info(f"Speed: {len(texts) / classify_time:.1f} segments/second")

    # Build results for ALL segments (including skipped ones)
    predictions = []
    processed_idx = 0

    for i, segment in enumerate(segments):
        segment_id = segment.get("local_segment_id", i)

        # Check if this segment was processed
        if i in segment_indices:
            # Get VA prediction
            va_pred = va_predictions[processed_idx]

            predictions.append(
                {
                    "segment_id": segment_id,
                    "local_segment_id": segment_id,
                    "global_scene_id": segment.get("global_scene_id"),
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "text": segment["text"],
                    "sentence_count": segment.get("sentence_count", 0),
                    "valence": va_pred["valence"],
                    "arousal": va_pred["arousal"],
                    "has_prediction": True,
                    "is_advertisement": segment.get("is_advertisement", False),
                }
            )

            processed_idx += 1
        else:
            # Segment was skipped (no speech or is ad)
            skip_reason = []
            if not segment.get("has_speech", False):
                skip_reason.append("no_speech")
            if skip_ads and segment.get("is_advertisement", False):
                skip_reason.append("advertisement")
            if not segment.get("text", "").strip():
                skip_reason.append("no_text")

            predictions.append(
                {
                    "segment_id": segment_id,
                    "local_segment_id": segment_id,
                    "global_scene_id": segment.get("global_scene_id"),
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "text": segment.get("text", ""),
                    "sentence_count": segment.get("sentence_count", 0),
                    "valence": None,
                    "arousal": None,
                    "has_prediction": False,
                    "reason": "_".join(skip_reason) if skip_reason else "unknown",
                    "is_advertisement": segment.get("is_advertisement", False),
                }
            )

    # Calculate video-specific emotion threshold
    classified_arousal = [p["arousal"] for p in predictions if p["has_prediction"]]

    if classified_arousal:
        threshold = calculate_emotion_threshold(classified_arousal, method=threshold_method)
        logger.info(f"Calculated emotion threshold: {threshold:.3f} (method: {threshold_method})")
    else:
        threshold = 0.5  # Fallback
        logger.warning("No classified segments, using default threshold 0.5")

    # Apply binary filter for emotion classification
    predictions = apply_emotion_significance_filter(predictions, threshold)

    # Calculate statistics (continuous distribution, not discrete bins)
    classified = [p for p in predictions if p["has_prediction"]]

    arousal_values = [p["arousal"] for p in classified]
    valence_values = [p["valence"] for p in classified]

    emotionally_significant_count = sum(1 for p in classified if p["is_emotionally_significant"])

    statistics = {
        "total_segments": len(segments),
        "segments_with_speech": sum(1 for s in segments if s.get("has_speech", False)),
        "advertisement_segments": sum(1 for s in segments if s.get("is_advertisement", False)),
        "classified_segments": len(classified),
        "skipped_segments": len(segments) - len(classified),
        # Arousal distribution (continuous)
        "arousal": {
            "mean": float(np.mean(arousal_values)) if arousal_values else 0.0,
            "median": float(np.median(arousal_values)) if arousal_values else 0.0,
            "std": float(np.std(arousal_values)) if arousal_values else 0.0,
            "min": float(np.min(arousal_values)) if arousal_values else 0.0,
            "max": float(np.max(arousal_values)) if arousal_values else 0.0,
            "percentiles": {
                "10": float(np.percentile(arousal_values, 10)) if arousal_values else 0.0,
                "25": float(np.percentile(arousal_values, 25)) if arousal_values else 0.0,
                "50": float(np.percentile(arousal_values, 50)) if arousal_values else 0.0,
                "75": float(np.percentile(arousal_values, 75)) if arousal_values else 0.0,
                "90": float(np.percentile(arousal_values, 90)) if arousal_values else 0.0,
            },
        },
        # Valence distribution (continuous)
        "valence": {
            "mean": float(np.mean(valence_values)) if valence_values else 0.0,
            "median": float(np.median(valence_values)) if valence_values else 0.0,
            "std": float(np.std(valence_values)) if valence_values else 0.0,
            "min": float(np.min(valence_values)) if valence_values else 0.0,
            "max": float(np.max(valence_values)) if valence_values else 0.0,
        },
        # Emotion filtering statistics
        "emotion_threshold": {
            "value": float(threshold),
            "method": threshold_method,
            "emotionally_significant_count": emotionally_significant_count,
            "emotionally_significant_percentage": round(
                emotionally_significant_count / len(classified) * 100, 1
            )
            if classified
            else 0.0,
        },
        # Arousal-Valence correlation
        "arousal_valence_correlation": float(np.corrcoef(arousal_values, valence_values)[0, 1])
        if len(arousal_values) > 1
        else 0.0,
    }

    # Build output
    output = {
        "metadata": {
            "model": model_size,
            "model_path": str(model_path),
            "output_type": "continuous_arousal_valence",
            "local_segments_file": local_segments_name,
            "input_type": "local_segments",
            "processing_date": datetime.utcnow().isoformat() + "Z",
            "device_used": classifier.device,
            "processing_time_seconds": classify_time,
            "config": {
                "batch_size": batch_size,
                "skip_advertisements": skip_ads,
                "threshold_method": threshold_method,
            },
        },
        "predictions": predictions,
        "statistics": statistics,
    }

    # Save results with new filename pattern
    if local_segments_name.startswith("local_segments_"):
        segments_base = local_segments_name.replace("local_segments_", "").replace(".json", "")
    else:
        segments_base = local_segments_name.replace(".json", "")

    output_path = generate_arousal_filename(video_dir, model_size, segments_base)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved arousal predictions to {output_path.name}")
    logger.info("\nStatistics:")
    logger.info(f"  Classified segments: {statistics['classified_segments']}")
    logger.info(
        f"  Arousal: mean={statistics['arousal']['mean']:.3f}, "
        f"median={statistics['arousal']['median']:.3f}, "
        f"std={statistics['arousal']['std']:.3f}"
    )
    logger.info(
        f"  Valence: mean={statistics['valence']['mean']:.3f}, "
        f"median={statistics['valence']['median']:.3f}"
    )
    logger.info(f"  Emotion threshold: {statistics['emotion_threshold']['value']:.3f}")
    logger.info(
        f"  Emotionally significant: {statistics['emotion_threshold']['emotionally_significant_count']} "
        f"({statistics['emotion_threshold']['emotionally_significant_percentage']:.1f}%)"
    )

    return output


def process_video(video_dir: Path, model_config: Dict, force_reprocess: bool = False) -> List[Dict]:
    """
    Process all local segments files in a video directory.

    CRITICAL: Only processes ORIGINAL Russian segments files.
    Translated files (*_translated.json) are handled by English classifier (6B-EN).

    Args:
        video_dir: Path to video-N directory
        model_config: Model configuration dict
        force_reprocess: If True, reprocess even if already done

    Returns:
        List of result dicts (one per local segments file)
    """
    video_dir = Path(video_dir)

    # Find all local segments files
    segments_files = sorted(video_dir.glob("local_segments_*.json"))

    # CRITICAL: Exclude translated files - those are for English classifier (6B-EN)
    # This prevents duplicate processing and ensures clean separation:
    # - Russian classifier (6A-RU): processes local_segments_*.json
    # - English classifier (6B-EN): processes local_segments_*_translated.json
    segments_files = [f for f in segments_files if not f.name.endswith("_translated.json")]

    if not segments_files:
        logger.warning(f"No local segments files found in {video_dir.name}")
        logger.info("Run scene_align.py (Stage 5) first to create local segments")
        return []

    logger.info(f"Found {len(segments_files)} local segments file(s) in {video_dir.name}")

    results = []

    for segments_file in segments_files:
        segments_name = segments_file.name
        model_size = model_config["name"]

        # Check if already processed
        if not force_reprocess and check_if_processed(video_dir, segments_name, model_size):
            logger.info(f"Already processed: {segments_name} with {model_size} - SKIPPING")
            logger.info("  (Use force_reprocess=True to reprocess)")

            # Load existing results
            if segments_name.startswith("local_segments_"):
                segments_base = segments_name.replace("local_segments_", "").replace(".json", "")
            else:
                segments_base = segments_name.replace(".json", "")

            output_path = generate_arousal_filename(video_dir, model_size, segments_base)
            with open(output_path, "r", encoding="utf-8") as f:
                existing_result = json.load(f)

            results.append(
                {
                    "status": "skipped",
                    "segments_file": segments_name,
                    "model": model_size,
                    "result": existing_result,
                }
            )
            continue

        # Process local segments
        try:
            result = classify_arousal_valence_from_local_segments(
                video_dir=video_dir,
                local_segments_name=segments_name,
                model_path=model_config["path"],
                model_size=model_size,
                config=model_config.get("config", {}),
            )

            results.append(
                {
                    "status": "success",
                    "segments_file": segments_name,
                    "model": model_size,
                    "result": result,
                }
            )

        except Exception as e:
            logger.error(f"Failed to process {segments_name}: {e}")
            import traceback

            traceback.print_exc()

            results.append(
                {
                    "status": "failed",
                    "segments_file": segments_name,
                    "model": model_size,
                    "error": str(e),
                }
            )

    return results


if __name__ == "__main__":
    """
    Test arousal-valence classification on local segments.
    
    NEW APPROACH:
    - Continuous arousal/valence (no discretization)
    - Video-specific threshold (adapts to content)
    - Binary filter (emotionally significant Y/N)
    - Better for visualization and emotion classification
    """
    import sys

    downloads_dir = Path("downloads")

    if not downloads_dir.exists():
        print("Downloads directory not found. Run downloader.py first.")
        sys.exit(1)

    video_dirs = sorted(downloads_dir.glob("video-*"))

    if not video_dirs:
        print("No video directories found in downloads/")
        sys.exit(1)

    print(f"Found {len(video_dirs)} video(s) to process")
    print("=" * 70)

    # Model configurations (same models, new output format)
    MODEL_CONFIGS = {
        "distilbert": {
            "name": "distilbert",
            "path": "models/distilbert-va",
            "description": "Fastest, 134M parameters",
            "config": {
                "batch_size": 64,
                "device": None,
                "skip_advertisements": True,
                "threshold_method": "median",  # NEW: threshold calculation method
            },
        },
        "xlmroberta-base": {
            "name": "xlmroberta-base",
            "path": "models/xlmroberta-base-va",
            "description": "Balanced, 270M parameters",
            "config": {
                "batch_size": 32,
                "device": None,
                "skip_advertisements": True,
                "threshold_method": "median",
            },
        },
        "xlmroberta-large": {
            "name": "xlmroberta-large",
            "path": "models/xlmroberta-large-va",
            "description": "Best accuracy, 550M parameters",
            "config": {
                "batch_size": 16,
                "device": None,
                "skip_advertisements": True,
                "threshold_method": "median",
            },
        },
    }

    # Choose which model
    active_models = ["xlmroberta-large"]

    print("Available models:")
    for name, cfg in MODEL_CONFIGS.items():
        status = "ACTIVE" if name in active_models else "inactive"
        print(f"  [{name}]: {cfg['description']} ({status})")

    print("\n" + "=" * 70)
    print("NEW APPROACH: Continuous Arousal + Binary Filtering")
    print("=" * 70)
    print("Changes from old intensity classifier:")
    print("1. Keep arousal/valence continuous (no 1-5 scale)")
    print("2. Video-specific threshold (adapts to content baseline)")
    print("3. Binary filter: emotionally_significant (Y/N)")
    print("4. Output filename: arousal_*.json (was intensity_*.json)")
    print("5. Better for smooth visualizations")
    print("=" * 70)

    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nGPU detected: {gpu_name}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("\nNo GPU detected - using CPU (will be slower)")

    print("=" * 70)

    force_reprocess = False

    # Process each video directory
    for video_dir in video_dirs:
        print(f"\n{'=' * 70}")
        print(f"Processing: {video_dir.name}")
        print("=" * 70)

        # Check if local segments exist
        local_files = list(video_dir.glob("local_segments_*.json"))
        if not local_files:
            print(f"\nNo local segments found in {video_dir.name}")
            print("Run scene_align.py (Stage 5) first")
            continue

        print(f"\nFound {len(local_files)} local segments file(s):")
        for lf in local_files:
            print(f"  - {lf.name}")

        # Process with each active model
        for model_name in active_models:
            if model_name not in MODEL_CONFIGS:
                print(f"Unknown model: {model_name}, skipping")
                continue

            model_config = MODEL_CONFIGS[model_name]

            # Check if model exists
            model_path = Path(model_config["path"])
            if not model_path.exists():
                print(f"\nModel not found: {model_path}")
                print("Download from: https://github.com/gmendes9/multilingual_va_prediction")
                continue

            print(f"\n--- Using model: {model_name} ---")
            print(f"Description: {model_config['description']}")
            print(f"Threshold method: {model_config['config']['threshold_method']}")

            try:
                start_time = time.time()

                results = process_video(
                    video_dir=video_dir, model_config=model_config, force_reprocess=force_reprocess
                )

                elapsed = time.time() - start_time

                # Display summary
                successful = sum(1 for r in results if r["status"] == "success")
                skipped = sum(1 for r in results if r["status"] == "skipped")
                failed = sum(1 for r in results if r["status"] == "failed")

                print(f"\nResults for {model_name}:")
                print(f"  Total files: {len(results)}")
                print(f"  Processed: {successful}")
                print(f"  Skipped: {skipped}")
                print(f"  Failed: {failed}")
                print(f"  Total time: {elapsed:.1f}s")

                # Show details
                for result in results:
                    if result["status"] in ["success", "skipped"]:
                        stats = result["result"]["statistics"]
                        threshold_stats = stats["emotion_threshold"]

                        print(f"\n  {result['segments_file']}:")
                        print(f"    Total segments: {stats['total_segments']}")
                        print(f"    Classified: {stats['classified_segments']}")
                        print(f"    Skipped: {stats['skipped_segments']}")

                        if stats.get("advertisement_segments", 0) > 0:
                            print(
                                f"    Advertisements (skipped): {stats['advertisement_segments']}"
                            )

                        if stats["classified_segments"] > 0:
                            print(
                                f"    Arousal: mean={stats['arousal']['mean']:.3f}, "
                                f"median={stats['arousal']['median']:.3f}, "
                                f"std={stats['arousal']['std']:.3f}"
                            )
                            print(
                                f"    Valence: mean={stats['valence']['mean']:.3f}, "
                                f"median={stats['valence']['median']:.3f}"
                            )
                            print(f"    Emotion threshold: {threshold_stats['value']:.3f}")
                            print(
                                f"    Emotionally significant: {threshold_stats['emotionally_significant_count']} "
                                f"({threshold_stats['emotionally_significant_percentage']:.1f}%)"
                            )

            except Exception as e:
                print(f"Error processing with {model_name}: {e}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 70)
    print("AROUSAL-VALENCE CLASSIFICATION COMPLETE")
    print("=" * 70)
    print("\nKey improvements over discrete intensity approach:")
    print("1. Continuous arousal (preserves all information)")
    print("2. Video-specific threshold (adapts to content)")
    print("3. Binary emotion filter (simpler than 5-level scale)")
    print("4. Better for smooth timeline visualizations")
    print("5. Matches academic best practices")
    print("\nOutput files: arousal_{model}_local_*.json")
    print("Next step: Use 'is_emotionally_significant' flag in Stage 6B")
