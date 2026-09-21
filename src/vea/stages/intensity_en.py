"""
English Arousal-Valence Classification Module - Continuous Scale Approach

Updated to work with the new translation module output format.

Key changes from previous version:
- Handles simplified translated file structure (fewer fields)
- Graceful fallback for missing segment IDs (uses index)
- Assumes False for missing is_advertisement field
- Better error messages for malformed input files
- Maintains backward compatibility with old format
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


class ValenceArousalClassifierEnglish:
    """
    Predict valence and arousal for English text using multilingual models.

    Same implementation as before - no changes needed here.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading VA model for English: {model_path}")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        ).to(self.device)

        self.model.eval()
        logger.info("VA model loaded successfully")

    def predict(self, text: str) -> Dict:
        """Predict valence and arousal for single English text segment."""
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
        """Batch prediction for English texts."""
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

    Same implementation as Russian classifier - adapts to content baseline.
    """
    if not arousal_values:
        return 0.5

    arr = np.array(arousal_values)

    if method == "median":
        return float(np.median(arr))
    elif method == "mean":
        return float(np.mean(arr))
    elif method == "percentile_60":
        return float(np.percentile(arr, 60))
    elif method == "mad":
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return float(median + mad)
    else:
        logger.warning(f"Unknown method '{method}', using median")
        return float(np.median(arr))


def apply_emotion_significance_filter(predictions: List[Dict], threshold: float) -> List[Dict]:
    """
    Flag segments as emotionally significant based on arousal threshold.

    Binary decision: Should we classify emotion for this segment?
    """
    for pred in predictions:
        if pred.get("has_prediction", False):
            pred["is_emotionally_significant"] = pred["arousal"] > threshold
            pred["threshold_used"] = threshold

            # Store percentile rank for analysis
            all_arousal = [p["arousal"] for p in predictions if p.get("has_prediction")]
            if all_arousal:
                percentile = (
                    np.sum(np.array(all_arousal) < pred["arousal"]) / len(all_arousal) * 100
                )
                pred["arousal_percentile"] = round(percentile, 1)
            else:
                pred["arousal_percentile"] = None
        else:
            pred["is_emotionally_significant"] = False
            pred["threshold_used"] = None
            pred["arousal_percentile"] = None

    return predictions


def generate_arousal_filename(output_dir: Path, model_size: str, segments_base: str) -> Path:
    """
    Generate filename for English arousal predictions.

    Pattern: arousal_{model}_en_local_{segments_base}.json
    """
    filename = f"arousal_{model_size}_en_local_{segments_base}.json"
    return output_dir / filename


def check_if_processed(video_dir: Path, local_segments_name: str, model_size: str) -> bool:
    """
    Check if English arousal predictions already exist.
    """
    video_dir = Path(video_dir)

    # Extract base name from translated segments filename
    # Input: local_segments_transcription_large-v3_translated.json
    # Extract: transcription_large-v3
    if "_translated.json" in local_segments_name:
        segments_base = local_segments_name.replace("local_segments_", "").replace(
            "_translated.json", ""
        )
    else:
        logger.warning(f"Unexpected filename format: {local_segments_name}")
        segments_base = local_segments_name.replace(".json", "")

    output_path = generate_arousal_filename(video_dir, model_size, segments_base)

    if not output_path.exists():
        return False

    # Verify file has valid content
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        is_english = metadata.get("language") == "en"
        has_continuous_output = metadata.get("output_type") == "continuous_arousal_valence"

        if "predictions" in data and is_english and has_continuous_output:
            return True
        else:
            logger.warning(
                f"Found file {output_path.name} but format doesn't match English continuous output"
            )
            return False

    except Exception as e:
        logger.warning(f"Could not read {output_path.name}: {e}")
        return False


def classify_arousal_valence_english(
    video_dir: Path, local_segments_name: str, model_path: str, model_size: str, config: Dict = None
) -> Dict:
    """
    Classify arousal and valence for English translations.

    Updated to handle new translation module output format:
    - Simplified segment structure (fewer fields)
    - Missing segment IDs (uses index as fallback)
    - Missing is_advertisement flag (assumes False)
    - Missing global_scene_id (uses None)

    Maintains backward compatibility with old format.
    """
    video_dir = Path(video_dir)
    segments_path = video_dir / local_segments_name

    # Verify input is a translated file
    if "_translated.json" not in local_segments_name:
        raise ValueError(
            f"Expected translated file (local_segments_*_translated.json), "
            f"got: {local_segments_name}"
        )

    if not segments_path.exists():
        raise FileNotFoundError(f"Translated segments not found: {segments_path}")

    logger.info(f"Processing English: {video_dir.name}/{local_segments_name}")

    # Load translated segments
    with open(segments_path, "r", encoding="utf-8") as f:
        segments_data = json.load(f)

    segments = segments_data.get("segments", [])

    if not segments:
        logger.warning(f"No segments found in {local_segments_name}")
        return {
            "metadata": {
                "model": model_size,
                "language": "en",
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
    logger.info(f"Loading model: {model_size} for English")
    start_time = time.time()

    classifier = ValenceArousalClassifierEnglish(model_path, device=device)

    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Filter segments for processing
    segments_to_process = []
    segment_indices = []

    for i, segment in enumerate(segments):
        # Skip if no speech
        # NOTE: New format always has has_speech, but check for safety
        if not segment.get("has_speech", True):
            continue

        # Skip advertisements (configurable)
        # NOTE: New format doesn't include is_advertisement, assume False
        if skip_ads and segment.get("is_advertisement", False):
            continue

        # Check English text exists
        # CRITICAL: Use 'text_en' field for English translation
        text_en = segment.get("text_en", "").strip()
        if not text_en:
            continue

        segments_to_process.append(segment)
        segment_indices.append(i)

    if not segments_to_process:
        logger.warning("No English segments to process after filtering")
        logger.info(f"  Total segments: {len(segments)}")
        logger.info(f"  With speech: {sum(1 for s in segments if s.get('has_speech', True))}")
        logger.info(
            f"  With English text: {sum(1 for s in segments if s.get('text_en', '').strip())}"
        )
        logger.info(
            f"  Advertisements: {sum(1 for s in segments if s.get('is_advertisement', False))}"
        )

        return {
            "metadata": {
                "model": model_size,
                "language": "en",
                "output_type": "continuous_arousal_valence",
                "local_segments_file": local_segments_name,
                "processing_date": datetime.utcnow().isoformat() + "Z",
                "threshold_method": threshold_method,
            },
            "predictions": [],
            "statistics": {
                "total_segments": len(segments),
                "segments_with_speech": sum(1 for s in segments if s.get("has_speech", True)),
                "segments_with_english": sum(1 for s in segments if s.get("text_en", "").strip()),
                "advertisement_segments": sum(
                    1 for s in segments if s.get("is_advertisement", False)
                ),
                "classified_segments": 0,
            },
        }

    logger.info(f"Classifying {len(segments_to_process)} English segments (after filtering)...")
    logger.info(f"  Skipped ads: {skip_ads}")

    # Extract English texts for batch processing
    texts_en = [seg["text_en"] for seg in segments_to_process]

    classify_start = time.time()

    # Batch predict arousal and valence for English
    va_predictions = classifier.predict_batch(texts_en, batch_size=batch_size)

    classify_time = time.time() - classify_start
    logger.info(f"English classification complete in {classify_time:.1f}s")
    logger.info(f"Speed: {len(texts_en) / classify_time:.1f} segments/second")

    # Build results for ALL segments (including skipped ones)
    predictions = []
    processed_idx = 0

    for i, segment in enumerate(segments):
        # Handle missing segment ID gracefully
        # New format doesn't include local_segment_id, use index
        segment_id = segment.get("local_segment_id", i)

        # Check if this segment was processed
        if i in segment_indices:
            # Get VA prediction
            va_pred = va_predictions[processed_idx]

            predictions.append(
                {
                    "segment_id": segment_id,
                    "local_segment_id": segment_id,
                    "global_scene_id": segment.get("global_scene_id"),  # May be None in new format
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "text_en": segment["text_en"],
                    "text_ru": segment.get("text", ""),
                    "sentence_count": segment.get("sentence_count", 0),
                    "valence": va_pred["valence"],
                    "arousal": va_pred["arousal"],
                    "has_prediction": True,
                    "is_advertisement": segment.get("is_advertisement", False),
                }
            )

            processed_idx += 1
        else:
            # Segment was skipped
            skip_reason = []
            if not segment.get("has_speech", True):
                skip_reason.append("no_speech")
            if skip_ads and segment.get("is_advertisement", False):
                skip_reason.append("advertisement")
            if not segment.get("text_en", "").strip():
                skip_reason.append("no_english_text")

            predictions.append(
                {
                    "segment_id": segment_id,
                    "local_segment_id": segment_id,
                    "global_scene_id": segment.get("global_scene_id"),
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "text_en": segment.get("text_en", ""),
                    "text_ru": segment.get("text", ""),
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
        logger.info(
            f"Calculated English emotion threshold: {threshold:.3f} (method: {threshold_method})"
        )
    else:
        threshold = 0.5
        logger.warning("No classified segments, using default threshold 0.5")

    # Apply binary filter for emotion classification
    predictions = apply_emotion_significance_filter(predictions, threshold)

    # Calculate statistics
    classified = [p for p in predictions if p["has_prediction"]]

    arousal_values = [p["arousal"] for p in classified]
    valence_values = [p["valence"] for p in classified]

    emotionally_significant_count = sum(1 for p in classified if p["is_emotionally_significant"])

    statistics = {
        "total_segments": len(segments),
        "segments_with_speech": sum(1 for s in segments if s.get("has_speech", True)),
        "segments_with_english": sum(1 for s in segments if s.get("text_en", "").strip()),
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
            "language": "en",
            "output_type": "continuous_arousal_valence",
            "local_segments_file": local_segments_name,
            "input_type": "translated_segments",
            "input_format_note": "Updated for new translation module format",
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

    # Save results with English-specific filename
    if "_translated.json" in local_segments_name:
        segments_base = local_segments_name.replace("local_segments_", "").replace(
            "_translated.json", ""
        )
    else:
        segments_base = local_segments_name.replace(".json", "")

    output_path = generate_arousal_filename(video_dir, model_size, segments_base)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved English arousal predictions to {output_path.name}")
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
    Process all translated segments files in a video directory.

    Automatically finds local_segments_*_translated.json files.
    """
    video_dir = Path(video_dir)

    # Find all translated segments files
    segments_files = sorted(video_dir.glob("local_segments_*_translated.json"))

    if not segments_files:
        logger.warning(f"No translated segments files found in {video_dir.name}")
        logger.info("Run translator.py (Stage 5B) first to create translated segments")
        return []

    logger.info(f"Found {len(segments_files)} translated segments file(s) in {video_dir.name}")

    results = []

    for segments_file in segments_files:
        segments_name = segments_file.name
        model_size = model_config["name"]

        # Check if already processed
        if not force_reprocess and check_if_processed(video_dir, segments_name, model_size):
            logger.info(f"Already processed: {segments_name} with {model_size} - SKIPPING")
            logger.info("  (Use force_reprocess=True to reprocess)")

            # Load existing results
            if "_translated.json" in segments_name:
                segments_base = segments_name.replace("local_segments_", "").replace(
                    "_translated.json", ""
                )
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

        # Process translated segments for English
        try:
            result = classify_arousal_valence_english(
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
    Classify English arousal-valence for translated segments.
    
    Updated to work with new translation module format:
    - Simplified segment structure
    - Missing segment/scene IDs handled gracefully
    - Backward compatible with old format
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

    # Model configurations
    MODEL_CONFIGS = {
        "distilbert": {
            "name": "distilbert",
            "path": "models/distilbert-va",
            "description": "Fastest, 134M parameters",
            "config": {
                "batch_size": 64,
                "device": None,
                "skip_advertisements": True,
                "threshold_method": "median",
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
    print("ENGLISH INTENSITY CLASSIFICATION (UPDATED)")
    print("=" * 70)
    print("Updates:")
    print("1. Works with new translation module format")
    print("2. Handles simplified segment structure")
    print("3. Graceful fallback for missing IDs")
    print("4. Backward compatible with old format")
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

        # Check if translated segments exist
        translated_files = list(video_dir.glob("local_segments_*_translated.json"))
        if not translated_files:
            print(f"\nNo translated segments found in {video_dir.name}")
            print("Run translator.py (Stage 5B) first")
            continue

        print(f"\nFound {len(translated_files)} translated segments file(s):")
        for tf in translated_files:
            print(f"  - {tf.name}")

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

            print(f"\n--- Using model: {model_name} for English ---")
            print(f"Description: {model_config['description']}")

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

                print(f"\nResults for {model_name} (English):")
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
                        print(f"    With English: {stats['segments_with_english']}")
                        print(f"    Classified: {stats['classified_segments']}")

                        if stats["classified_segments"] > 0:
                            print(
                                f"    Arousal: mean={stats['arousal']['mean']:.3f}, "
                                f"median={stats['arousal']['median']:.3f}"
                            )
                            print(
                                f"    Valence: mean={stats['valence']['mean']:.3f}, "
                                f"median={stats['valence']['median']:.3f}"
                            )
                            print(
                                f"    Threshold: {threshold_stats['value']:.3f}, "
                                f"Significant: {threshold_stats['emotionally_significant_count']} "
                                f"({threshold_stats['emotionally_significant_percentage']:.1f}%)"
                            )

            except Exception as e:
                print(f"Error processing with {model_name}: {e}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print("\nOutput files: arousal_{model}_en_local_*.json")
    print("Next steps:")
    print("  - Compare with Russian predictions")
    print("  - Use for bilingual emotion ensemble")
    print("  - Analyze translation effects on emotion detection")
