"""
Emotion Classification Module for Russian Text (Stage 6C)

Classifies emotions in Russian speech segments using ruBERT-based models.
Only processes segments marked as emotionally significant by intensity classifier.

Design rationale:
- Two-stage approach: Intensity filtering (Stage 6A) → Emotion classification (Stage 6C)
- Neutral assignment without inference for low-arousal segments (efficient)
- Batch processing for speed (processes multiple segments at once)
- Modular architecture supports multiple emotion models for cross-validation
- Each model produces separate output file for independent evaluation

Integration with pipeline:
- Input 1: local_segments_*.json (text content)
- Input 2: arousal_*_local_*.json (intensity + significance flags)
- Output: emotion_{model}_local_*.json (emotion predictions)
- Next stage: Cross-validation between multiple emotion models (future)

Model notes:
- Djacon/rubert-tiny2-russian-emotion-detection: 7-class emotion classifier
- Classes: neutral, joy, sadness, anger, fear, surprise, disgust (typical ruBERT setup)
- Native Russian processing (no translation needed)
- ~30M parameters, fast inference on GPU

Alternative approaches considered:
- Single-stage emotion classification: Slower, processes all segments including neutral
- Context-window classification: More complex, marginal benefit for short segments
- Multi-model ensemble in same module: Less modular, harder to debug
- Current approach: Simple, efficient, supports independent model evaluation
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


# =============================================================================
# FILE DISCOVERY
# =============================================================================


def find_matching_arousal_file(
    video_dir: Path, segments_filename: str, preferred_model: str = "xlmroberta-large"
) -> Optional[Path]:
    """
    Find arousal predictions file matching the local segments file.

    Why we need arousal file:
    - Contains is_emotionally_significant flags (intensity filtering)
    - Provides threshold values for validation
    - Ensures we process same segments that passed intensity check

    Matching logic:
    - Extract base name from segments file (e.g., "transcription_large-v3")
    - Look for arousal_{model}_local_{base}.json
    - Try preferred model first, then any available model

    Args:
        video_dir: Video directory path
        segments_filename: Name of local segments file
        preferred_model: Preferred intensity model (default: 'xlmroberta-large')

    Returns:
        Path to arousal file or None if not found
    """
    video_dir = Path(video_dir)

    # Extract base name from segments filename
    # Input: local_segments_transcription_large-v3.json
    # Output: transcription_large-v3
    if segments_filename.startswith("local_segments_"):
        segments_base = segments_filename.replace("local_segments_", "").replace(".json", "")
    else:
        segments_base = segments_filename.replace(".json", "")

    # Try preferred model first
    preferred_arousal = video_dir / f"arousal_{preferred_model}_local_{segments_base}.json"
    if preferred_arousal.exists():
        logger.info(f"Found preferred arousal file: {preferred_arousal.name}")
        return preferred_arousal

    # Search for any arousal file matching this segments base
    arousal_pattern = f"arousal_*_local_{segments_base}.json"
    arousal_files = list(video_dir.glob(arousal_pattern))

    if arousal_files:
        # Use first available
        arousal_file = arousal_files[0]
        logger.info(f"Found arousal file: {arousal_file.name}")
        return arousal_file

    logger.warning(
        f"No arousal predictions found for {segments_filename}\n"
        f"  Expected pattern: arousal_*_local_{segments_base}.json\n"
        f"  Run intensity classifier (Stage 6A) first"
    )
    return None


# =============================================================================
# EMOTION CLASSIFICATION
# =============================================================================


class EmotionClassifier:
    """
    Russian emotion classifier using ruBERT-based models.

    Model architecture:
    - Base: ruBERT (Russian BERT) pre-trained on Russian text
    - Fine-tuned on emotion classification task
    - Output: Probabilities for each emotion class

    Why ruBERT over multilingual models:
    - Better Russian understanding (trained specifically on Russian)
    - More accurate emotion detection for Russian text
    - Smaller model size (30-100M params vs 270-550M for XLM-R)
    - Faster inference for same quality
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading emotion model: {model_path}")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Get emotion labels from model config
        # Most ruBERT emotion models use id2label mapping
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

        logger.info("Model loaded successfully")
        logger.info(f"Emotion classes: {list(self.id2label.values())}")

    def predict(self, text: str) -> Dict:
        """
        Predict emotion for single text segment.

        Returns:
            Dictionary with predicted_emotion, confidence, and probabilities
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get predicted class
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            predicted_emotion = self.id2label[predicted_id]
            confidence = probabilities[0, predicted_id].item()

            # Convert probabilities to dict
            probs_dict = {
                self.id2label[i]: float(probabilities[0, i]) for i in range(len(self.id2label))
            }

        return {
            "predicted_emotion": predicted_emotion,
            "confidence": float(confidence),
            "probabilities": probs_dict,
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Batch prediction for efficiency.

        Why batch processing:
        - Single: ~0.03-0.05s per segment (sequential)
        - Batch 32: ~0.001-0.002s per segment (parallel)
        - ~20-30x speedup with batching

        Args:
            texts: List of text segments
            batch_size: Number of segments to process at once

        Returns:
            List of prediction dictionaries
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
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Process each item in batch
            for j in range(len(batch_texts)):
                predicted_id = torch.argmax(probabilities[j]).item()
                predicted_emotion = self.id2label[predicted_id]
                confidence = probabilities[j, predicted_id].item()

                probs_dict = {
                    self.id2label[k]: float(probabilities[j, k]) for k in range(len(self.id2label))
                }

                results.append(
                    {
                        "predicted_emotion": predicted_emotion,
                        "confidence": float(confidence),
                        "probabilities": probs_dict,
                    }
                )

        return results


# =============================================================================
# FILE GENERATION
# =============================================================================


def generate_emotion_filename(output_dir: Path, model_name: str, segments_base: str) -> Path:
    """
    Generate filename for emotion predictions.

    Pattern: emotion_{model}_local_{segments_base}.json

    Why this naming:
    - Consistent with arousal_{model}_local_*.json pattern
    - Model name allows multiple emotion models to coexist
    - segments_base links to specific transcription
    - Enables cross-validation between different emotion models

    Examples:
    - emotion_rubert-tiny2_local_transcription_large-v3.json
    - emotion_sberbank-rubert_local_transcription_large-v3.json
    """
    filename = f"emotion_{model_name}_local_{segments_base}.json"
    return output_dir / filename


def check_if_processed(
    video_dir: Path, segments_filename: str, model_name: str, arousal_file: Path
) -> bool:
    """
    Check if emotion classification already completed with matching configuration.

    Validation checks:
    - Output file exists
    - Model name matches
    - Input segments file matches
    - Arousal threshold matches (ensures same filtering)

    Why check arousal threshold:
    - Different thresholds = different segments classified
    - Need to reprocess if intensity filtering changed
    - Ensures consistency across pipeline stages

    Args:
        video_dir: Video directory
        segments_filename: Local segments filename
        model_name: Emotion model identifier
        arousal_file: Path to arousal predictions file

    Returns:
        True if already processed with matching config, False otherwise
    """
    video_dir = Path(video_dir)

    # Extract segments base name
    if segments_filename.startswith("local_segments_"):
        segments_base = segments_filename.replace("local_segments_", "").replace(".json", "")
    else:
        segments_base = segments_filename.replace(".json", "")

    # Check for emotion predictions file
    output_path = generate_emotion_filename(video_dir, model_name, segments_base)

    if not output_path.exists():
        return False

    # Validate configuration match
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

        metadata = existing.get("metadata", {})

        # Check critical parameters
        stored_model = metadata.get("model_name")
        stored_segments = metadata.get("local_segments_file")
        stored_arousal = metadata.get("arousal_file")

        # Load arousal threshold for comparison
        with open(arousal_file, "r", encoding="utf-8") as f:
            arousal_data = json.load(f)

        current_threshold = (
            arousal_data.get("statistics", {}).get("emotion_threshold", {}).get("value")
        )
        stored_threshold = metadata.get("arousal_threshold")

        # Configuration matches if all critical parameters identical
        config_matches = (
            stored_model == model_name
            and stored_segments == segments_filename
            and stored_arousal == arousal_file.name
            and abs(current_threshold - stored_threshold) < 0.001  # Float comparison tolerance
        )

        if config_matches:
            return True
        else:
            logger.info("Existing emotion predictions found but config differs")
            logger.info(f"  Stored: model={stored_model}, threshold={stored_threshold}")
            logger.info(f"  Current: model={model_name}, threshold={current_threshold}")
            return False

    except Exception as e:
        logger.warning(f"Could not read existing emotion file: {e}")
        return False


# =============================================================================
# MAIN PROCESSING
# =============================================================================


def classify_emotions_from_local_segments(
    video_dir: Path, segments_filename: str, model_path: str, model_name: str, config: Dict = None
) -> Dict:
    """
    Classify emotions for local segments using intensity-based filtering.

    Processing workflow:
    1. Load local segments (text content)
    2. Load arousal predictions (intensity + significance flags)
    3. Match segments between files
    4. Classify ONLY emotionally significant segments
    5. Assign "neutral" to low-arousal segments without model inference

    Why two-stage approach (intensity → emotion):
    - Efficiency: Skip emotion model on 40-60% of segments (neutral content)
    - Quality: Focus emotion model on expressive speech
    - Pipeline design: Intensity is continuous dimension, emotion is categorical
    - Research-backed: Arousal correlates with emotional expressiveness

    Args:
        video_dir: Path to video directory
        segments_filename: Name of local segments file
        model_path: Path to emotion model (HuggingFace model name or local path)
        model_name: Model identifier for output filename
        config: Optional configuration parameters

    Returns:
        Dictionary with emotion predictions and statistics
    """
    video_dir = Path(video_dir)
    segments_path = video_dir / segments_filename

    if not segments_path.exists():
        raise FileNotFoundError(f"Segments file not found: {segments_path}")

    logger.info(f"Processing: {video_dir.name}/{segments_filename}")

    # Find matching arousal predictions
    arousal_file = find_matching_arousal_file(video_dir, segments_filename)

    if arousal_file is None:
        raise FileNotFoundError(
            f"No arousal predictions found for {segments_filename}\n"
            "Run intensity classifier (Stage 6A) first"
        )

    # Load segments and arousal data
    with open(segments_path, "r", encoding="utf-8") as f:
        segments_data = json.load(f)

    with open(arousal_file, "r", encoding="utf-8") as f:
        arousal_data = json.load(f)

    segments = segments_data.get("segments", [])
    arousal_predictions = arousal_data.get("predictions", [])

    if not segments:
        logger.warning("No segments found")
        return _build_empty_output(model_name, segments_filename, arousal_file.name, segments_data)

    if not arousal_predictions:
        raise ValueError(f"No arousal predictions found in {arousal_file.name}")

    # Get configuration
    config = config or {}
    batch_size = config.get("batch_size", 32)
    device = config.get("device", None)

    # Initialize emotion classifier
    logger.info(f"Loading emotion model: {model_name}")
    start_time = time.time()

    classifier = EmotionClassifier(model_path, device=device)

    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Match segments with arousal predictions
    # Arousal file should have same segment_ids as segments file
    arousal_by_id = {pred["segment_id"]: pred for pred in arousal_predictions}

    # Collect segments for classification
    segments_to_classify = []
    classify_indices = []
    neutral_count = 0

    for i, segment in enumerate(segments):
        segment_id = segment.get("local_segment_id", i)

        # Get corresponding arousal prediction
        arousal_pred = arousal_by_id.get(segment_id)

        if arousal_pred is None:
            logger.warning(f"No arousal prediction for segment {segment_id}")
            continue

        # Check if emotionally significant
        is_significant = arousal_pred.get("is_emotionally_significant", False)

        if is_significant and segment.get("has_speech", False):
            # Classify this segment
            text = segment.get("text", "").strip()
            if text:
                segments_to_classify.append(segment)
                classify_indices.append(i)
        else:
            # Will be assigned neutral without classification
            neutral_count += 1

    logger.info("Segments breakdown:")
    logger.info(f"  Total: {len(segments)}")
    logger.info(f"  Emotionally significant (will classify): {len(segments_to_classify)}")
    logger.info(f"  Low arousal (assign neutral): {neutral_count}")

    if not segments_to_classify:
        logger.warning("No emotionally significant segments to classify")
        # All segments are neutral
        return _build_all_neutral_output(
            model_name, segments_filename, arousal_file.name, segments, arousal_data, classifier
        )

    # Extract texts for batch processing
    texts_to_classify = [seg["text"] for seg in segments_to_classify]

    # Classify emotions
    logger.info(f"Classifying {len(texts_to_classify)} emotionally significant segments...")
    classify_start = time.time()

    emotion_predictions = classifier.predict_batch(texts_to_classify, batch_size=batch_size)

    classify_time = time.time() - classify_start
    logger.info(f"Classification complete in {classify_time:.1f}s")
    logger.info(f"Speed: {len(texts_to_classify) / classify_time:.1f} segments/second")

    # Build predictions for ALL segments
    predictions = []
    classified_idx = 0

    for i, segment in enumerate(segments):
        segment_id = segment.get("local_segment_id", i)
        arousal_pred = arousal_by_id.get(segment_id)

        # Base prediction structure
        pred = {
            "segment_id": segment_id,
            "start": segment["start"],
            "end": segment["end"],
            "duration": segment["duration"],
            "text": segment.get("text", ""),
            "has_speech": segment.get("has_speech", False),
        }

        # Add arousal context
        if arousal_pred:
            pred["arousal"] = arousal_pred.get("arousal")
            pred["valence"] = arousal_pred.get("valence")
            pred["is_emotionally_significant"] = arousal_pred.get(
                "is_emotionally_significant", False
            )

        # Emotion classification result
        if i in classify_indices:
            # Classified segment
            emotion_result = emotion_predictions[classified_idx]

            pred["predicted_emotion"] = emotion_result["predicted_emotion"]
            pred["confidence"] = emotion_result["confidence"]
            pred["probabilities"] = emotion_result["probabilities"]
            pred["classification_method"] = "model"

            classified_idx += 1
        else:
            # Low arousal segment - assign neutral
            pred["predicted_emotion"] = "neutral"
            pred["confidence"] = 1.0  # High confidence for rule-based neutral
            pred["probabilities"] = None  # No model probabilities
            pred["classification_method"] = "intensity_filter"

        predictions.append(pred)

    # Calculate statistics
    classified_segments = [p for p in predictions if p["classification_method"] == "model"]
    neutral_segments = [p for p in predictions if p["classification_method"] == "intensity_filter"]

    # Emotion distribution (from classified segments)
    emotion_counts = {}
    for pred in classified_segments:
        emotion = pred["predicted_emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    # Add neutral from filtering
    emotion_counts["neutral"] = emotion_counts.get("neutral", 0) + len(neutral_segments)

    # Average confidence (for classified segments only)
    confidences = [p["confidence"] for p in classified_segments]
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    # Get arousal threshold used
    arousal_threshold = (
        arousal_data.get("statistics", {}).get("emotion_threshold", {}).get("value", 0.5)
    )

    # Build output
    output = {
        "metadata": {
            "model_name": model_name,
            "model_path": str(model_path),
            "emotion_classes": list(classifier.id2label.values()),
            "local_segments_file": segments_filename,
            "arousal_file": arousal_file.name,
            "arousal_threshold": arousal_threshold,
            "processing_date": datetime.utcnow().isoformat() + "Z",
            "device_used": classifier.device,
            "classification_time_seconds": classify_time,
            "config": {
                "batch_size": batch_size,
                "two_stage_filtering": True,
                "neutral_assignment": "intensity_filter",
            },
        },
        "predictions": predictions,
        "statistics": {
            "total_segments": len(segments),
            "segments_with_speech": sum(1 for s in segments if s.get("has_speech", False)),
            "emotionally_significant": len(classified_segments),
            "low_arousal_neutral": len(neutral_segments),
            "emotion_distribution": emotion_counts,
            "emotion_percentages": {
                emotion: round(count / len(predictions) * 100, 1)
                for emotion, count in emotion_counts.items()
            },
            "average_confidence_classified": round(avg_confidence, 3),
            "classification_method_breakdown": {
                "model": len(classified_segments),
                "intensity_filter": len(neutral_segments),
            },
        },
    }

    # Save results
    segments_base = segments_filename.replace("local_segments_", "").replace(".json", "")
    output_path = generate_emotion_filename(video_dir, model_name, segments_base)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved emotion predictions to {output_path.name}")
    logger.info("\nStatistics:")
    logger.info(f"  Total segments: {len(predictions)}")
    logger.info(f"  Classified by model: {len(classified_segments)}")
    logger.info(f"  Assigned neutral (low arousal): {len(neutral_segments)}")
    logger.info(f"  Average confidence: {avg_confidence:.3f}")
    logger.info("\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = count / len(predictions) * 100
        logger.info(f"    {emotion}: {count} ({pct:.1f}%)")

    return output


def _build_empty_output(
    model_name: str, segments_file: str, arousal_file: str, segments_data: Dict
) -> Dict:
    """Helper to build empty output when no segments found."""
    return {
        "metadata": {
            "model_name": model_name,
            "local_segments_file": segments_file,
            "arousal_file": arousal_file,
            "processing_date": datetime.utcnow().isoformat() + "Z",
        },
        "predictions": [],
        "statistics": {"total_segments": 0, "emotionally_significant": 0, "low_arousal_neutral": 0},
    }


def _build_all_neutral_output(
    model_name: str,
    segments_file: str,
    arousal_file: str,
    segments: List[Dict],
    arousal_data: Dict,
    classifier: EmotionClassifier,
) -> Dict:
    """Helper to build output when all segments are neutral (low arousal)."""

    arousal_by_id = {pred["segment_id"]: pred for pred in arousal_data.get("predictions", [])}

    predictions = []
    for i, segment in enumerate(segments):
        segment_id = segment.get("local_segment_id", i)
        arousal_pred = arousal_by_id.get(segment_id, {})

        predictions.append(
            {
                "segment_id": segment_id,
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["duration"],
                "text": segment.get("text", ""),
                "has_speech": segment.get("has_speech", False),
                "arousal": arousal_pred.get("arousal"),
                "valence": arousal_pred.get("valence"),
                "is_emotionally_significant": False,
                "predicted_emotion": "neutral",
                "confidence": 1.0,
                "probabilities": None,
                "classification_method": "intensity_filter",
            }
        )

    arousal_threshold = (
        arousal_data.get("statistics", {}).get("emotion_threshold", {}).get("value", 0.5)
    )

    return {
        "metadata": {
            "model_name": model_name,
            "emotion_classes": list(classifier.id2label.values()),
            "local_segments_file": segments_file,
            "arousal_file": arousal_file,
            "arousal_threshold": arousal_threshold,
            "processing_date": datetime.utcnow().isoformat() + "Z",
            "device_used": classifier.device,
            "classification_time_seconds": 0.0,
        },
        "predictions": predictions,
        "statistics": {
            "total_segments": len(segments),
            "segments_with_speech": sum(1 for s in segments if s.get("has_speech", False)),
            "emotionally_significant": 0,
            "low_arousal_neutral": len(predictions),
            "emotion_distribution": {"neutral": len(predictions)},
            "emotion_percentages": {"neutral": 100.0},
            "average_confidence_classified": 0.0,
            "classification_method_breakdown": {"model": 0, "intensity_filter": len(predictions)},
        },
    }


# =============================================================================
# VIDEO PROCESSING
# =============================================================================


def process_video(video_dir: Path, model_config: Dict, force_reprocess: bool = False) -> List[Dict]:
    """
    Process all local segments files in a video directory.

    Workflow:
    1. Find all local_segments_*.json files
    2. For each file, find matching arousal predictions
    3. Check if already processed with same config
    4. Classify emotions for emotionally significant segments

    Args:
        video_dir: Path to video directory
        model_config: Model configuration dict with:
            - name: Model identifier (for output filename)
            - path: HuggingFace model name or local path
            - config: Optional processing parameters
        force_reprocess: If True, reprocess even if already done

    Returns:
        List of result dicts (one per segments file)
    """
    video_dir = Path(video_dir)

    # Find all local segments files
    segments_files = sorted(video_dir.glob("local_segments_*.json"))

    if not segments_files:
        logger.warning(f"No local segments files found in {video_dir.name}")
        logger.info("Run scene alignment (Stage 5) first")
        return []

    logger.info(f"Found {len(segments_files)} local segments file(s) in {video_dir.name}")

    results = []

    for segments_file in segments_files:
        segments_name = segments_file.name
        model_name = model_config["name"]

        # Find matching arousal file
        try:
            arousal_file = find_matching_arousal_file(video_dir, segments_name)

            if arousal_file is None:
                logger.warning(f"Skipping {segments_name}: no arousal predictions found")
                results.append(
                    {
                        "status": "skipped",
                        "segments_file": segments_name,
                        "model": model_name,
                        "reason": "no_arousal_predictions",
                    }
                )
                continue

            # Check if already processed
            if not force_reprocess and check_if_processed(
                video_dir, segments_name, model_name, arousal_file
            ):
                logger.info(f"Already processed: {segments_name} with {model_name} - SKIPPING")
                logger.info("  (Use force_reprocess=True to reprocess)")

                # Load existing results
                segments_base = segments_name.replace("local_segments_", "").replace(".json", "")
                output_path = generate_emotion_filename(video_dir, model_name, segments_base)

                with open(output_path, "r", encoding="utf-8") as f:
                    existing_result = json.load(f)

                results.append(
                    {
                        "status": "skipped",
                        "segments_file": segments_name,
                        "model": model_name,
                        "result": existing_result,
                    }
                )
                continue

            # Process emotion classification
            result = classify_emotions_from_local_segments(
                video_dir=video_dir,
                segments_filename=segments_name,
                model_path=model_config["path"],
                model_name=model_name,
                config=model_config.get("config", {}),
            )

            results.append(
                {
                    "status": "success",
                    "segments_file": segments_name,
                    "model": model_name,
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
                    "model": model_name,
                    "error": str(e),
                }
            )

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    """
    Emotion classification for Russian speech segments.
    
    Pipeline integration:
    - Input: local_segments_*.json + arousal_*_local_*.json
    - Output: emotion_{model}_local_*.json
    - Next: Cross-validation between multiple emotion models (future)
    
    Key features:
    - Two-stage filtering: Intensity check → Emotion classification
    - Efficient: Only classify emotionally significant segments
    - Modular: Each emotion model produces separate output
    - Batch processing: Fast GPU inference
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
    # Future: Add more emotion models here for cross-validation
    MODEL_CONFIGS = {
        "rubert-tiny2": {
            "name": "rubert-tiny2",
            "path": "Djacon/rubert-tiny2-russian-emotion-detection",
            "description": "Fast Russian emotion detection, 7 classes",
            "config": {
                "batch_size": 64,  # Smaller model, can use larger batches
                "device": None,  # Auto-detect
            },
        },
        # Future models for cross-validation:
        # 'sberbank-rubert': {
        #     'name': 'sberbank-rubert',
        #     'path': 'sberbank-ai/rubert-base-cased-conversational',
        #     'description': 'Conversational emotion detection',
        #     'config': {'batch_size': 32, 'device': None}
        # }
    }

    # Active models (currently only one, will add more for cross-validation)
    active_models = ["rubert-tiny2"]

    print("Available emotion models:")
    for name, cfg in MODEL_CONFIGS.items():
        status = "ACTIVE" if name in active_models else "inactive"
        print(f"  [{name}]: {cfg['description']} ({status})")

    print("\n" + "=" * 70)
    print("EMOTION CLASSIFICATION (Stage 6C)")
    print("=" * 70)
    print("Two-stage approach:")
    print("1. Intensity filtering (Stage 6A): Identify emotionally significant segments")
    print("2. Emotion classification (Stage 6C): Classify only significant segments")
    print("3. Low-arousal segments → neutral (no model inference, efficient)")
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

        # Check prerequisites
        segments_files = list(video_dir.glob("local_segments_*.json"))
        arousal_files = list(video_dir.glob("arousal_*_local_*.json"))

        if not segments_files:
            print(f"\nNo local segments found in {video_dir.name}")
            print("Run scene alignment (Stage 5) first")
            continue

        if not arousal_files:
            print(f"\nNo arousal predictions found in {video_dir.name}")
            print("Run intensity classifier (Stage 6A) first")
            continue

        print(f"\nFound {len(segments_files)} segments file(s)")
        print(f"Found {len(arousal_files)} arousal prediction file(s)")

        # Process with each active model
        for model_name in active_models:
            if model_name not in MODEL_CONFIGS:
                print(f"Unknown model: {model_name}, skipping")
                continue

            model_config = MODEL_CONFIGS[model_name]

            print(f"\n--- Using model: {model_name} ---")
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

                print(f"\nResults for {model_name}:")
                print(f"  Total files: {len(results)}")
                print(f"  Processed: {successful}")
                print(f"  Skipped: {skipped}")
                print(f"  Failed: {failed}")
                print(f"  Total time: {elapsed:.1f}s")

                # Show detailed results
                for result in results:
                    if result["status"] in ["success", "skipped"] and "result" in result:
                        stats = result["result"]["statistics"]

                        print(f"\n  {result['segments_file']}:")
                        print(f"    Total segments: {stats['total_segments']}")
                        print(f"    Classified by model: {stats['emotionally_significant']}")
                        print(f"    Assigned neutral: {stats['low_arousal_neutral']}")

                        if stats["emotionally_significant"] > 0:
                            print(
                                f"    Average confidence: {stats['average_confidence_classified']:.3f}"
                            )

                        print("    Emotion distribution:")
                        for emotion, pct in sorted(
                            stats["emotion_percentages"].items(), key=lambda x: -x[1]
                        ):
                            count = stats["emotion_distribution"][emotion]
                            print(f"      {emotion}: {count} ({pct:.1f}%)")

            except Exception as e:
                print(f"Error processing with {model_name}: {e}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 70)
    print("EMOTION CLASSIFICATION COMPLETE")
    print("=" * 70)
    print("\nOutput files: emotion_{model}_local_*.json")
    print("\nKey features:")
    print("1. Two-stage filtering: Only classify emotionally significant segments")
    print("2. Efficient: Low-arousal segments assigned neutral without inference")
    print("3. Modular: Each model produces separate output for cross-validation")
    print("4. Reproducible: Skip logic prevents redundant processing")
    print("\nNext steps:")
    print("  1. Add more emotion models to MODEL_CONFIGS")
    print("  2. Run all models on same videos")
    print("  3. Compare predictions between models (cross-validation)")
    print("  4. Build ensemble or voting system (future Stage 7)")
