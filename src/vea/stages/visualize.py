"""
Enhanced Emotion Timeline Visualization with Multi-Model Ensemble

Improvements over previous version:
- Ensemble cross-validation across 3 models (2 English + 1 Russian)
- Automatic quality analysis and error detection
- Preserved two-panel layout (arousal + valence)
- Subtle agreement indicators via opacity
- Less aggressive smoothing (preserves local arousal trends)

Design rationale:
- Multi-model consensus improves accuracy and reliability
- Two-panel layout provides comprehensive emotional context
- Arousal shows intensity, valence shows positive/negative direction
- Agreement indicators help identify uncertain segments through opacity
- Quality metrics enable systematic error analysis
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from vea.config import EMOTION_ALIASES

# Logging is configured by the CLI (vea.cli) or by the calling application.
# Library modules must not call logging.basicConfig at import time.
logger = logging.getLogger(__name__)


# =============================================================================
# EMOTION CONFIGURATION
# =============================================================================

# `aliases` is load-bearing, not cosmetic: normalize_emotion_label() rewrites
# any label not listed here - key or alias - to "neutral", silently. So a label
# the Russian stage 7A model emits but this table omits does not surface as an
# error, it surfaces as grey on the timeline and as an inflated Neutral count.
#
# That happened. `Djacon/rubert-tiny2-russian-emotion-detection` emits
# `enthusiasm`, which was absent here, and on the first full run it accounted
# for 45 of 311 segments - 14.5% of the video, relabelled Neutral. The mapping
# below is not a guess: the corpus build record for that same model's training
# data (benchmarks/russian/build-record.json in the group's reimplementation)
# records "collapse by priority; enthusiasm becomes Joy", moving 1,115 rows.
# Joy is where it belongs; Neutral is the opposite end of the valence axis.
# See docs/PROVENANCE.md section 13.
EMOTION_CONFIG = {
    "neutral": {"color": "#95a5a6", "priority": 0, "display_name": "Neutral"},
    "joy": {
        "color": "#FFD700",
        "priority": 1,
        # Derived from vea.config.EMOTION_ALIASES so this table and the CSV
        # exporter cannot disagree about what a label means.
        "aliases": [n for n, target in EMOTION_ALIASES.items() if target == "joy"],
        "display_name": "Happiness",
    },
    "fear": {"color": "#9b59b6", "priority": 2, "display_name": "Fear"},
    "anger": {"color": "#DC143C", "priority": 3, "display_name": "Anger"},
    "surprise": {"color": "#9400D3", "priority": 4, "display_name": "Surprise"},
    "sadness": {
        "color": "#00008B",
        "priority": 5,
        "aliases": [n for n, t in EMOTION_ALIASES.items() if t == "sadness"],
        "display_name": "Sadness",
    },
    "disgust": {"color": "#6B8E23", "priority": 6, "display_name": "Disgust"},
}

KNOWN_EMOTIONS = set(EMOTION_CONFIG.keys())
for config in EMOTION_CONFIG.values():
    if "aliases" in config:
        KNOWN_EMOTIONS.update(config["aliases"])


def normalize_emotion_label(emotion: str) -> Tuple[str, bool]:
    """
    Map emotion labels to standardized categories.

    Returns:
        Tuple of (normalized_emotion, is_known)
    """
    emotion_lower = emotion.lower()

    if emotion_lower in EMOTION_CONFIG:
        return emotion_lower, True

    for standard_emotion, config in EMOTION_CONFIG.items():
        if "aliases" in config and emotion_lower in config["aliases"]:
            return standard_emotion, True

    return "neutral", False


# =============================================================================
# FILE LOADING
# =============================================================================


def find_emotion_files(video_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Find all emotion prediction files (2 English models + 1 Russian).

    Returns:
        Tuple of (distilroberta_file, deberta_file, russian_file)
    """
    video_dir = Path(video_dir)

    # Find DistilRoBERTa (standard English model)
    distilroberta_files = list(video_dir.glob("emotion_distilroberta_en_local_*.json"))
    distilroberta_file = distilroberta_files[0] if distilroberta_files else None

    # Find DeBERTa (fine-tuned English model)
    deberta_files = list(video_dir.glob("emotion_deberta-finetuned_en_local_*.json"))
    deberta_file = deberta_files[0] if deberta_files else None

    # Find Russian model
    ru_files = list(video_dir.glob("emotion_*_local_*.json"))
    ru_files = [f for f in ru_files if "_en_" not in f.name]
    ru_file = ru_files[0] if ru_files else None

    if distilroberta_file:
        logger.info(f"Found DistilRoBERTa: {distilroberta_file.name}")
    if deberta_file:
        logger.info(f"Found DeBERTa: {deberta_file.name}")
    if ru_file:
        logger.info(f"Found Russian: {ru_file.name}")

    return distilroberta_file, deberta_file, ru_file


def load_emotion_predictions(file_path: Path) -> Dict:
    """Load emotion predictions from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_video_metadata(video_dir: Path) -> Optional[Dict]:
    """Load video metadata for context."""
    metadata_file = video_dir / "metadata.json"
    if not metadata_file.exists():
        return None

    with open(metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_global_scenes(video_dir: Path) -> Optional[Dict]:
    """Load global scene boundaries for visualization markers."""
    scenes_file = video_dir / "global_scenes.json"
    if not scenes_file.exists():
        return None

    with open(scenes_file, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# ENSEMBLE CROSS-VALIDATION
# =============================================================================


def ensemble_cross_validation(
    distilroberta_preds: Optional[List[Dict]],
    deberta_preds: Optional[List[Dict]],
    russian_preds: Optional[List[Dict]],
    min_confidence: float = 0.25,
) -> List[Dict]:
    """
    Ensemble voting across multiple models.

    Strategy:
    1. All 3 models agree → use consensus (high confidence)
    2. 2 models agree → use majority vote (medium confidence)
    3. All disagree → confidence-weighted selection (low confidence)
    4. Unknown emotions → prefer known emotion from other models
    5. Track agreement metrics for quality analysis

    Why ensemble approach:
    - Single model can have biases or errors
    - Multi-model consensus is more reliable
    - Confidence weighting helps when models disagree
    - Gracefully handles model failures (works with 1-3 models)

    Args:
        distilroberta_preds: Predictions from DistilRoBERTa model
        deberta_preds: Predictions from DeBERTa model
        russian_preds: Predictions from Russian model
        min_confidence: Minimum confidence threshold

    Returns:
        List of validated predictions with agreement metrics
    """
    # Collect available models
    available_models = []
    if distilroberta_preds:
        available_models.append(("distilroberta", distilroberta_preds))
    if deberta_preds:
        available_models.append(("deberta", deberta_preds))
    if russian_preds:
        available_models.append(("russian", russian_preds))

    if not available_models:
        raise ValueError("No emotion predictions available")

    # Use first available model's segments as reference
    reference_preds = available_models[0][1]
    num_segments = len(reference_preds)

    logger.info(f"Ensemble validation with {len(available_models)} model(s)")

    validated = []

    # Track agreement statistics
    full_agreement_count = 0
    majority_agreement_count = 0
    disagreement_count = 0
    unknown_emotion_handled = 0

    for i in range(num_segments):
        # Collect predictions from all available models
        model_predictions = []

        for model_name, preds in available_models:
            if i < len(preds):
                pred = preds[i]
                emotion = pred.get("predicted_emotion", "neutral")
                confidence = pred.get("confidence", 0.0)

                # Normalize emotion label
                normalized_emotion, is_known = normalize_emotion_label(emotion)

                model_predictions.append(
                    {
                        "model": model_name,
                        "emotion": normalized_emotion,
                        "original_emotion": emotion,
                        "is_known": is_known,
                        "confidence": confidence,
                        "arousal": pred.get("arousal", 0.5),
                        "valence": pred.get("valence", 0.5),
                    }
                )

        if not model_predictions:
            continue

        # Filter out unknown emotions if we have known alternatives
        known_predictions = [p for p in model_predictions if p["is_known"]]

        if not known_predictions and model_predictions:
            # All predictions are unknown - use first one
            unknown_emotion_handled += 1
            known_predictions = model_predictions

        # Count emotion votes
        emotion_votes = Counter([p["emotion"] for p in known_predictions])
        most_common_emotion, vote_count = emotion_votes.most_common(1)[0]

        # Determine agreement level
        total_models = len(known_predictions)

        if vote_count == total_models and total_models > 1:
            # Full agreement
            agreement_level = "full"
            full_agreement_count += 1
        elif vote_count >= (total_models / 2):
            # Majority agreement
            agreement_level = "majority"
            majority_agreement_count += 1
        else:
            # Disagreement - use confidence weighting
            agreement_level = "disagreement"
            disagreement_count += 1

            # Select prediction with highest confidence
            best_pred = max(known_predictions, key=lambda x: x["confidence"])
            most_common_emotion = best_pred["emotion"]

        # Calculate ensemble confidence
        # Average confidence of models that voted for selected emotion
        matching_confidences = [
            p["confidence"] for p in known_predictions if p["emotion"] == most_common_emotion
        ]
        ensemble_confidence = np.mean(matching_confidences) if matching_confidences else 0.0

        # Average arousal and valence from all models
        avg_arousal = np.mean([p["arousal"] for p in model_predictions])
        avg_valence = np.mean([p["valence"] for p in model_predictions])

        # Determine source description
        if agreement_level == "full":
            source = f"ensemble_{total_models}_models_full_agreement"
        elif agreement_level == "majority":
            source = f"ensemble_{vote_count}_of_{total_models}_agree"
        else:
            source = "ensemble_confidence_weighted"

        # Build validated prediction
        ref_pred = reference_preds[i]
        validated.append(
            {
                "segment_id": ref_pred.get("segment_id", i),
                "start": ref_pred["start"],
                "end": ref_pred["end"],
                "duration": ref_pred["duration"],
                "emotion": most_common_emotion,
                "confidence": ensemble_confidence,
                "arousal": avg_arousal,
                "valence": avg_valence,
                "agreement_level": agreement_level,
                "models_agree": vote_count,
                "total_models": total_models,
                "source": source,
            }
        )

    # Log ensemble statistics
    logger.info("Ensemble validation results:")
    logger.info(
        f"  Full agreement: {full_agreement_count} ({full_agreement_count / num_segments * 100:.1f}%)"
    )
    logger.info(
        f"  Majority agreement: {majority_agreement_count} ({majority_agreement_count / num_segments * 100:.1f}%)"
    )
    logger.info(
        f"  Disagreement: {disagreement_count} ({disagreement_count / num_segments * 100:.1f}%)"
    )
    if unknown_emotion_handled > 0:
        logger.info(f"  Unknown emotions handled: {unknown_emotion_handled}")

    return validated


# =============================================================================
# QUALITY ANALYSIS
# =============================================================================


def analyze_prediction_quality(predictions: List[Dict]) -> Dict:
    """
    Comprehensive quality analysis of predictions.

    Analysis checks:
    1. Inter-model agreement rate (how often models agree)
    2. Confidence distribution (average and low-confidence segments)
    3. Temporal consistency (detect rapid emotion changes that may indicate noise)
    4. Neutral proportion (should be 75-85% for realistic content)
    5. Overall quality score (0-100)

    Why quality analysis matters:
    - Identifies potential processing errors
    - Helps validate model performance
    - Guides interpretation of results
    - Enables systematic improvement

    Returns:
        Dictionary with quality metrics and warnings
    """
    if not predictions:
        return {"status": "no_data"}

    # Agreement metrics
    full_agreement = sum(1 for p in predictions if p["agreement_level"] == "full")
    majority_agreement = sum(1 for p in predictions if p["agreement_level"] == "majority")
    disagreement = sum(1 for p in predictions if p["agreement_level"] == "disagreement")

    total = len(predictions)
    agreement_rate = (full_agreement + majority_agreement) / total * 100

    # Confidence distribution
    confidences = [p["confidence"] for p in predictions]
    avg_confidence = np.mean(confidences)
    low_confidence_count = sum(1 for c in confidences if c < 0.3)

    # Emotion distribution
    emotion_counts = Counter([p["emotion"] for p in predictions])
    neutral_pct = emotion_counts.get("neutral", 0) / total * 100

    # Temporal consistency check
    # Rapid emotion changes in consecutive segments can indicate noise
    emotion_changes = 0
    for i in range(1, len(predictions)):
        if predictions[i]["emotion"] != predictions[i - 1]["emotion"]:
            emotion_changes += 1

    change_rate = emotion_changes / (total - 1) * 100 if total > 1 else 0

    # Generate quality warnings
    warnings = []

    if agreement_rate < 50:
        warnings.append(f"Low inter-model agreement ({agreement_rate:.1f}%)")

    if avg_confidence < 0.4:
        warnings.append(f"Low average confidence ({avg_confidence:.3f})")

    # Updated: 75-85% neutral is normal for realistic content
    if neutral_pct < 65 or neutral_pct > 90:
        warnings.append(f"Unusual neutral proportion ({neutral_pct:.1f}%, expected 75-85%)")

    if change_rate > 60:
        warnings.append(f"High emotion change rate ({change_rate:.1f}%, may indicate noise)")

    if low_confidence_count > total * 0.3:
        warnings.append(f"Many low-confidence segments ({low_confidence_count}/{total})")

    # Calculate overall quality score (0-100)
    # Weighted combination of multiple factors
    # Updated: neutral proportion target is now 80% (midpoint of 75-85%)
    quality_score = (
        agreement_rate * 0.4  # Agreement is most important
        + avg_confidence * 100 * 0.3  # Confidence matters
        + (100 - abs(neutral_pct - 80)) * 0.2  # Neutral proportion near 80%
        + (100 - change_rate) * 0.1  # Fewer changes is better
    )

    analysis = {
        "status": "analyzed",
        "total_segments": total,
        "agreement": {
            "full": full_agreement,
            "majority": majority_agreement,
            "disagreement": disagreement,
            "rate_pct": round(agreement_rate, 1),
        },
        "confidence": {
            "average": round(avg_confidence, 3),
            "low_confidence_count": low_confidence_count,
            "low_confidence_pct": round(low_confidence_count / total * 100, 1),
        },
        "emotion_distribution": dict(emotion_counts),
        "neutral_pct": round(neutral_pct, 1),
        "temporal_consistency": {
            "emotion_changes": emotion_changes,
            "change_rate_pct": round(change_rate, 1),
        },
        "quality_score": round(quality_score, 1),
        "warnings": warnings,
    }

    # Log quality analysis
    logger.info("Quality analysis:")
    logger.info(f"  Overall quality score: {quality_score:.1f}/100")
    logger.info(f"  Agreement rate: {agreement_rate:.1f}%")
    logger.info(f"  Average confidence: {avg_confidence:.3f}")
    logger.info(f"  Neutral proportion: {neutral_pct:.1f}%")

    if warnings:
        logger.warning(f"  Quality warnings ({len(warnings)}):")
        for warning in warnings:
            logger.warning(f"    - {warning}")

    return analysis


# =============================================================================
# SMOOTHING (PRESERVED FROM CURRENT VERSION)
# =============================================================================


def create_smooth_timeline(
    predictions: List[Dict], num_points: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create smoothed timeline with gentle Gaussian filter.

    Less aggressive smoothing to preserve local trends (per user feedback).
    Now also returns agreement scores for visualization.
    """
    if not predictions:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Extract segment midpoints and values
    time_points = []
    arousal_points = []
    valence_points = []
    emotion_ids = []
    agreement_points = []

    emotion_to_id = {emotion: i for i, emotion in enumerate(EMOTION_CONFIG.keys())}

    for pred in predictions:
        midpoint = (pred["start"] + pred["end"]) / 2
        time_points.append(midpoint)
        arousal_points.append(pred["arousal"] if pred["arousal"] is not None else 0.5)
        valence_points.append(pred["valence"] if pred["valence"] is not None else 0.5)
        emotion_id = emotion_to_id.get(pred["emotion"], 0)
        emotion_ids.append(emotion_id)

        # Calculate agreement score (0-1)
        agreement_score = pred["models_agree"] / pred["total_models"]
        agreement_points.append(agreement_score)

    time_points = np.array(time_points)
    arousal_points = np.array(arousal_points)
    valence_points = np.array(valence_points)
    emotion_ids = np.array(emotion_ids)
    agreement_points = np.array(agreement_points)

    # Create smooth timeline
    time_min = predictions[0]["start"]
    time_max = predictions[-1]["end"]
    time_smooth = np.linspace(time_min, time_max, num_points)

    # Interpolate arousal with gentle smoothing
    arousal_interp = interp1d(
        time_points, arousal_points, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
    arousal_smooth = arousal_interp(time_smooth)
    arousal_smooth = gaussian_filter1d(arousal_smooth, sigma=3)
    arousal_smooth = np.clip(arousal_smooth, 0, 1)

    # Interpolate valence with gentle smoothing
    valence_interp = interp1d(
        time_points, valence_points, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
    valence_smooth = valence_interp(time_smooth)
    valence_smooth = gaussian_filter1d(valence_smooth, sigma=3)
    valence_smooth = np.clip(valence_smooth, 0, 1)

    # Interpolate agreement scores
    agreement_interp = interp1d(
        time_points, agreement_points, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    agreement_smooth = agreement_interp(time_smooth)
    agreement_smooth = np.clip(agreement_smooth, 0, 1)

    # Emotion IDs - nearest neighbor (no interpolation)
    emotion_interp = interp1d(
        time_points, emotion_ids, kind="nearest", bounds_error=False, fill_value="extrapolate"
    )
    emotion_ids_smooth = emotion_interp(time_smooth)

    return time_smooth, arousal_smooth, valence_smooth, emotion_ids_smooth, agreement_smooth


def interpolate_colors(emotion_ids: np.ndarray, window_size: int = 30) -> np.ndarray:
    """
    Create gradient color transitions.

    Smaller window (30 vs 50) for more responsive color changes.
    """
    emotions_list = list(EMOTION_CONFIG.keys())
    colors = np.zeros((len(emotion_ids), 3))

    for i in range(len(emotion_ids)):
        current_id = int(emotion_ids[i])
        current_emotion = emotions_list[current_id]
        current_color = np.array(to_rgb(EMOTION_CONFIG[current_emotion]["color"]))

        # Check transition zone
        if i > 0 and i < len(emotion_ids) - 1:
            window_start = max(0, i - window_size // 2)
            window_end = min(len(emotion_ids), i + window_size // 2)

            # Count emotion occurrences in window with distance weighting
            weights = {}
            for j in range(window_start, window_end):
                emo_id = int(emotion_ids[j])
                distance = abs(j - i)
                weight = 1.0 / (1.0 + distance)
                weights[emo_id] = weights.get(emo_id, 0) + weight

            # Blend colors if multiple emotions present
            if len(weights) > 1:
                total_weight = sum(weights.values())
                blended_color = np.zeros(3)
                for emo_id, weight in weights.items():
                    emo_name = emotions_list[emo_id]
                    emo_color = np.array(to_rgb(EMOTION_CONFIG[emo_name]["color"]))
                    blended_color += emo_color * (weight / total_weight)
                colors[i] = blended_color
            else:
                colors[i] = current_color
        else:
            colors[i] = current_color

    return colors


# =============================================================================
# VISUALIZATION (PRESERVED TWO-PANEL LAYOUT)
# =============================================================================


def create_emotion_timeline_visualization(
    video_dir: Path,
    predictions: List[Dict],
    quality_analysis: Dict,
    video_metadata: Optional[Dict] = None,
    global_scenes: Optional[Dict] = None,
    output_filename: str = "emotion_timeline.png",
):
    """
    Create two-panel emotion timeline visualization with ensemble enhancements.

    Preserved layout:
    - Two panels: Arousal (top) + Valence (bottom)
    - Clean arousal panel (no data point clutter)
    - Emotion colors with agreement-based opacity
    - Scene markers with labels
    - Valence panel with confidence indicators

    Updated features:
    - More transparent neutral (gray) color
    - More defined/accented emotion colors
    - Shorter y-axis labels
    - No green agreement line
    - No warning text display
    """
    if not predictions:
        logger.warning("No predictions to visualize")
        return

    logger.info("Creating emotion timeline visualization...")

    # Create smooth timeline with agreement scores
    time_smooth, arousal_smooth, valence_smooth, emotion_ids_smooth, agreement_smooth = (
        create_smooth_timeline(predictions, num_points=500)
    )

    if len(time_smooth) == 0:
        logger.warning("Failed to create smooth timeline")
        return

    # Create gradient colors
    colors_smooth = interpolate_colors(emotion_ids_smooth, window_size=30)

    # Extract raw data points for valence panel
    time_raw = [(p["start"] + p["end"]) / 2 for p in predictions]
    valence_raw = [p["valence"] for p in predictions]
    confidence_raw = [p["confidence"] for p in predictions]

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    # Title with video ID if available
    if video_metadata:
        video_id = video_dir.name.replace("video-", "")
        video_title = video_metadata.get("title", "")
        if video_title and len(video_title) > 60:
            video_title = video_title[:57] + "..."
        title = f"Emotion Timeline: {video_title} [ID: {video_id}]"
    else:
        title = "Emotion Timeline Analysis"

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

    # =========================================================================
    # PANEL 1: AROUSAL WITH EMOTION COLORS
    # =========================================================================

    # Plot colored gradient with agreement-based opacity
    # More transparent for neutral, more opaque for emotions
    emotions_list = list(EMOTION_CONFIG.keys())

    for i in range(len(time_smooth) - 1):
        current_emotion = emotions_list[int(emotion_ids_smooth[i])]

        # Base opacity depends on agreement
        base_alpha = 0.70 + 0.25 * agreement_smooth[i]  # Range: 0.70-0.95

        # Make neutral more transparent, emotions more defined
        if current_emotion == "neutral":
            alpha = base_alpha * 0.3  # Much more transparent for neutral
        else:
            alpha = base_alpha * 1.1  # More opaque for emotions (capped at 1.0)
            alpha = min(alpha, 0.95)  # Cap at 0.95

        ax1.fill_between(
            [time_smooth[i], time_smooth[i + 1]],
            [0, 0],
            [arousal_smooth[i], arousal_smooth[i + 1]],
            color=colors_smooth[i],
            alpha=alpha,
            linewidth=0,
        )

    # Plot smooth arousal line
    ax1.plot(time_smooth, arousal_smooth, color="black", linewidth=2.5, alpha=0.9, zorder=10)

    # Scene markers
    if global_scenes is not None:
        scenes = global_scenes.get("scenes", [])
        for idx, scene in enumerate(scenes):
            scene_start = scene["start"]
            ax1.axvline(
                x=scene_start, color="gray", linestyle="--", linewidth=1.5, alpha=0.4, zorder=1
            )
            # Add scene numbers at top
            if idx < len(scenes) - 1:
                ax1.text(
                    scene_start,
                    1.02,
                    f"S{idx + 1}",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    color="gray",
                    alpha=0.6,
                    transform=ax1.get_xaxis_transform(),
                )

    # Shorter y-axis label
    ax1.set_ylabel("Intensity", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
    ax1.set_title("Emotional Intensity Over Time (with emotion colors)", fontsize=11, pad=10)

    # Emotion legend
    legend_elements = []
    for emotion, config in EMOTION_CONFIG.items():
        display_label = config.get("display_name", emotion.capitalize())
        legend_elements.append(
            plt.Line2D([0], [0], color=config["color"], linewidth=5, label=display_label)
        )
    ax1.legend(handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=9, ncol=4)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # =========================================================================
    # PANEL 2: VALENCE (POSITIVE/NEGATIVE)
    # =========================================================================

    # Plot smooth valence line
    ax2.plot(
        time_smooth,
        valence_smooth,
        color="#2ecc71",
        linewidth=2.5,
        alpha=0.9,
        label="Smoothed Valence",
        zorder=10,
    )

    # Overlay raw valence points with confidence indicators
    for i, (t, v, conf) in enumerate(zip(time_raw, valence_raw, confidence_raw)):
        alpha = 0.3 + 0.7 * conf
        size = 20 if conf >= 0.5 else 15
        color = "#2ecc71" if v >= 0.5 else "#e74c3c"
        ax2.scatter(
            t, v, c=color, s=size, alpha=alpha, edgecolors="black", linewidth=0.5, zorder=11
        )

    # Fill positive/negative regions
    ax2.fill_between(
        time_smooth,
        0.5,
        valence_smooth,
        where=(valence_smooth >= 0.5),
        color="#2ecc71",
        alpha=0.2,
        label="Positive",
    )
    ax2.fill_between(
        time_smooth,
        0.5,
        valence_smooth,
        where=(valence_smooth < 0.5),
        color="#e74c3c",
        alpha=0.2,
        label="Negative",
    )

    # Neutral line
    ax2.axhline(y=0.5, color="gray", linestyle="-", linewidth=1, alpha=0.5, label="Neutral")

    # Scene markers
    if global_scenes is not None:
        for scene in global_scenes.get("scenes", []):
            ax2.axvline(
                x=scene["start"], color="gray", linestyle="--", linewidth=1.5, alpha=0.4, zorder=1
            )

    ax2.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    # Shorter y-axis label
    ax2.set_ylabel("Positive ↔ Negative", fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
    ax2.set_title("Emotional Valence Over Time (positive vs negative emotion)", fontsize=11, pad=10)

    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # =========================================================================
    # QUALITY SUMMARY BOX (TOP RIGHT OF UPPER PANEL)
    # =========================================================================

    if quality_analysis and quality_analysis.get("status") == "analyzed":
        qa = quality_analysis

        # Create summary text
        summary_lines = [
            f"Quality: {qa['quality_score']:.0f}/100",
            f"Agreement: {qa['agreement']['rate_pct']:.0f}%",
            f"Confidence: {qa['confidence']['average']:.2f}",
            f"Models: {predictions[0]['total_models']}",
        ]

        # Color based on quality score
        if qa["quality_score"] >= 70:
            box_color = "#2ecc71"  # Green
        elif qa["quality_score"] >= 50:
            box_color = "#f39c12"  # Orange
        else:
            box_color = "#e74c3c"  # Red

        box_text = "\n".join(summary_lines)

        bbox_props = dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=box_color,
            linewidth=2,
            alpha=0.95,
        )

        ax1.text(
            0.98,
            0.97,
            box_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=bbox_props,
            family="monospace",
        )

    # Shared x-axis
    ax2.set_xlim(time_smooth[0], time_smooth[-1])

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = video_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(f"Saved visualization: {output_path.name}")


# =============================================================================
# MAIN PROCESSING
# =============================================================================


def process_video(video_dir: Path, min_confidence: float = 0.25) -> Dict:
    """
    Process emotion timeline with ensemble validation and quality analysis.

    Workflow:
    1. Find all available emotion model predictions (2 English + 1 Russian)
    2. Load predictions from all available models
    3. Perform ensemble cross-validation
    4. Analyze prediction quality
    5. Create two-panel visualization with enhancements
    """
    video_dir = Path(video_dir)

    logger.info(f"Processing: {video_dir.name}")

    # Find emotion files
    distilroberta_file, deberta_file, russian_file = find_emotion_files(video_dir)

    if not any([distilroberta_file, deberta_file, russian_file]):
        logger.error("No emotion predictions found")
        return {"status": "failed", "reason": "no_emotion_predictions"}

    # Load predictions
    distilroberta_data = (
        load_emotion_predictions(distilroberta_file) if distilroberta_file else None
    )
    deberta_data = load_emotion_predictions(deberta_file) if deberta_file else None
    russian_data = load_emotion_predictions(russian_file) if russian_file else None

    distilroberta_preds = distilroberta_data.get("predictions", []) if distilroberta_data else None
    deberta_preds = deberta_data.get("predictions", []) if deberta_data else None
    russian_preds = russian_data.get("predictions", []) if russian_data else None

    # Ensemble cross-validation
    try:
        validated_predictions = ensemble_cross_validation(
            distilroberta_preds, deberta_preds, russian_preds, min_confidence=min_confidence
        )
    except Exception as e:
        logger.error(f"Ensemble validation failed: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "failed", "reason": f"ensemble_error: {e}"}

    # Quality analysis
    quality_analysis = analyze_prediction_quality(validated_predictions)

    # Load metadata and scenes
    video_metadata = load_video_metadata(video_dir)
    global_scenes = load_global_scenes(video_dir)

    # Create visualization
    try:
        create_emotion_timeline_visualization(
            video_dir, validated_predictions, quality_analysis, video_metadata, global_scenes
        )
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "failed", "reason": f"visualization_error: {e}"}

    # Calculate statistics
    emotion_counts = Counter([p["emotion"] for p in validated_predictions])
    confidence_sum = sum(p["confidence"] for p in validated_predictions)
    avg_confidence = confidence_sum / len(validated_predictions)

    return {
        "status": "success",
        "total_segments": len(validated_predictions),
        "quality_analysis": quality_analysis,
        "average_confidence": round(avg_confidence, 3),
        "emotion_distribution": dict(emotion_counts),
        "emotion_percentages": {
            emotion: round(count / len(validated_predictions) * 100, 1)
            for emotion, count in emotion_counts.items()
        },
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    """
    Enhanced emotion timeline visualization with ensemble validation.
    
    Key features:
    - Ensemble validation across 3 models (2 English + 1 Russian)
    - Automatic quality analysis and error detection
    - Two-panel layout (arousal + valence)
    - More transparent neutral, accented emotion colors
    - Agreement indicators via opacity
    - Quality score display
    - Shorter, cleaner labels
    """
    import sys

    downloads_dir = Path("downloads")

    if not downloads_dir.exists():
        print("Downloads directory not found")
        sys.exit(1)

    video_dirs = sorted(downloads_dir.glob("video-*"))

    if not video_dirs:
        print("No video directories found in downloads/")
        sys.exit(1)

    print(f"Found {len(video_dirs)} video(s) to process")
    print("=" * 70)
    print("EMOTION TIMELINE WITH ENSEMBLE VALIDATION")
    print("=" * 70)
    print("Enhancements:")
    print("1. Ensemble validation (DistilRoBERTa + DeBERTa + Russian)")
    print("2. Automatic quality analysis (75-85% neutral is normal)")
    print("3. Agreement indicators via opacity")
    print("4. More transparent neutral, accented emotion colors")
    print("5. Clean, shorter labels")
    print("6. Two-panel layout (arousal + valence)")
    print("=" * 70)

    # Configuration
    min_confidence = 0.25

    print("\nConfiguration:")
    print(f"  Minimum confidence: {min_confidence}")
    print("=" * 70)

    # Process each video
    success_count = 0
    failed_count = 0

    for video_dir in video_dirs:
        print(f"\n{'=' * 70}")
        print(f"Processing: {video_dir.name}")
        print("=" * 70)

        try:
            result = process_video(video_dir, min_confidence=min_confidence)

            if result["status"] == "success":
                success_count += 1

                qa = result["quality_analysis"]

                print("\nSuccess!")
                print(f"  Total segments: {result['total_segments']}")
                print(f"  Quality score: {qa['quality_score']:.0f}/100")
                print(f"  Agreement rate: {qa['agreement']['rate_pct']:.0f}%")
                print(f"  Average confidence: {result['average_confidence']:.3f}")

                if qa["warnings"]:
                    print(f"\n  Quality warnings ({len(qa['warnings'])}):")
                    for warning in qa["warnings"]:
                        print(f"    - {warning}")

                print("\n  Emotion distribution:")
                for emotion, pct in sorted(
                    result["emotion_percentages"].items(), key=lambda x: -x[1]
                ):
                    count = result["emotion_distribution"][emotion]
                    print(f"    {emotion}: {count} ({pct:.1f}%)")

                print(f"\n  Output: {video_dir.name}/emotion_timeline.png")
            else:
                failed_count += 1
                print(f"\nFailed: {result.get('reason', 'unknown')}")

        except Exception as e:
            failed_count += 1
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"Total videos: {len(video_dirs)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")

    print("\n" + "=" * 70)
    print("OUTPUT FEATURES")
    print("=" * 70)
    print("Visualization: emotion_timeline.png (two-panel layout)")
    print("\nPanel 1 (Top): Intensity with emotion colors")
    print("  - Agreement-based opacity (more opaque = more agreement)")
    print("  - Neutral is very transparent, emotions are accented")
    print("  - Smooth curve preserving local trends")
    print("  - Scene markers with labels")
    print("  - Quality score in corner box")
    print("\nPanel 2 (Bottom): Positive ↔ Negative")
    print("  - Green area = positive emotions")
    print("  - Red area = negative emotions")
    print("  - Data points with confidence indicators")
    print("\nQuality Analysis:")
    print("  - Overall quality score (0-100)")
    print("  - Inter-model agreement rate")
    print("  - 75-85% neutral is considered normal")
    print("  - Automatic warning detection")
