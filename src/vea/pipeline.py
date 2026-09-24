"""
Main Pipeline Orchestrator - Complete (Stages 1-9)

This script orchestrates all 9 stages of the video emotion analysis pipeline:
  Stage 1: Download YouTube video and audio
  Stage 2: Detect scene boundaries
  Stage 3: Preprocess audio (normalize, optional noise reduction)
  Stage 4: Transcribe audio with Whisper
  Stage 5A: Scene alignment (global scenes + local segments)
  Stage 5B: Translation (Russian to English)
  Stage 6A: Russian intensity classification (arousal-valence)
  Stage 6B: English intensity classification (arousal-valence)
  Stage 7A: Russian emotion classification (7-class)
  Stage 7B: English emotion classification (7-class)
  Stage 8: Visualization (emotion timeline)
  Stage 9: CSV export (consolidated data)

Design principles:
- Sequential processing: Each stage validates previous stage output
- Skip logic: Reuses existing processed data (unless force flags set)
- Fail fast: a failure in stages 1-7 stops the run; stages 8 and 9 only log
- Progress visibility: Clear logging of what's happening
- Configuration transparency: Shows what settings are being used
- Resource management: the sentence embedder and NLLB are loaded once per run

Usage:
    uv run vea run <youtube_url>

Example:
    uv run vea run "https://www.youtube.com/watch?v=mqGSkDFeLEo"
"""

import copy
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Stage imports.
#
# Originally these were bare imports of sibling files (`from download_module
# import ...`) wrapped in a try/except that called sys.exit(1) on failure. That
# only worked when the interpreter was launched from inside the module
# directory, and made the module impossible to import from a test. They are now
# package-relative and allowed to raise: an ImportError here is a broken
# install, and the traceback is more useful than a printed checklist.
from vea.config import (
    MODEL_REGISTRY,
    apply_device_setting,
    resolve_model,
    translation_model_ref,
)

# Stages 6A/6B and 7A/7B are imported as modules rather than as symbols, because
# the Russian and English variants define same-named functions.
from vea.stages import emotion_en as EmotionEN
from vea.stages import emotion_ru as EmotionRU
from vea.stages import export as CSVExportModule
from vea.stages import intensity_en as IntensityEN
from vea.stages import intensity_ru as IntensityRU
from vea.stages import visualize as VisualizationModule
from vea.stages.align import load_semantic_model
from vea.stages.align import process_video as align_scenes
from vea.stages.audio import process_video as preprocess_audio
from vea.stages.download import download_youtube_content, extract_video_id_from_url
from vea.stages.scenes import process_video as detect_scenes
from vea.stages.transcribe import process_video as transcribe_audio
from vea.stages.translate import NLLBTranslator
from vea.stages.translate import process_directory as translate_directory

# Logging is configured by the CLI (vea.cli) or by the calling application.
# Library modules must not call logging.basicConfig at import time.
logger = logging.getLogger(__name__)


def _resolve_stage_model(stage_config: Dict) -> Dict:
    """Fill in the concrete model path for a stage config.

    Stage configs declare a registry key under ``'model'``; the stage modules
    expect a loadable reference under ``'path'``. Resolution happens here, at
    the moment the stage runs, rather than at import time - a missing local
    checkpoint must not stop the whole module from importing.

    Args:
        stage_config: One entry of :data:`DEFAULT_CONFIGS`, or one element of a
            stage's ``'models'`` list.

    Returns:
        A shallow copy with ``'path'`` set.

    Raises:
        vea.config.MissingCheckpointError: The model is a local checkpoint that
            is not present. Callers catch this and fail just their own stage.
    """
    resolved = dict(stage_config)
    if "model" in stage_config:
        resolved["path"] = resolve_model(stage_config["model"])
        # Carry the model's declared text transform into the config the stage
        # module reads, so each ensemble member is fed the text convention its
        # own training data used. Stage 7B's two members differ here: the
        # DeBERTa checkpoint was trained on normalised text, DistilRoBERTa on
        # raw text, and feeding either the other's convention would create a
        # train/serve skew rather than remove one.
        spec = MODEL_REGISTRY[stage_config["model"]]
        if spec.preprocess:
            resolved["config"] = {
                **resolved.get("config", {}),
                "preprocess": spec.preprocess,
            }
    return resolved


# Default configurations for each stage
# These follow the recommended settings from each module
DEFAULT_CONFIGS = {
    "stage_1_download": {"base_output": "downloads", "force_redownload": False},
    "stage_2_scene_detection": {
        "detector_type": "adaptive",
        "threshold": 3.0,
        "min_scene_length": 2.0,
        "downscale": None,
        "force_redetect": False,
    },
    "stage_3_audio_preprocessing": {
        "normalize": True,
        "target_db": -12.0,
        "reduce_noise": False,
        "noise_strength": 0.5,
        "force_reprocess": False,
    },
    "stage_4_transcription": {
        "model_size": "large-v3",
        "language": "ru",
        "duration_limit": "full",
        "device": None,
        "compute_type": "float16",
    },
    "stage_5a_scene_alignment": {
        # Global scenes (for visualization structure)
        "max_global_scene_duration": 900.0,  # 15 min max
        "min_gap_for_global_split": 5.0,  # 5s silence triggers split
        # Local segments (for emotion classification)
        "min_words_per_segment": 8,  # Prevent short fragments
        "target_words_per_segment": 25,  # Optimal for emotion
        "max_words_per_segment": 60,  # Prevent overly long
        "max_sentences_per_segment": 5,  # Safety limit
        # Semantic similarity (multilingual embeddings)
        "min_similarity_threshold": 0.35,  # 0.3-0.5 recommended
        "force_reprocess": False,
    },
    "stage_5b_translation": {
        # Resolved from VEA_TRANSLATION_MODEL, defaulting to the 3.3B the
        # server has always used. The two smaller NLLB sizes were declared in
        # the registry and unreachable; 3.3B is ~17 GB of weights and wants
        # 8 GB of VRAM, which no laptop is going to provide.
        "model_name": translation_model_ref(),
        "num_beams": 4,  # Optimal quality/speed for NLLB
        "batch_size": 32,  # Auto-reduces on OOM
        "preferred_gpu": None,  # None for auto-select
        "skip_existing": True,  # Skip already translated files
        "src_lang": "rus_Cyrl",  # Russian (Cyrillic)
        "tgt_lang": "eng_Latn",  # English (Latin)
    },
    # Stages 6 and 7 name models by their key in vea.config.MODEL_REGISTRY
    # instead of carrying a hardcoded path. The concrete path is resolved at the
    # start of each stage by _resolve_stage_model(), so a missing checkpoint is
    # reported as a named, actionable failure instead of an OSError from inside
    # transformers.
    "stage_6a_russian_intensity": {
        "name": "xlmroberta-large",
        "model": "va-xlmroberta-large",
        "description": "Multilingual XLM-RoBERTa for Russian VA prediction",
        "config": {
            "batch_size": 32,
            "device": None,  # None -> VEA_DEVICE if set, else the stage decides
            "skip_advertisements": True,
            "threshold_method": "median",  # Video-adaptive threshold
        },
    },
    "stage_6b_english_intensity": {
        "name": "xlmroberta-large",
        "model": "va-xlmroberta-large",  # Same weights, English inputs
        "description": "Multilingual XLM-RoBERTa for English VA prediction",
        "config": {
            "batch_size": 32,
            "device": None,
            "skip_advertisements": True,
            "threshold_method": "median",
        },
    },
    "stage_7a_russian_emotion": {
        "name": "rubert-tiny2",
        "model": "emotion-ru-rubert-tiny2",
        "description": "Fast Russian emotion detection, 7 classes",
        "config": {
            "batch_size": 64,  # Smaller model, can use larger batches
            "device": None,
        },
    },
    "stage_7b_english_emotion": {
        # Two English emotion models for ensemble validation.
        #
        # The second slot now runs the real thing. The original orchestrator
        # named it 'deberta-finetuned' and wrote its outputs to
        # emotion_deberta-finetuned_en_local_*.json while pointing it at
        # tae898/emoberta-large - a third-party RoBERTa-large trained on MELD,
        # not the checkpoint described in docs/model_cards/emotion_en_deberta.md
        # - so every published "deberta" result actually came from EmoBERTa.
        # That checkpoint was retrained on 2026-09-16 (macro F1 0.8162 against
        # the card's 0.8127) and is wired in below. 'emotion-en-emoberta' stays
        # in MODEL_REGISTRY so the old results remain reproducible; any output
        # generated before the switch is still EmoBERTa and must be regenerated
        # before it is reported. See docs/PROVENANCE.md sections 1 and 10.
        #
        # Note the two members now receive *different text*: the registry gives
        # emotion-en-deberta preprocess='super_emotion_v1', so _resolve_stage_model
        # injects the twelve-step cleaner for this slot only. DistilRoBERTa was
        # trained on raw GoEmotions text and keeps getting raw text. That
        # asymmetry is deliberate - see docs/PROVENANCE.md section 6.
        "models": [
            {
                "name": "distilroberta",
                "model": "emotion-en-distilroberta",
                "description": "English emotion detection (DistilRoBERTa)",
                "config": {"batch_size": 32, "device": None},
            },
            {
                "name": "deberta-finetuned",
                "model": "emotion-en-deberta",
                "description": "English emotion detection (DeBERTa-v3-base, retrained)",
                "config": {
                    # Was 16 for EmoBERTa-large. deberta-v3-base is ~183M
                    # parameters against EmoBERTa's ~355M, so 32 fits the same
                    # VRAM; drop it back to 16 if stage 7B OOMs on a busy card.
                    "batch_size": 32,
                    "device": None,
                },
            },
        ]
    },
    "stage_8_visualization": {
        "min_confidence": 0.25,  # Minimum confidence for predictions
        "output_filename": "emotion_timeline.png",
    },
    "stage_9_csv_export": {"output_filename": "emotion_analysis_data.csv"},
}


def run_stage_1_download(url: str, config: Dict) -> Optional[Path]:
    """
    Stage 1: Download YouTube video and audio

    Returns: Path to video directory or None if failed
    """
    logger.info("=" * 70)
    logger.info("STAGE 1: DOWNLOADING VIDEO")
    logger.info("=" * 70)

    try:
        results = download_youtube_content(
            urls=[url],
            base_output=config["base_output"],
            force_redownload=config["force_redownload"],
        )

        if not results or results[0]["status"] not in ["success", "skipped"]:
            logger.error(f"Download failed: {results[0].get('error', 'Unknown error')}")
            return None

        result = results[0]
        video_dir = Path(result["video_dir"])

        logger.info("\nStage 1 complete:")
        logger.info(f"  Video ID: {result['video_id']}")
        logger.info(f"  Status: {result['status'].upper()}")
        logger.info(f"  Location: {video_dir}")

        if result["status"] == "skipped":
            logger.info("  Note: Video already downloaded, reusing existing data")

        return video_dir

    except Exception as e:
        logger.error(f"Stage 1 failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_stage_2_scene_detection(video_dir: Path, config: Dict) -> bool:
    """
    Stage 2: Detect scene boundaries

    Returns: True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: DETECTING SCENE BOUNDARIES")
    logger.info("=" * 70)

    try:
        result = detect_scenes(
            video_dir=video_dir, config=config, force_redetect=config.get("force_redetect", False)
        )

        stats = result["statistics"]
        logger.info("\nStage 2 complete:")
        logger.info(f"  Scenes detected: {stats['total_scenes']}")
        logger.info(f"  Average duration: {stats['average_duration']:.1f}s")
        logger.info(f"  Range: {stats['shortest_scene']:.1f}s - {stats['longest_scene']:.1f}s")
        logger.info(f"  Output: {video_dir / 'scene_boundaries.json'}")

        return True

    except Exception as e:
        logger.error(f"Stage 2 failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_stage_3_audio_preprocessing(video_dir: Path, config: Dict) -> bool:
    """
    Stage 3: Preprocess audio (normalize volume, optional noise reduction)

    Returns: True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3: PREPROCESSING AUDIO")
    logger.info("=" * 70)

    try:
        quality_metrics = preprocess_audio(
            video_dir=video_dir, config=config, force_reprocess=config.get("force_reprocess", False)
        )

        norm = quality_metrics["normalization"]
        logger.info("\nStage 3 complete:")
        logger.info(f"  Duration: {quality_metrics['duration_seconds']:.1f}s")
        logger.info(f"  RMS level: {norm['normalized_rms_db']:.1f} dB")
        logger.info(f"  Peak level: {norm['final_peak']:.3f}")
        logger.info(f"  Silence detected: {quality_metrics['silence_total_seconds']:.1f}s")

        if quality_metrics["quality_flags"]:
            logger.warning(f"  Quality issues: {', '.join(quality_metrics['quality_flags'])}")
        else:
            logger.info("  Quality: OK")

        logger.info(f"  Output: {video_dir / 'preprocessed_audio.wav'}")

        return True

    except Exception as e:
        logger.error(f"Stage 3 failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_stage_4_transcription(video_dir: Path, config: Dict) -> bool:
    """
    Stage 4: Transcribe audio with Whisper (Russian speech recognition)

    Returns: True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 4: TRANSCRIBING AUDIO")
    logger.info("=" * 70)

    try:
        result = transcribe_audio(video_dir=video_dir, config=config)

        meta = result["metadata"]
        stats = result["statistics"]

        logger.info("\nStage 4 complete:")
        logger.info(f"  Model: {meta['model']}")
        logger.info(f"  Duration transcribed: {meta['duration']:.1f}s")
        logger.info(f"  Total words: {stats['total_words']}")
        logger.info(f"  Total segments: {stats['total_segments']}")
        logger.info(f"  Average confidence: {stats['average_confidence']:.3f}")

        if stats.get("low_confidence_segments", 0) > 0:
            logger.warning(f"  Low confidence segments: {stats['low_confidence_segments']}")

        transcription_files = list(video_dir.glob("transcription_*.json"))
        if transcription_files:
            logger.info(f"  Output: {transcription_files[0].name}")

        return True

    except Exception as e:
        logger.error(f"Stage 4 failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_stage_5a_scene_alignment(video_dir: Path, config: Dict, semantic_model) -> bool:
    """
    Stage 5A: Scene alignment (create global scenes and local segments)

    Design: Two-level hierarchical segmentation
    - Global scenes: High-level narrative structure (5-15 min)
    - Local segments: Coherent ideas for emotion analysis (8-60 words)

    Note: Uses semantic embeddings for better understanding of idea boundaries

    Returns: True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 5A: SCENE ALIGNMENT")
    logger.info("=" * 70)

    try:
        # Check prerequisites
        scene_boundaries = video_dir / "scene_boundaries.json"
        transcription_files = list(video_dir.glob("transcription_*.json"))

        if not scene_boundaries.exists():
            logger.error(f"Scene boundaries not found: {scene_boundaries}")
            logger.error("Stage 2 (scene detection) must complete first")
            return False

        if not transcription_files:
            logger.error("No transcription files found")
            logger.error("Stage 4 (transcription) must complete first")
            return False

        logger.info(f"Using transcription: {transcription_files[0].name}")
        logger.info(
            f"Semantic model: {'Enabled' if semantic_model else 'Disabled (word overlap fallback)'}"
        )

        # Process video
        global_result, local_result = align_scenes(
            video_dir=video_dir,
            config=config,
            transcription_name=None,  # Auto-select best
            force_reprocess=config.get("force_reprocess", False),
            semantic_model=semantic_model,
        )

        # Display results
        logger.info("\nStage 5A complete:")
        logger.info(f"  Global scenes created: {len(global_result.get('global_scenes', []))}")

        global_stats = global_result.get("statistics", {})
        if global_stats:
            logger.info(
                f"  Global scene duration range: {global_stats.get('shortest_scene', 0):.1f}s - {global_stats.get('longest_scene', 0):.1f}s"
            )

        logger.info(f"  Local segments created: {len(local_result.get('segments', []))}")

        local_stats = local_result.get("statistics", {})
        if local_stats:
            logger.info(f"  Average segment words: {local_stats.get('average_word_count', 0):.1f}")
            logger.info(
                f"  Average segment duration: {local_stats.get('average_duration', 0):.1f}s"
            )

        logger.info("  Output: global_scenes.json")

        # Find local segments file
        trans_base = transcription_files[0].name.replace(".json", "")
        local_file = video_dir / f"local_segments_{trans_base}.json"
        if local_file.exists():
            logger.info(f"  Output: {local_file.name}")

        return True

    except Exception as e:
        logger.error(f"Stage 5A failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_stage_5b_translation(video_dir: Path, config: Dict, translator: NLLBTranslator) -> bool:
    """
    Stage 5B: Translation (Russian to English)

    Design: Uses NLLB-200 model with beam search for quality
    - Translates local segments from Russian to English
    - Enables cross-validation with English emotion models
    - Batch processing with auto OOM handling

    Note: This stage can be resource-intensive. The translator
    handles batch size reduction automatically if GPU memory is insufficient.

    Returns: True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 5B: TRANSLATION")
    logger.info("=" * 70)

    try:
        # Check prerequisites
        segments_files = list(video_dir.glob("local_segments_*.json"))
        segments_files = [f for f in segments_files if not f.name.endswith("_translated.json")]

        if not segments_files:
            logger.error("No local segments files found")
            logger.error("Stage 5A (scene alignment) must complete first")
            return False

        logger.info(f"Found {len(segments_files)} local segments file(s) to translate")
        logger.info(f"Model: {config['model_name']}")
        logger.info(f"Beam search: {config['num_beams']} beams")
        logger.info(f"Batch size: {config['batch_size']} (auto-reduces on OOM)")

        # Check if already translated
        if config.get("skip_existing", True):
            translated_files = list(video_dir.glob("local_segments_*_translated.json"))
            if translated_files:
                logger.info(f"Found {len(translated_files)} already translated file(s)")
                if len(translated_files) >= len(segments_files):
                    logger.info("All files already translated - SKIPPING")
                    logger.info("  (Set skip_existing=False to retranslate)")
                    return True

        # Process directory
        results = translate_directory(
            data_dir=video_dir,
            translator=translator,
            file_pattern="local_segments_*.json",
            skip_existing=config.get("skip_existing", True),
            src_lang=config.get("src_lang", "rus_Cyrl"),
            tgt_lang=config.get("tgt_lang", "eng_Latn"),
            batch_size=config.get("batch_size", 32),
            num_beams=config.get("num_beams", 4),
        )

        # Count results
        success = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        skipped = len(segments_files) - success - failed

        logger.info("\nStage 5B complete:")
        logger.info(f"  Successfully translated: {success}")
        if skipped > 0:
            logger.info(f"  Skipped (already translated): {skipped}")
        if failed > 0:
            logger.warning(f"  Failed: {failed}")

        # Find translated files
        translated_files = list(video_dir.glob("local_segments_*_translated.json"))
        if translated_files:
            logger.info(f"  Output: {len(translated_files)} translated file(s)")

        return failed == 0  # Success only if no failures

    except Exception as e:
        logger.error(f"Stage 5B failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_stage_6a_russian_intensity(video_dir: Path, config: Dict) -> bool:
    """
    Stage 6A: Russian intensity classification (arousal-valence)

    Design: Continuous arousal/valence prediction with video-adaptive thresholds
    - Processes original Russian segments (not translated)
    - Uses multilingual XLM-RoBERTa model
    - Calculates video-specific emotion significance threshold
    - Binary filter for downstream emotion classification

    Returns: True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 6A: RUSSIAN INTENSITY CLASSIFICATION")
    logger.info("=" * 70)

    try:
        # Check prerequisites
        segments_files = list(video_dir.glob("local_segments_*.json"))
        segments_files = [f for f in segments_files if not f.name.endswith("_translated.json")]

        if not segments_files:
            logger.error("No Russian segments files found")
            logger.error("Stage 5A (scene alignment) must complete first")
            return False

        logger.info(f"Model: {config['name']}")
        logger.info(f"Threshold method: {config['config']['threshold_method']}")

        # Process video
        results = IntensityRU.process_video(
            video_dir=video_dir, model_config=_resolve_stage_model(config), force_reprocess=False
        )

        # Display results
        success_count = sum(1 for r in results if r["status"] == "success")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        failed_count = sum(1 for r in results if r["status"] == "failed")

        logger.info("\nStage 6A complete:")
        logger.info(f"  Processed: {success_count}")
        if skipped_count > 0:
            logger.info(f"  Skipped (already processed): {skipped_count}")
        if failed_count > 0:
            logger.warning(f"  Failed: {failed_count}")

        # Show statistics from first successful result
        for result in results:
            if result["status"] in ["success", "skipped"]:
                stats = result["result"]["statistics"]
                logger.info(f"  Segments classified: {stats['classified_segments']}")
                logger.info(
                    f"  Arousal: mean={stats['arousal']['mean']:.3f}, median={stats['arousal']['median']:.3f}"
                )
                logger.info(f"  Emotion threshold: {stats['emotion_threshold']['value']:.3f}")
                logger.info(
                    f"  Emotionally significant: {stats['emotion_threshold']['emotionally_significant_percentage']:.1f}%"
                )
                break

        return failed_count == 0

    except Exception as e:
        logger.error(f"Stage 6A failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_stage_6b_english_intensity(video_dir: Path, config: Dict) -> bool:
    """
    Stage 6B: English intensity classification (arousal-valence)

    Design: Processes translated segments for cross-language validation
    - Uses same multilingual XLM-RoBERTa model
    - Processes English translations (*_translated.json)
    - Enables comparison with Russian intensity predictions

    Returns: True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 6B: ENGLISH INTENSITY CLASSIFICATION")
    logger.info("=" * 70)

    try:
        # Check prerequisites
        translated_files = list(video_dir.glob("local_segments_*_translated.json"))

        if not translated_files:
            logger.error("No translated segments files found")
            logger.error("Stage 5B (translation) must complete first")
            return False

        logger.info(f"Model: {config['name']}")
        logger.info(f"Threshold method: {config['config']['threshold_method']}")

        # Process video
        results = IntensityEN.process_video(
            video_dir=video_dir, model_config=_resolve_stage_model(config), force_reprocess=False
        )

        # Display results
        success_count = sum(1 for r in results if r["status"] == "success")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        failed_count = sum(1 for r in results if r["status"] == "failed")

        logger.info("\nStage 6B complete:")
        logger.info(f"  Processed: {success_count}")
        if skipped_count > 0:
            logger.info(f"  Skipped (already processed): {skipped_count}")
        if failed_count > 0:
            logger.warning(f"  Failed: {failed_count}")

        # Show statistics from first successful result
        for result in results:
            if result["status"] in ["success", "skipped"]:
                stats = result["result"]["statistics"]
                logger.info(f"  Segments classified: {stats['classified_segments']}")
                logger.info(
                    f"  Arousal: mean={stats['arousal']['mean']:.3f}, median={stats['arousal']['median']:.3f}"
                )
                logger.info(f"  Emotion threshold: {stats['emotion_threshold']['value']:.3f}")
                logger.info(
                    f"  Emotionally significant: {stats['emotion_threshold']['emotionally_significant_percentage']:.1f}%"
                )
                break

        return failed_count == 0

    except Exception as e:
        logger.error(f"Stage 6B failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_stage_7a_russian_emotion(video_dir: Path, config: Dict) -> bool:
    """
    Stage 7A: Russian emotion classification (7-class)

    Design: Two-stage approach (intensity filter + emotion classification)
    - Only classifies emotionally significant segments (from Stage 6A)
    - Low-arousal segments automatically assigned as neutral
    - Uses ruBERT-based model for Russian text
    - 7 emotion classes: neutral, joy, sadness, anger, fear, surprise, disgust

    Returns: True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 7A: RUSSIAN EMOTION CLASSIFICATION")
    logger.info("=" * 70)

    try:
        # Check prerequisites
        segments_files = list(video_dir.glob("local_segments_*.json"))
        segments_files = [f for f in segments_files if not f.name.endswith("_translated.json")]

        arousal_files = list(video_dir.glob("arousal_*_local_*.json"))
        arousal_files = [
            f for f in arousal_files if "_en_" not in f.name
        ]  # Exclude English arousal

        if not segments_files:
            logger.error("No Russian segments files found")
            logger.error("Stage 5A (scene alignment) must complete first")
            return False

        if not arousal_files:
            logger.error("No Russian arousal predictions found")
            logger.error("Stage 6A (Russian intensity) must complete first")
            return False

        logger.info(f"Model: {config['name']}")
        logger.info("Emotion classes: 7 (neutral, joy, sadness, anger, fear, surprise, disgust)")

        # Process video
        results = EmotionRU.process_video(
            video_dir=video_dir, model_config=_resolve_stage_model(config), force_reprocess=False
        )

        # Display results
        success_count = sum(1 for r in results if r["status"] == "success")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        failed_count = sum(1 for r in results if r["status"] == "failed")

        logger.info("\nStage 7A complete:")
        logger.info(f"  Processed: {success_count}")
        if skipped_count > 0:
            logger.info(f"  Skipped (already processed): {skipped_count}")
        if failed_count > 0:
            logger.warning(f"  Failed: {failed_count}")

        # Show statistics from first successful result
        for result in results:
            if result["status"] in ["success", "skipped"]:
                stats = result["result"]["statistics"]
                logger.info(f"  Total segments: {stats['total_segments']}")
                logger.info(f"  Emotionally significant: {stats['emotionally_significant']}")
                logger.info(f"  Low-arousal neutral: {stats['low_arousal_neutral']}")

                # Show emotion distribution
                if "emotion_percentages" in stats:
                    logger.info("  Top emotions:")
                    sorted_emotions = sorted(
                        stats["emotion_percentages"].items(), key=lambda x: -x[1]
                    )[:3]
                    for emotion, pct in sorted_emotions:
                        count = stats["emotion_distribution"].get(emotion, 0)
                        logger.info(f"    {emotion}: {count} ({pct:.1f}%)")
                break

        return failed_count == 0

    except Exception as e:
        logger.error(f"Stage 7A failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_stage_7b_english_emotion(video_dir: Path, config: Dict) -> bool:
    """
    Stage 7B: English emotion classification (7-class)

    Design: Processes translated segments with multiple English models
    - Only classifies emotionally significant segments (from Stage 6B)
    - Uses two English models for ensemble validation:
      1. DistilRoBERTa-base (fast, accurate)
      2. DeBERTa-large (more powerful)
    - Enables cross-language emotion comparison

    Returns: True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 7B: ENGLISH EMOTION CLASSIFICATION")
    logger.info("=" * 70)

    try:
        # Check prerequisites
        translated_files = list(video_dir.glob("local_segments_*_translated.json"))
        arousal_files = list(video_dir.glob("arousal_*_en_local_*.json"))

        if not translated_files:
            logger.error("No translated segments files found")
            logger.error("Stage 5B (translation) must complete first")
            return False

        if not arousal_files:
            logger.error("No English arousal predictions found")
            logger.error("Stage 6B (English intensity) must complete first")
            return False

        logger.info(f"Processing with {len(config['models'])} English emotion models")

        # Process with each model
        all_results = []
        for model_config in config["models"]:
            logger.info(f"\nProcessing with {model_config['name']}...")

            results = EmotionEN.process_video(
                video_dir=video_dir,
                model_config=_resolve_stage_model(model_config),
                force_reprocess=False,
            )
            all_results.extend(results)

        # Display aggregated results
        success_count = sum(1 for r in all_results if r["status"] == "success")
        skipped_count = sum(1 for r in all_results if r["status"] == "skipped")
        failed_count = sum(1 for r in all_results if r["status"] == "failed")

        logger.info("\nStage 7B complete:")
        logger.info(f"  Total model runs: {len(all_results)}")
        logger.info(f"  Processed: {success_count}")
        if skipped_count > 0:
            logger.info(f"  Skipped (already processed): {skipped_count}")
        if failed_count > 0:
            logger.warning(f"  Failed: {failed_count}")

        # Show statistics from first successful result
        for result in all_results:
            if result["status"] in ["success", "skipped"]:
                stats = result["result"]["statistics"]
                logger.info(f"  Model: {result['model']}")
                logger.info(f"    Total segments: {stats['total_segments']}")
                logger.info(f"    Emotionally significant: {stats['emotionally_significant']}")
                logger.info(f"    Low-arousal neutral: {stats['low_arousal_neutral']}")
                break

        return failed_count == 0

    except Exception as e:
        logger.error(f"Stage 7B failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_stage_8_visualization(video_dir: Path, config: Dict) -> bool:
    """
    Stage 8: Emotion timeline visualization

    Design: Multi-model ensemble validation with two-panel layout
    - Combines predictions from 3 models (2 English + 1 Russian)
    - Two-panel layout: arousal (top) + valence (bottom)
    - Agreement-based opacity for confidence visualization
    - Automatic quality analysis and error detection
    - Scene markers for temporal context

    Returns: True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 8: VISUALIZATION")
    logger.info("=" * 70)

    try:
        # Check prerequisites
        emotion_ru_files = list(video_dir.glob("emotion_*_local_*.json"))
        emotion_en_files = list(video_dir.glob("emotion_*_en_local_*.json"))

        if not emotion_ru_files:
            logger.error("No Russian emotion predictions found")
            logger.error("Stage 7A (Russian emotion) must complete first")
            return False

        if not emotion_en_files:
            logger.error("No English emotion predictions found")
            logger.error("Stage 7B (English emotion) must complete first")
            return False

        logger.info(f"Found {len(emotion_ru_files)} Russian emotion file(s)")
        logger.info(f"Found {len(emotion_en_files)} English emotion file(s)")
        logger.info(f"Minimum confidence: {config['min_confidence']}")

        # Process video
        result = VisualizationModule.process_video(
            video_dir=video_dir, min_confidence=config["min_confidence"]
        )

        if result["status"] == "success":
            qa = result["quality_analysis"]

            logger.info("\nStage 8 complete:")
            logger.info(f"  Total segments: {result['total_segments']}")
            logger.info(f"  Quality score: {qa['quality_score']:.0f}/100")
            logger.info(f"  Agreement rate: {qa['agreement']['rate_pct']:.0f}%")
            logger.info(f"  Average confidence: {result['average_confidence']:.3f}")

            if qa["warnings"]:
                logger.warning(f"  Quality warnings ({len(qa['warnings'])}):")
                for warning in qa["warnings"]:
                    logger.warning(f"    - {warning}")

            logger.info(f"  Output: {video_dir / config['output_filename']}")
            return True
        else:
            logger.error(f"Visualization failed: {result.get('reason', 'unknown')}")
            return False

    except Exception as e:
        logger.error(f"Stage 8 failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_stage_9_csv_export(video_dir: Path, config: Dict) -> bool:
    """
    Stage 9: CSV export (consolidated emotion data)

    Design: Consolidates all pipeline results into single CSV
    - Combines transcription, translation, VA predictions, emotions
    - Ensemble emotion calculation with agreement levels
    - One row per segment for easy analysis
    - Handles missing data gracefully

    Returns: True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 9: CSV EXPORT")
    logger.info("=" * 70)

    try:
        # Process video
        result = CSVExportModule.create_emotion_csv(
            video_dir=video_dir, output_filename=config["output_filename"]
        )

        if result["status"] == "success":
            logger.info("\nStage 9 complete:")
            logger.info(f"  Output: {result['output_file']}")
            logger.info(f"  Total rows: {result['total_rows']}")
            logger.info(
                f"  Complete rows: {result['complete_rows']} ({result['complete_percentage']:.1f}%)"
            )

            logger.info("  Agreement breakdown:")
            for level, count in result["agreement_breakdown"].items():
                pct = count / result["total_rows"] * 100
                logger.info(f"    {level}: {count} ({pct:.1f}%)")

            return True
        else:
            logger.error(f"CSV export failed: {result.get('reason', 'unknown')}")
            return False

    except Exception as e:
        logger.error(f"Stage 9 failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def main(youtube_url: str, configs: Optional[Dict] = None):
    """
    Main pipeline orchestrator - Complete (Stages 1-9)

    Sequential processing through all stages with skip logic at each stage.
    Models are loaded once and reused across processing to optimize performance.

    Args:
        youtube_url: YouTube video URL to process
        configs: Optional custom configurations (uses defaults if not provided)
    """
    # A deep copy: filling in devices below must not leak into DEFAULT_CONFIGS
    # or into a dict the caller still holds.
    configs = copy.deepcopy(DEFAULT_CONFIGS if configs is None else configs)
    device = apply_device_setting(configs)

    # Extract video ID for display
    video_id = extract_video_id_from_url(youtube_url)

    print("\n" + "=" * 70)
    print("VIDEO EMOTION ANALYSIS PIPELINE - COMPLETE")
    print("=" * 70)
    print(f"URL: {youtube_url}")
    if video_id:
        print(f"Video ID: {video_id}")
    print("\nProcessing stages: 1 → 2 → 3 → 4 → 5A → 5B → 6A → 6B → 7A → 7B → 8 → 9")
    print("=" * 70)

    # Load shared models once before processing
    # These models are reused across all stages for efficiency
    print("\n" + "=" * 70)
    print("INITIALIZING MODELS")
    print("=" * 70)

    # Stage 5A: Load semantic model for scene alignment
    # Note: Uses multilingual embeddings for Russian text understanding
    try:
        semantic_model = load_semantic_model("paraphrase-multilingual-MiniLM-L12-v2", device=device)
        if semantic_model:
            logger.info("Semantic model loaded (Stage 5A)")
        else:
            logger.warning("Semantic model not available - using word overlap fallback")
    except Exception as e:
        logger.warning(f"Failed to load semantic model: {e}")
        semantic_model = None

    # Stage 5B: Initialize translator
    # Note: This can take several minutes on first run (model download)
    translator = None
    try:
        logger.info("Initializing NLLB translator (Stage 5B)...")
        translator = NLLBTranslator(
            model_name=configs["stage_5b_translation"]["model_name"],
            preferred_gpu=configs["stage_5b_translation"].get("preferred_gpu"),
            device=configs["stage_5b_translation"].get("device") or device,
        )
        logger.info("Translator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize translator: {e}")
        logger.error("Stage 5B will be skipped")

    print("=" * 70)

    try:
        # Stage 1: Download
        video_dir = run_stage_1_download(youtube_url, configs["stage_1_download"])
        if not video_dir:
            logger.error("\nPipeline stopped: Stage 1 failed")
            return False

        # Stage 2: Scene detection
        success = run_stage_2_scene_detection(video_dir, configs["stage_2_scene_detection"])
        if not success:
            logger.error("\nPipeline stopped: Stage 2 failed")
            return False

        # Stage 3: Audio preprocessing
        success = run_stage_3_audio_preprocessing(video_dir, configs["stage_3_audio_preprocessing"])
        if not success:
            logger.error("\nPipeline stopped: Stage 3 failed")
            return False

        # Stage 4: Transcription
        success = run_stage_4_transcription(video_dir, configs["stage_4_transcription"])
        if not success:
            logger.error("\nPipeline stopped: Stage 4 failed")
            return False

        # Stage 5A: Scene alignment
        success = run_stage_5a_scene_alignment(
            video_dir, configs["stage_5a_scene_alignment"], semantic_model
        )
        if not success:
            logger.error("\nPipeline stopped: Stage 5A failed")
            return False

        # Stage 5B: Translation
        if translator:
            success = run_stage_5b_translation(
                video_dir, configs["stage_5b_translation"], translator
            )
            if not success:
                logger.error("\nPipeline stopped: Stage 5B failed")
                return False
        else:
            logger.warning("\nSkipping Stage 5B: Translator not initialized")
            logger.warning("Stages 6B and 7B will be skipped")

        # Stage 6A: Russian intensity classification
        success = run_stage_6a_russian_intensity(video_dir, configs["stage_6a_russian_intensity"])
        if not success:
            logger.error("\nPipeline stopped: Stage 6A failed")
            return False

        # Stage 6B: English intensity classification
        if translator:  # Only if translation succeeded
            success = run_stage_6b_english_intensity(
                video_dir, configs["stage_6b_english_intensity"]
            )
            if not success:
                logger.error("\nPipeline stopped: Stage 6B failed")
                return False
        else:
            logger.warning("\nSkipping Stage 6B: No translations available")

        # Stage 7A: Russian emotion classification
        success = run_stage_7a_russian_emotion(video_dir, configs["stage_7a_russian_emotion"])
        if not success:
            logger.error("\nPipeline stopped: Stage 7A failed")
            return False

        # Stage 7B: English emotion classification
        if translator:  # Only if translation succeeded
            success = run_stage_7b_english_emotion(video_dir, configs["stage_7b_english_emotion"])
            if not success:
                logger.error("\nPipeline stopped: Stage 7B failed")
                return False
        else:
            logger.warning("\nSkipping Stage 7B: No translations available")

        # Stage 8: Visualization
        # Note: Can work with partial data, but best with all 3 emotion models
        success = run_stage_8_visualization(video_dir, configs["stage_8_visualization"])
        if not success:
            logger.warning("\nStage 8 (Visualization) failed - continuing to CSV export")

        # Stage 9: CSV export
        success = run_stage_9_csv_export(video_dir, configs["stage_9_csv_export"])
        if not success:
            logger.warning("\nStage 9 (CSV export) failed")

        # Pipeline complete
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE (ALL STAGES)")
        print("=" * 70)
        print(f"Video directory: {video_dir}")
        print("\nGenerated files:")
        print("  Stage 1: metadata.json (video metadata)")
        print("  Stage 2: scene_boundaries.json (scene detection)")
        print("  Stage 3: preprocessed_audio.wav, audio_quality.json")
        print("  Stage 4: transcription_*.json (speech transcription)")
        print("  Stage 5A: global_scenes.json, local_segments_*.json")
        print("  Stage 5B: local_segments_*_translated.json")
        print("  Stage 6A: arousal_*_local_*.json (Russian VA)")
        print("  Stage 6B: arousal_*_en_local_*.json (English VA)")
        print("  Stage 7A: emotion_*_local_*.json (Russian emotion)")
        print("  Stage 7B: emotion_*_en_local_*.json (English emotion)")
        print("  Stage 8: emotion_timeline.png (visualization)")
        print("  Stage 9: emotion_analysis_data.csv (consolidated data)")
        print("\nAnalysis capabilities:")
        print("  - Visual timeline with emotion colors and intensity")
        print("  - Ensemble emotion validation across 3 models")
        print("  - Cross-language emotion comparison (Russian vs English)")
        print("  - Quality metrics and agreement analysis")
        print("  - CSV export for statistical analysis")
        print("=" * 70)

        return True

    finally:
        # Cleanup translator resources
        if translator:
            try:
                translator.cleanup()
                logger.info("Translator resources cleaned up")
            except Exception as e:
                logger.warning(f"Error during translator cleanup: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python -m vea.pipeline <youtube_url>  (prefer `uv run vea run`)")
        print("\nExample:")
        print('  python -m vea.pipeline "https://www.youtube.com/watch?v=mqGSkDFeLEo"')
        sys.exit(1)

    youtube_url = sys.argv[1]

    # Run pipeline with default configs
    success = main(youtube_url)

    # Exit with appropriate code
    sys.exit(0 if success else 1)
