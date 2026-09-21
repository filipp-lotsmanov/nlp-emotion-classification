"""
Speech Transcription Module with faster-whisper

Uses faster-whisper (CTranslate2 implementation) instead of openai-whisper.
Implements anti-hallucination best practices for long video processing.

Why faster-whisper over openai-whisper:
- 2-4x faster inference speed with same accuracy
- Built-in Silero VAD for filtering non-speech segments
- Lower memory footprint
- Same model architecture, just optimized backend

Anti-hallucination strategy for long videos:
1. VAD filtering: Removes silence/noise where Whisper tends to hallucinate
2. condition_on_previous_text=False: Prevents context-based hallucinations
3. Low temperature (0.0): Deterministic output, reduces random hallucinations
4. Quality monitoring: Flags segments with hallucination indicators
5. Conservative VAD parameters: Preserves speech edges while removing long silences

Key differences from openai-whisper:
- Returns generator of segments (memory efficient for long videos)
- Must iterate through segments to build output
- VAD parameters can be tuned for your specific audio characteristics
- Slightly different quality metrics structure
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

# faster-whisper uses different import
from faster_whisper import WhisperModel

# Logging is configured by the CLI (vea.cli) or by the calling application.
# Library modules must not call logging.basicConfig at import time.
logger = logging.getLogger(__name__)


def generate_transcription_filename(
    output_dir: Path, model_size: str, duration_limit: Union[str, int]
) -> Path:
    """
    Generate descriptive filename for transcription based on configuration.

    Same versioning system as before - enables comparing different configs.
    """
    filename = f"transcription_{model_size}"

    if duration_limit != "full":
        try:
            duration_seconds = int(duration_limit)
            filename += f"_{duration_seconds}s"
        except (ValueError, TypeError):
            pass

    filename += ".json"
    return output_dir / filename


def check_existing_transcription(
    output_path: Path, model_size: str, duration_limit: Union[str, int]
) -> Optional[Dict]:
    """
    Check if transcription with same config already exists.
    Returns existing transcription dict if found, None otherwise.
    """
    if not output_path.exists():
        return None

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

        metadata = existing.get("metadata", {})
        existing_model = metadata.get("model")
        existing_duration = metadata.get("duration_limit")

        current_duration = str(duration_limit)

        if existing_model == model_size and existing_duration == current_duration:
            return existing
        else:
            return None

    except Exception as e:
        logger.warning(f"Could not read existing transcription: {e}")
        return None


def transcribe_audio(
    audio_path: Path,
    output_path: Optional[Path] = None,
    model_size: str = "medium",
    language: str = "ru",
    duration_limit: Optional[Union[str, int]] = "full",
    device: Optional[str] = None,
    compute_type: str = "float16",
) -> Dict:
    """
    Transcribe Russian speech using faster-whisper with anti-hallucination measures.

    Why these specific parameters:

    1. vad_filter=True (CRITICAL for long videos)
       - Filters out silence and noise where Whisper hallucinates most
       - Especially important for videos >10 minutes
       - Research shows 80%+ reduction in hallucinations with VAD

    2. condition_on_previous_text=False
       - Prevents context-based hallucinations (ghost repetitions)
       - Common issue: model gets "stuck" repeating previous phrases
       - Trade-off: Slightly less coherent but much fewer hallucinations

    3. temperature=0.0
       - Deterministic output (no sampling randomness)
       - Reduces random hallucinations
       - For long videos, consistency > diversity

    4. VAD parameters tuned for speech preservation:
       - min_silence_duration_ms=2000: Conservative, waits for real silence
       - speech_pad_ms=400: Generous padding to avoid cutting speech edges
       - threshold=0.5: Balanced speech detection

    5. beam_size=5
       - Standard value, good quality/speed balance
       - Higher values (>5) don't significantly improve accuracy

    Args:
        audio_path: Path to preprocessed audio WAV file (16kHz mono)
        output_path: Path for output JSON (default: auto-generated)
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large', 'large-v3')
        language: Language code (default: 'ru' for Russian)
        duration_limit: 'full' or integer seconds to transcribe
        device: 'cuda', 'cpu', or None (auto-detect)
        compute_type: 'float16', 'int8', 'float32' (float16 recommended for GPU)

    Returns:
        Dictionary with transcription results in same format as openai-whisper
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Set output path with versioning
    if output_path is None:
        output_path = generate_transcription_filename(
            audio_path.parent, model_size=model_size, duration_limit=duration_limit
        )
    output_path = Path(output_path)

    # Check if already transcribed with same config
    existing = check_existing_transcription(output_path, model_size, duration_limit)
    if existing is not None:
        logger.info(f"Transcription already exists: {output_path.name}")
        logger.info(f"  Model: {existing['metadata']['model']}")
        logger.info(f"  Duration: {existing['metadata']['duration']:.1f}s")
        logger.info(f"  Words: {existing['statistics']['total_words']}")
        logger.info("Skipping transcription (delete file to re-run)")
        return existing

    logger.info(f"Transcribing audio: {audio_path.name}")

    # Auto-detect device if not specified
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # Adjust compute_type based on device
    # float16 only works on GPU, CPU needs int8 or float32
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"
        logger.info("Using int8 compute_type for CPU (float16 requires GPU)")

    logger.info(f"Model: {model_size}, Language: {language}, Device: {device}")
    logger.info(f"Compute type: {compute_type}, Duration limit: {duration_limit}")

    start_time = time.time()

    # Load faster-whisper model
    # Model is cached after first load
    logger.info(f"Loading faster-whisper {model_size} model...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    logger.info(f"Model loaded on {device}")

    # Load audio for duration calculation and optional trimming
    # faster-whisper can work directly with file path, but we need duration info
    import soundfile as sf

    audio_data, sample_rate = sf.read(audio_path, dtype="float32")

    original_duration = len(audio_data) / sample_rate
    logger.info(f"Original audio duration: {original_duration:.1f}s")

    # Trim audio if duration_limit specified
    # Write trimmed version to temp file since faster-whisper works with file paths
    audio_file_to_transcribe = audio_path
    trimmed_duration = original_duration

    if duration_limit != "full":
        try:
            duration_seconds = int(duration_limit)
            if duration_seconds > 0 and duration_seconds < original_duration:
                # Create temporary trimmed file
                max_samples = int(duration_seconds * sample_rate)
                trimmed_audio = audio_data[:max_samples]

                temp_path = audio_path.parent / f"temp_trimmed_{audio_path.name}"
                sf.write(temp_path, trimmed_audio, sample_rate)

                audio_file_to_transcribe = temp_path
                trimmed_duration = len(trimmed_audio) / sample_rate
                logger.info(f"Trimmed audio to first {trimmed_duration:.1f}s")
        except (ValueError, TypeError):
            logger.warning(f"Invalid duration_limit '{duration_limit}', using full audio")

    # Transcribe with anti-hallucination parameters
    logger.info("Transcribing with VAD filtering and anti-hallucination measures...")

    try:
        # faster-whisper.transcribe returns (segments_generator, transcription_info)
        segments_generator, info = model.transcribe(
            str(audio_file_to_transcribe),
            language=language,
            # CRITICAL: Enable VAD to filter non-speech
            # This is the single most important anti-hallucination measure
            vad_filter=True,
            # VAD parameters tuned for long videos
            # These are more conservative than defaults to preserve speech
            vad_parameters={
                "threshold": 0.5,  # Speech detection sensitivity (0-1)
                "min_speech_duration_ms": 250,  # Ignore very short sounds
                "min_silence_duration_ms": 2000,  # Wait 2s before splitting (default)
                "speech_pad_ms": 400,  # Padding around speech (default, prevents cutting)
            },
            # Anti-hallucination: Disable context conditioning
            # Prevents "ghost" repetitions from previous segments
            condition_on_previous_text=False,
            # Deterministic output, reduces randomness-based hallucinations
            temperature=0.0,
            # Standard quality parameters
            beam_size=5,
            # Enable word timestamps (critical for alignment)
            word_timestamps=True,
            # Quality thresholds
            compression_ratio_threshold=2.4,  # Detect repetitions
            log_prob_threshold=-1.0,  # Filter low confidence
            no_speech_threshold=0.6,  # Detect silence
        )

        # faster-whisper returns generator, must convert to list
        # Process segments and build output structure
        segments_list = []
        all_words = []

        logger.info("Processing transcription segments...")
        for segment in segments_generator:
            # Convert faster-whisper Segment to dict matching openai-whisper format
            segment_dict = {
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            }

            # Add word timestamps if available
            if segment.words:
                segment_dict["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ]
                all_words.extend(segment_dict["words"])
            else:
                segment_dict["words"] = []

            segments_list.append(segment_dict)

        # Combine all text
        full_text = " ".join(segment["text"].strip() for segment in segments_list)

    finally:
        # Clean up temporary trimmed file if created
        if audio_file_to_transcribe != audio_path and audio_file_to_transcribe.exists():
            audio_file_to_transcribe.unlink()

    elapsed = time.time() - start_time
    logger.info(f"Transcription complete in {elapsed:.1f}s")

    # Post-process transcription to add quality flags
    segments_list = post_process_transcription_segments(segments_list)

    # Calculate statistics
    total_words = sum(len(segment.get("words", [])) for segment in segments_list)

    # Calculate average confidence from word probabilities
    all_word_probs = [
        word.get("probability", 0.0)
        for segment in segments_list
        for word in segment.get("words", [])
    ]
    avg_confidence = float(np.mean(all_word_probs)) if all_word_probs else 0.0

    # Count low confidence and potential hallucination segments
    low_confidence_count = sum(
        1 for segment in segments_list if segment.get("avg_logprob", 0) < -0.5
    )

    high_compression_count = sum(
        1 for segment in segments_list if segment.get("compression_ratio", 1.0) > 2.4
    )

    high_no_speech_count = sum(
        1 for segment in segments_list if segment.get("no_speech_prob", 0) > 0.6
    )

    # Build output matching openai-whisper schema
    output = {
        "metadata": {
            "model": model_size,
            "language": language,
            "language_probability": float(info.language_probability),
            "duration": trimmed_duration,
            "duration_limit": str(duration_limit),
            "transcription_date": datetime.utcnow().isoformat() + "Z",
            "device_used": device,
            "compute_type": compute_type,
            # faster-whisper specific info
            "implementation": "faster-whisper",
            "vad_enabled": True,
            "condition_on_previous_text": False,
        },
        "text": full_text,
        "segments": segments_list,
        "statistics": {
            "total_words": total_words,
            "total_segments": len(segments_list),
            "average_confidence": avg_confidence,
            "low_confidence_segments": low_confidence_count,
            "high_compression_segments": high_compression_count,
            "high_no_speech_segments": high_no_speech_count,
            "words_with_timestamps": sum(1 for segment in segments_list if segment.get("words")),
        },
    }

    # Save transcription
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved transcription to {output_path.name}")
    logger.info(
        f"Statistics: {total_words} words, {len(segments_list)} segments, "
        f"avg confidence: {avg_confidence:.3f}"
    )

    # Log warnings about potential issues
    if low_confidence_count > 0:
        logger.warning(f"Found {low_confidence_count} low-confidence segments")
    if high_compression_count > 0:
        logger.warning(
            f"Found {high_compression_count} segments with high compression (possible repetitions)"
        )
    if high_no_speech_count > 0:
        logger.info(
            f"Found {high_no_speech_count} segments marked as likely silence "
            f"(VAD may have missed some)"
        )

    return output


def post_process_transcription_segments(segments: List[Dict]) -> List[Dict]:
    """
    Add quality flags to segments based on hallucination indicators.

    Flags help identify segments that may need review:
    - likely_silence: High no_speech_prob
    - low_confidence: Low avg_logprob
    - possible_repetition: High compression_ratio
    - no_word_timestamps: Missing word-level timing
    """
    for segment in segments:
        # Clean text
        if "text" in segment:
            segment["text"] = segment["text"].strip()

        # Add quality warnings
        warnings = []

        # Flag 1: Likely silence
        if segment.get("no_speech_prob", 0) > 0.6:
            warnings.append("likely_silence")

        # Flag 2: Low confidence
        # avg_logprob is negative, closer to 0 is better
        if segment.get("avg_logprob", 0) < -0.5:
            warnings.append("low_confidence")

        # Flag 3: High compression ratio (possible repetition/hallucination)
        if segment.get("compression_ratio", 1.0) > 2.4:
            warnings.append("possible_repetition")

        # Flag 4: Missing word timestamps
        if not segment.get("words") or len(segment.get("words", [])) == 0:
            warnings.append("no_word_timestamps")

        if warnings:
            segment["quality_warnings"] = warnings

    return segments


def validate_transcription(transcription: Dict) -> List[str]:
    """
    Validate transcription data for consistency.
    Same validation as openai-whisper version.
    """
    errors = []
    segments = transcription.get("segments", [])

    if not segments:
        errors.append("No segments in transcription")
        return errors

    for i, segment in enumerate(segments):
        # Check required fields
        required_fields = ["start", "end", "text"]
        for field in required_fields:
            if field not in segment:
                errors.append(f"Segment {i}: missing field '{field}'")

        # Check timestamp validity
        start = segment.get("start", 0)
        end = segment.get("end", 0)

        if start < 0:
            errors.append(f"Segment {i}: negative start time {start}")

        if end < start:
            errors.append(f"Segment {i}: end ({end}) before start ({start})")

        # Check temporal ordering
        if i > 0:
            prev_end = segments[i - 1].get("end", 0)
            if start < prev_end - 0.1:
                errors.append(
                    f"Segment {i}: starts before previous segment ends ({start} vs {prev_end})"
                )

        # Validate word timestamps if present
        if "words" in segment and segment["words"]:
            for j, word in enumerate(segment["words"]):
                w_start = word.get("start", 0)
                w_end = word.get("end", 0)

                # Words should be within segment bounds
                if w_start < start - 0.1 or w_end > end + 0.1:
                    errors.append(
                        f"Segment {i}, word {j}: timestamp outside segment bounds "
                        f"(word: {w_start}-{w_end}, segment: {start}-{end})"
                    )

                # Words should not overlap
                if j > 0:
                    prev_w_end = segment["words"][j - 1].get("end", 0)
                    if w_start < prev_w_end - 0.01:
                        errors.append(f"Segment {i}, word {j}: overlaps with previous word")

    return errors


def list_transcriptions(video_dir: Path) -> List[Dict]:
    """List all transcription files in a video directory."""
    video_dir = Path(video_dir)
    transcription_files = list(video_dir.glob("transcription_*.json"))

    transcriptions = []

    for trans_file in sorted(transcription_files):
        try:
            with open(trans_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            stats = data.get("statistics", {})

            transcriptions.append(
                {
                    "filename": trans_file.name,
                    "model": metadata.get("model", "unknown"),
                    "implementation": metadata.get("implementation", "unknown"),
                    "duration": metadata.get("duration", 0),
                    "duration_limit": metadata.get("duration_limit", "unknown"),
                    "words": stats.get("total_words", 0),
                    "segments": stats.get("total_segments", 0),
                    "avg_confidence": stats.get("average_confidence", 0.0),
                    "vad_enabled": metadata.get("vad_enabled", False),
                    "date": metadata.get("transcription_date", "unknown"),
                }
            )
        except Exception as e:
            logger.warning(f"Could not read {trans_file.name}: {e}")

    return transcriptions


def process_video(video_dir: Path, config: Dict = None) -> Dict:
    """
    Process transcription for a single video directory.
    Compatible with existing pipeline structure.
    """
    video_dir = Path(video_dir)

    # Load metadata
    metadata_path = video_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Find audio file (prefer preprocessed, fallback to original)
    preprocessed_audio = video_dir / "preprocessed_audio.wav"
    original_audio = video_dir / metadata["file_paths"].get("audio", "audio.wav")

    if preprocessed_audio.exists():
        audio_path = preprocessed_audio
        logger.info("Using preprocessed audio")
    elif original_audio.exists():
        audio_path = original_audio
        logger.warning("Preprocessed audio not found, using original")
    else:
        raise FileNotFoundError(f"No audio file found in {video_dir}")

    # Get configuration with defaults
    config = config or {}
    model_size = config.get("model_size", "medium")
    language = config.get("language", "ru")
    duration_limit = config.get("duration_limit", "full")
    device = config.get("device", None)
    compute_type = config.get("compute_type", "float16")

    # Transcribe
    logger.info(f"Transcribing audio in {video_dir.name}")

    result = transcribe_audio(
        audio_path=audio_path,
        model_size=model_size,
        language=language,
        duration_limit=duration_limit,
        device=device,
        compute_type=compute_type,
    )

    # Validate
    errors = validate_transcription(result)
    if errors:
        logger.warning(f"Validation found {len(errors)} issues:")
        for error in errors[:5]:
            logger.warning(f"  - {error}")
    else:
        logger.info("Validation: PASSED")

    logger.info(f"Transcription complete: {result['metadata']['model']}")

    return result


if __name__ == "__main__":
    """
    Test faster-whisper transcription with anti-hallucination measures.
    """
    import sys

    downloads_dir = Path("downloads")

    if not downloads_dir.exists():
        print("Downloads directory not found. Run downloader.py and audio_preprocessor.py first.")
        sys.exit(1)

    video_dirs = sorted(downloads_dir.glob("video-*"))

    if not video_dirs:
        print("No video directories found in downloads/")
        sys.exit(1)

    print(f"Found {len(video_dirs)} video(s) to process")
    print("=" * 60)

    # Configuration examples
    configs = [
        {
            "name": "production_medium",
            "model_size": "medium",
            "language": "ru",
            "duration_limit": "full",
            "device": None,
            "compute_type": "float16",
        },
        {
            "name": "quick_test_60s",
            "model_size": "small",
            "language": "ru",
            "duration_limit": 60,
            "device": None,
            "compute_type": "float16",
        },
        {
            "name": "high_accuracy",
            "model_size": "large-v3",
            "language": "ru",
            "duration_limit": "full",
            "device": None,
            "compute_type": "float16",
        },
        {
            "name": "high_test",
            "model_size": "large-v3",
            "language": "ru",
            "duration_limit": 600,
            "device": None,
            "compute_type": "float16",
        },
    ]

    # Choose which config to run
    active_configs = [2]

    print("Available configurations:")
    for i, cfg in enumerate(configs):
        status = "ACTIVE" if i in active_configs else "inactive"
        print(
            f"  [{i}] {cfg['name']}: {cfg['model_size']} model, "
            f"duration={cfg['duration_limit']} ({status})"
        )
    print("\nTo run different configs, modify 'active_configs' list in script")
    print("=" * 60)

    # Check device availability
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\nGPU detected: {gpu_name}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("\nNo GPU detected - using CPU (will be slower)")
    except ImportError:
        print("\nPyTorch not found - device detection unavailable")
    print("=" * 60)

    for video_dir in video_dirs:
        print(f"\n{'=' * 60}")
        print(f"Processing: {video_dir.name}")
        print("=" * 60)

        # Show existing transcriptions
        existing_transcriptions = list_transcriptions(video_dir)
        if existing_transcriptions:
            print(f"\nExisting transcriptions ({len(existing_transcriptions)}):")
            for trans in existing_transcriptions:
                print(f"  - {trans['filename']}")
                print(
                    f"    Model: {trans['model']} ({trans['implementation']}), "
                    f"Duration: {trans['duration']:.1f}s"
                )
                print(
                    f"    Words: {trans['words']}, Confidence: {trans['avg_confidence']:.3f}, "
                    f"VAD: {trans['vad_enabled']}"
                )
        else:
            print("\nNo existing transcriptions found")

        # Run active configurations
        for config_idx in active_configs:
            if config_idx >= len(configs):
                print(f"\nSkipping invalid config index {config_idx}")
                continue

            config = configs[config_idx]

            print(f"\n--- Running: {config['name']} ---")
            print(f"Model: {config['model_size']}, Duration: {config['duration_limit']}")
            print("VAD: ENABLED, condition_on_previous_text: FALSE")

            try:
                start_time = time.time()

                result = process_video(video_dir, config)

                elapsed = time.time() - start_time

                # Display summary
                stats = result["statistics"]
                meta = result["metadata"]

                print("  Success!")
                print(f"  Duration: {meta['duration']:.1f}s")
                print(f"  Segments: {stats['total_segments']}")
                print(f"  Words: {stats['total_words']}")
                print(f"  Avg confidence: {stats['average_confidence']:.3f}")

                # Show hallucination indicators
                if stats["low_confidence_segments"] > 0:
                    print(f"  Low confidence segments: {stats['low_confidence_segments']}")
                if stats["high_compression_segments"] > 0:
                    print(f"  Possible repetitions: {stats['high_compression_segments']}")
                if stats["high_no_speech_segments"] > 0:
                    print(f"  Likely silence segments: {stats['high_no_speech_segments']}")

                print(
                    f"  Processing time: {elapsed:.1f}s ({meta['duration'] / elapsed:.1f}x realtime)"
                )

                # Show output filename
                expected_filename = generate_transcription_filename(
                    video_dir, config["model_size"], config["duration_limit"]
                )
                print(f"  Saved to: {expected_filename.name}")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 60)
    print("ANTI-HALLUCINATION MEASURES APPLIED:")
    print("=" * 60)
    print("1. VAD filtering enabled (removes silence/noise)")
    print("2. condition_on_previous_text=False (prevents ghost repetitions)")
    print("3. temperature=0.0 (deterministic output)")
    print("4. Conservative VAD parameters (preserves speech edges)")
    print("5. Quality monitoring (flags suspicious segments)")
    print("\nFor best results on long videos, review segments with quality warnings.")
