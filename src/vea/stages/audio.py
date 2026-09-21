"""
Audio Preprocessing Module

This module prepares audio for optimal Whisper transcription by normalizing
volume and optionally reducing noise. Since the downloader already converts
to 16kHz mono WAV, we focus on quality improvements rather than format conversion.

Design decisions:
- RMS normalization: Simple, effective, preserves dynamic range better than peak
- Conservative noise reduction: Aggressive settings can distort speech
- Quality metrics: Help debug transcription issues downstream
- Skip resampling: Already done in downloader at 16kHz (Whisper's native rate)
- Skip existing: Avoid redundant processing by checking if audio already preprocessed
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

# Logging is configured by the CLI (vea.cli) or by the calling application.
# Library modules must not call logging.basicConfig at import time.
logger = logging.getLogger(__name__)


def check_if_preprocessed(video_dir: Path, config: Dict) -> bool:
    """
    Check if audio is already preprocessed with matching configuration.

    This prevents redundant processing while ensuring we reprocess if
    configuration changed (e.g., different normalization target).

    Design choice: Store config in quality file for validation
    - Alternative: Could use filename versioning like transcriber.py
    - Current approach chosen for: Simplicity, single canonical preprocessed file
    - Trade-off: Must reprocess if config changes (acceptable since preprocessing is fast)

    Args:
        video_dir: Path to video directory
        config: Current processing configuration

    Returns:
        True if already preprocessed with same config, False otherwise
    """
    video_dir = Path(video_dir)

    # Check if output files exist
    preprocessed_audio = video_dir / "preprocessed_audio.wav"
    quality_file = video_dir / "audio_quality.json"

    if not preprocessed_audio.exists() or not quality_file.exists():
        return False

    # Load existing quality metrics to check config
    try:
        with open(quality_file, "r", encoding="utf-8") as f:
            existing_quality = json.load(f)

        # Extract stored configuration
        stored_config = existing_quality.get("processing_config", {})

        # Compare with current config
        # Only check parameters that affect output
        config_matches = (
            stored_config.get("normalize") == config.get("normalize", True)
            and stored_config.get("target_db") == config.get("target_db", -12.0)
            and stored_config.get("reduce_noise") == config.get("reduce_noise", False)
            and stored_config.get("noise_strength") == config.get("noise_strength", 0.5)
        )

        if config_matches:
            return True
        else:
            logger.info("Existing preprocessed audio found but config differs")
            logger.info(f"  Stored: {stored_config}")
            logger.info("  Current: config")
            return False

    except Exception as e:
        logger.warning(f"Could not read existing quality file: {e}")
        return False


def preprocess_audio(
    audio_path: Path,
    output_path: Optional[Path] = None,
    normalize: bool = True,
    target_db: float = -12.0,
    reduce_noise: bool = False,
    noise_reduce_strength: float = 0.5,
) -> Path:
    """
    Complete audio preprocessing pipeline for speech recognition.

    Args:
        audio_path: Path to input audio WAV file
        output_path: Path for preprocessed audio (default: same dir, preprocessed_audio.wav)
        normalize: Apply RMS normalization (recommended: True)
        target_db: Target RMS level in dB (default: -12.0, good for speech)
        reduce_noise: Apply noise reduction (default: False, use only for noisy audio)
        noise_reduce_strength: Noise reduction strength 0.0-1.0 (default: 0.5, conservative)

    Returns:
        Path to preprocessed audio file

    Notes:
        - Input audio should already be 16kHz mono WAV (done by downloader)
        - RMS normalization prevents clipping while maintaining dynamics
        - Noise reduction is optional; skip if audio is already clean
        - Too much noise reduction can distort speech features
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Set output path
    if output_path is None:
        output_path = audio_path.parent / "preprocessed_audio.wav"
    output_path = Path(output_path)

    logger.info(f"Preprocessing audio: {audio_path.name}")
    start_time = time.time()

    # Load audio
    # Using soundfile instead of librosa.load because it's faster and we don't need resampling
    audio, sample_rate = sf.read(audio_path, dtype="float32")

    logger.info(f"Loaded audio: {len(audio) / sample_rate:.1f}s @ {sample_rate}Hz")

    # Verify sample rate
    if sample_rate != 16000:
        logger.warning(f"Expected 16kHz audio but got {sample_rate}Hz. Resampling to 16kHz...")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Store original for quality comparison
    original_rms = calculate_rms_db(audio)
    original_peak = np.max(np.abs(audio))

    # Step 1: Normalize audio volume
    if normalize:
        logger.info(f"Normalizing audio to {target_db} dB RMS...")
        audio = normalize_audio(audio, target_db=target_db)
        normalized_rms = calculate_rms_db(audio)
        logger.info(f"RMS: {original_rms:.1f} dB -> {normalized_rms:.1f} dB")
    else:
        normalized_rms = original_rms

    # Step 2: Optional noise reduction
    # Only use if audio is noticeably noisy, as it can introduce artifacts
    if reduce_noise:
        logger.info(f"Applying noise reduction (strength: {noise_reduce_strength})...")
        try:
            audio = reduce_noise_spectral(audio, sample_rate, strength=noise_reduce_strength)
            logger.info("Noise reduction applied")
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}. Continuing without it.")

    # Step 3: Detect silence regions
    silence_regions = detect_silence(audio, sample_rate, threshold_db=-40.0)
    total_silence = sum(end - start for start, end in silence_regions)

    logger.info(f"Detected {len(silence_regions)} silence regions ({total_silence:.1f}s total)")

    # Step 4: Quality checks
    quality_flags = check_audio_quality(audio, sample_rate)

    if quality_flags:
        logger.warning(f"Quality issues detected: {', '.join(quality_flags)}")
    else:
        logger.info("Audio quality checks passed")

    # Step 5: Save preprocessed audio
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")

    elapsed = time.time() - start_time
    logger.info(f"Preprocessing complete in {elapsed:.1f}s: {output_path.name}")

    # Step 6: Save quality metrics with processing config
    # Store config so we can validate if existing file matches requested processing
    quality_metrics = {
        "original_sample_rate": sample_rate,
        "target_sample_rate": 16000,
        "duration_seconds": float(len(audio) / sample_rate),
        "channels": 1,
        "processing_config": {
            "normalize": normalize,
            "target_db": target_db,
            "reduce_noise": reduce_noise,
            "noise_strength": noise_reduce_strength,
        },
        "normalization": {
            "applied": normalize,
            "method": "rms",
            "target_db": target_db,
            "original_rms_db": float(original_rms),
            "normalized_rms_db": float(normalized_rms),
            "original_peak": float(original_peak),
            "final_peak": float(np.max(np.abs(audio))),
        },
        "noise_reduction": {
            "applied": reduce_noise,
            "method": "spectral_gating" if reduce_noise else None,
            "strength": noise_reduce_strength if reduce_noise else None,
        },
        "silence_regions": [
            {"start": float(start), "end": float(end)} for start, end in silence_regions
        ],
        "silence_total_seconds": float(total_silence),
        "quality_flags": quality_flags,
        "processing_date": datetime.utcnow().isoformat() + "Z",
    }

    quality_path = audio_path.parent / "audio_quality.json"
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_metrics, f, indent=2)

    logger.info(f"Saved quality metrics to {quality_path.name}")

    return output_path


def normalize_audio(audio: np.ndarray, target_db: float = -12.0) -> np.ndarray:
    """
    Normalize audio using RMS (Root Mean Square) method.

    Why RMS instead of peak normalization:
    - Peak normalization: max(abs(audio)) -> 1.0
      Problem: Amplifies noise in quiet passages, can still clip on transients
    - RMS normalization: Adjusts average loudness
      Benefit: Preserves dynamic range, consistent perceived loudness

    Alternative considered: LUFS (Loudness Units Full Scale)
    - More sophisticated perceptual loudness matching
    - Rejected for now: Added complexity, RMS sufficient for speech

    Args:
        audio: Audio samples as numpy array
        target_db: Target RMS level in decibels (default: -12.0)
                   Typical values: -12 to -18 dB for speech

    Returns:
        Normalized audio array
    """
    # Calculate current RMS in dB
    current_rms_db = calculate_rms_db(audio)

    # Calculate gain needed (in dB)
    gain_db = target_db - current_rms_db

    # Convert dB gain to linear gain
    gain_linear = 10 ** (gain_db / 20)

    # Apply gain
    normalized = audio * gain_linear

    # Safety check: prevent clipping
    # If normalized audio would clip, reduce gain proportionally
    peak = np.max(np.abs(normalized))
    if peak > 0.99:  # Leave 1% headroom
        safety_gain = 0.99 / peak
        normalized = normalized * safety_gain
        logger.debug(f"Applied safety gain {safety_gain:.3f} to prevent clipping")

    return normalized


def calculate_rms_db(audio: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) level in decibels.

    RMS = sqrt(mean(audio^2))
    dB = 20 * log10(RMS)

    Reference level: 1.0 = 0 dB (full scale)
    """
    rms = np.sqrt(np.mean(audio**2))

    # Avoid log(0) which would give -inf
    if rms < 1e-10:
        return -100.0  # Essentially silence

    rms_db = 20 * np.log10(rms)
    return rms_db


def reduce_noise_spectral(audio: np.ndarray, sample_rate: int, strength: float = 0.5) -> np.ndarray:
    """
    Apply spectral noise reduction using noise gate approach.

    This is a simple spectral gating method:
    1. Estimate noise floor from quiet portions
    2. Suppress frequencies below threshold

    Alternative approaches:
    - Wiener filtering: More sophisticated, computationally expensive
    - Deep learning denoising: Best quality, requires GPU and large models
    - Spectral subtraction: Can introduce "musical noise" artifacts

    Current approach chosen for: Simplicity, speed, minimal artifacts

    Warning: Aggressive noise reduction can distort speech!
    Recommendation: Only use on noticeably noisy recordings

    Args:
        audio: Audio samples
        sample_rate: Sample rate in Hz
        strength: Noise reduction strength 0.0 (none) to 1.0 (aggressive)

    Returns:
        Noise-reduced audio
    """
    # Try using noisereduce library if available
    # If not installed, skip noise reduction
    try:
        import noisereduce as nr

        # Reduce noise using stationary noise profile
        # Conservative settings to avoid speech distortion
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=True,  # Assume noise is constant
            prop_decrease=strength,  # How much to reduce (0-1)
        )

        return reduced

    except ImportError:
        logger.warning("noisereduce library not installed. Install with: pip install noisereduce")
        logger.warning("Skipping noise reduction")
        return audio


def detect_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.3,
) -> List[Tuple[float, float]]:
    """
    Detect continuous silence regions in audio.

    Silence detection helps with:
    - Validating audio quality (too much silence = problem)
    - Understanding speech patterns
    - Potential segment boundaries

    Args:
        audio: Audio samples
        sample_rate: Sample rate in Hz
        threshold_db: Level below which audio is considered silence (default: -40 dB)
        min_silence_duration: Minimum silence length to report (default: 0.3s)

    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold_db / 20)

    # Calculate frame-level RMS with small window
    frame_length = int(0.05 * sample_rate)  # 50ms frames
    hop_length = frame_length // 2

    # Calculate RMS for each frame
    rms_frames = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Identify silent frames
    is_silent = rms_frames < threshold_linear

    # Find continuous silent regions
    silence_regions = []
    in_silence = False
    silence_start = 0

    for i, silent in enumerate(is_silent):
        time = i * hop_length / sample_rate

        if silent and not in_silence:
            # Start of silence
            in_silence = True
            silence_start = time
        elif not silent and in_silence:
            # End of silence
            silence_end = time
            silence_duration = silence_end - silence_start

            # Only record if longer than minimum
            if silence_duration >= min_silence_duration:
                silence_regions.append((silence_start, silence_end))

            in_silence = False

    # Handle case where audio ends in silence
    if in_silence:
        silence_end = len(audio) / sample_rate
        silence_duration = silence_end - silence_start
        if silence_duration >= min_silence_duration:
            silence_regions.append((silence_start, silence_end))

    return silence_regions


def check_audio_quality(audio: np.ndarray, sample_rate: int) -> List[str]:
    """
    Check for common audio quality issues.

    Returns list of warning strings, empty if no issues detected.

    Checks:
    - Clipping: Audio samples at maximum value
    - DC offset: Non-zero mean (can cause issues)
    - Low level: Audio too quiet
    - High noise floor: Likely noisy recording
    """
    issues = []

    # Check for clipping
    peak = np.max(np.abs(audio))
    if peak > 0.99:
        issues.append(f"clipping_detected_peak_{peak:.3f}")

    # Check for DC offset
    mean = np.mean(audio)
    if abs(mean) > 0.01:
        issues.append(f"dc_offset_{mean:.4f}")

    # Check if audio is too quiet
    rms_db = calculate_rms_db(audio)
    if rms_db < -40:
        issues.append(f"audio_too_quiet_rms_{rms_db:.1f}dB")

    # Estimate SNR (Signal to Noise Ratio)
    # Simple method: compare RMS of loudest vs quietest parts
    frame_length = int(0.1 * sample_rate)  # 100ms frames

    # Calculate RMS for each frame
    n_frames = len(audio) // frame_length
    if n_frames > 10:  # Need enough frames for estimate
        frame_rms = []
        for i in range(n_frames):
            start = i * frame_length
            end = start + frame_length
            frame = audio[start:end]
            frame_rms.append(np.sqrt(np.mean(frame**2)))

        # Sort to get percentiles
        frame_rms_sorted = np.sort(frame_rms)

        # Noise floor estimate: 10th percentile (quiet parts)
        noise_floor = frame_rms_sorted[len(frame_rms_sorted) // 10]

        # Signal level estimate: 90th percentile (loud parts)
        signal_level = frame_rms_sorted[int(len(frame_rms_sorted) * 0.9)]

        # Calculate SNR in dB
        if noise_floor > 1e-10 and signal_level > noise_floor:
            snr_db = 20 * np.log10(signal_level / noise_floor)

            if snr_db < 20:
                issues.append(f"low_snr_{snr_db:.1f}dB")

    return issues


def process_video(video_dir: Path, config: Dict = None, force_reprocess: bool = False) -> Dict:
    """
    Process audio for a single video directory.

    This function now checks if audio is already preprocessed with the same
    configuration and skips processing if so, similar to downloader behavior.

    Design choice: Skip existing vs always reprocess
    - Always reprocess: Wastes time on redundant work, may overwrite wanted results
    - Skip existing: Fast, preserves existing data, prevents accidental overwrites
    - Current choice: Skip existing (with force flag for explicit reprocessing)

    Why check config match:
    - Different normalization targets produce different output
    - Need to reprocess if user changed settings
    - Store config in quality file for validation

    Args:
        video_dir: Path to video-N directory
        config: Processing configuration (optional)
        force_reprocess: If True, reprocess even if already done (default: False)

    Returns:
        Quality metrics dictionary
    """
    video_dir = Path(video_dir)

    # Load metadata to check audio file exists
    metadata_path = video_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Find audio file
    audio_file = metadata["file_paths"].get("audio")
    if not audio_file:
        raise ValueError("Audio file path not in metadata")

    audio_path = video_dir / audio_file
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Get configuration with defaults
    config = config or {}
    normalize = config.get("normalize", True)
    target_db = config.get("target_db", -12.0)
    reduce_noise = config.get("reduce_noise", False)
    noise_strength = config.get("noise_strength", 0.5)

    # Build config dict for comparison
    current_config = {
        "normalize": normalize,
        "target_db": target_db,
        "reduce_noise": reduce_noise,
        "noise_strength": noise_strength,
    }

    # Check if already preprocessed with same config
    if not force_reprocess and check_if_preprocessed(video_dir, current_config):
        logger.info(f"Audio already preprocessed in {video_dir.name} - SKIPPING")
        logger.info("  (Use force_reprocess=True to reprocess)")

        # Load and return existing quality metrics
        quality_path = video_dir / "audio_quality.json"
        with open(quality_path, "r", encoding="utf-8") as f:
            quality_metrics = json.load(f)

        logger.info(f"  Duration: {quality_metrics['duration_seconds']:.1f}s")
        logger.info(f"  RMS: {quality_metrics['normalization']['normalized_rms_db']:.1f} dB")

        return quality_metrics

    # Process audio
    logger.info(f"Processing audio in {video_dir.name}")

    if force_reprocess:
        logger.info("Force reprocess enabled - will overwrite existing files")

    output_path = preprocess_audio(
        audio_path=audio_path,
        normalize=normalize,
        target_db=target_db,
        reduce_noise=reduce_noise,
        noise_reduce_strength=noise_strength,
    )

    # Load quality metrics
    quality_path = video_dir / "audio_quality.json"
    with open(quality_path, "r", encoding="utf-8") as f:
        quality_metrics = json.load(f)

    logger.info(f"Audio preprocessing complete: {output_path.name}")

    return quality_metrics


if __name__ == "__main__":
    """
    Test audio preprocessing on downloaded videos.
    
    Now includes skip logic demonstration:
    - First run: Processes all videos
    - Second run: Skips already-processed videos
    - With force_reprocess=True: Reprocesses everything
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
    print("=" * 60)

    # Configuration
    config = {
        "normalize": True,
        "target_db": -12.0,
        "reduce_noise": False,  # Set True only for noisy audio
        "noise_strength": 0.5,
    }

    # Control reprocessing behavior
    force_reprocess = False  # Set True to force reprocess all

    print("Configuration:")
    print(f"  Normalize: {config['normalize']} (target: {config['target_db']} dB)")
    print(f"  Noise reduction: {config['reduce_noise']}")
    if config["reduce_noise"]:
        print(f"  Noise strength: {config['noise_strength']}")
    print(f"  Force reprocess: {force_reprocess}")
    print("=" * 60)

    # Track statistics
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for video_dir in video_dirs:
        try:
            print(f"\nProcessing: {video_dir.name}")
            start_time = time.time()

            quality_metrics = process_video(video_dir, config, force_reprocess=force_reprocess)

            elapsed = time.time() - start_time

            # Check if was skipped (fast processing time indicates skip)
            was_skipped = elapsed < 0.5 and "processing_config" in quality_metrics

            if was_skipped:
                skipped_count += 1
            else:
                processed_count += 1

            # Display summary
            norm = quality_metrics["normalization"]
            print(f"  Duration: {quality_metrics['duration_seconds']:.1f}s")
            print(f"  RMS: {norm['normalized_rms_db']:.1f} dB")
            print(f"  Peak level: {norm['final_peak']:.3f}")
            print(
                f"  Silence: {quality_metrics['silence_total_seconds']:.1f}s "
                f"({len(quality_metrics['silence_regions'])} regions)"
            )

            if quality_metrics["quality_flags"]:
                print(f"  Quality issues: {', '.join(quality_metrics['quality_flags'])}")
            else:
                print("  Quality: OK")

            if was_skipped:
                print("  Status: SKIPPED (already processed)")
            else:
                print(f"  Processing time: {elapsed:.1f}s")

        except Exception as e:
            failed_count += 1
            print(f"  ERROR: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Total videos: {len(video_dirs)}")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")

    if skipped_count > 0 and not force_reprocess:
        print("\nSkipped videos are already preprocessed.")
        print("  Use force_reprocess=True to reprocess all")
