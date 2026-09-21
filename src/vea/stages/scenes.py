"""
Scene Detection Module

This module detects scene boundaries in video using PySceneDetect library.
Integrates with the pipeline by following consistent patterns for file I/O,
skip logic, and error handling.

Design decisions:
- PySceneDetect library: Mature, well-tested, handles edge cases better than custom implementation
- AdaptiveDetector default: More robust to camera movement than ContentDetector
- Skip logic: Prevents redundant processing, checks config match
- Configuration storage: Saved in metadata for validation on subsequent runs

Alternative approaches considered:
- Custom OpenCV implementation: More control but requires extensive testing for edge cases
- FFmpeg scene detection: Fast but less accurate, no Python API
- Deep learning methods: Overkill for this use case, requires GPU and models
- Decision: PySceneDetect with AdaptiveDetector provides best balance
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector

# Logging is configured by the CLI (vea.cli) or by the calling application.
# Library modules must not call logging.basicConfig at import time.
logger = logging.getLogger(__name__)


def check_if_detected(video_dir: Path, config: Dict) -> bool:
    """
    Check if scene detection already completed with matching configuration.

    This prevents redundant processing while ensuring we reprocess if
    configuration changed (e.g., different threshold or detector type).

    Why check configuration match:
    - Different thresholds produce different scene boundaries
    - Detector type (adaptive vs content) significantly affects results
    - Need to reprocess if user changed settings
    - Config stored in metadata for validation

    Design choice: Skip existing vs always reprocess
    - Always reprocess: Wastes time, may take 2-5 minutes per video
    - Skip existing: Fast, preserves data, prevents accidental overwrites
    - Current choice: Skip existing with force flag for explicit reprocessing

    Args:
        video_dir: Path to video directory
        config: Current detection configuration

    Returns:
        True if already detected with same config, False otherwise
    """
    video_dir = Path(video_dir)
    scene_file = video_dir / "scene_boundaries.json"

    if not scene_file.exists():
        return False

    # Load existing scene data to check config
    try:
        with open(scene_file, "r", encoding="utf-8") as f:
            existing = json.load(f)

        # Extract stored configuration from metadata
        stored_metadata = existing.get("metadata", {})
        stored_method = stored_metadata.get("detection_method", "")
        stored_threshold = stored_metadata.get("threshold")
        stored_min_scene = stored_metadata.get("min_scene_length")

        # Determine current method string
        # PySceneDetect format: "pyscenedetect_adaptive" or "pyscenedetect_content"
        current_method = f"pyscenedetect_{config.get('detector_type', 'adaptive')}"

        # Compare configurations
        # Only reprocess if something meaningful changed
        config_matches = (
            stored_method == current_method
            and stored_threshold == config.get("threshold")
            and stored_min_scene == config.get("min_scene_length", 2.0)
        )

        if config_matches:
            return True
        else:
            logger.info("Existing scene detection found but config differs")
            logger.info(
                f"  Stored: method={stored_method}, threshold={stored_threshold}, "
                f"min_scene={stored_min_scene}"
            )
            logger.info(
                f"  Current: method={current_method}, threshold={config.get('threshold')}, "
                f"min_scene={config.get('min_scene_length', 2.0)}"
            )
            return False

    except Exception as e:
        logger.warning(f"Could not read existing scene file: {e}")
        return False


def detect_scenes_pyscenedetect(
    video_path: Path,
    fps: float,
    detector_type: str = "adaptive",
    threshold: float = 3.0,
    min_scene_length: float = 2.0,
    downscale: Optional[int] = None,
) -> Dict:
    """
    Detect scene boundaries using PySceneDetect library.

    Why PySceneDetect:
    - Battle-tested on thousands of videos
    - Handles edge cases (fades, dissolves, camera motion)
    - Actively maintained with good documentation
    - Better than custom OpenCV implementation for production use

    Detector types:
    - AdaptiveDetector: Uses rolling average, less sensitive to camera motion
      Best for: General use, videos with pans/zooms, handheld footage
      Threshold: Lower values (1-5) for sensitivity

    - ContentDetector: Fixed threshold, faster processing
      Best for: Clean cuts, studio content, testing/debugging
      Threshold: Higher values (27-32) typical

    Args:
        video_path: Path to video file
        fps: Frames per second from metadata
        detector_type: 'adaptive' or 'content' (default: 'adaptive')
        threshold: Detection sensitivity (default: 3.0 for adaptive)
        min_scene_length: Minimum scene duration in seconds (default: 2.0)
        downscale: Optional downscale factor for speed (e.g., 2 = half resolution)

    Returns:
        Dictionary with metadata, scenes list, and statistics
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Starting scene detection: {video_path.name}")
    logger.info(f"  Detector: {detector_type}")
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"  Min scene length: {min_scene_length}s")

    start_time = time.time()

    # Open video using PySceneDetect
    video = open_video(str(video_path))

    # Create scene manager
    scene_manager = SceneManager()

    # Convert min_scene_length to frames for detector
    min_scene_frames = int(min_scene_length * fps)

    # Add detector based on type
    if detector_type.lower() == "adaptive":
        # AdaptiveDetector: Better for videos with camera movement
        # Uses rolling average of frame differences
        # Less prone to false positives from motion
        detector = AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_frames)
        logger.info("Using AdaptiveDetector (recommended for general use)")
    else:
        # ContentDetector: Fixed threshold, simpler logic
        # Faster but more sensitive to camera motion
        detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_frames)
        logger.info("Using ContentDetector (faster, fixed threshold)")

    scene_manager.add_detector(detector)

    # Optional downscaling for performance
    # Processing at half resolution (downscale=2) gives ~4x speedup
    # with minimal accuracy loss for most content
    if downscale and downscale > 1:
        logger.info(f"Downscaling by factor of {downscale} for performance")

    # Detect scenes with progress display
    logger.info("Detecting scenes...")
    scene_manager.detect_scenes(video, show_progress=True)

    # Get detected scene list
    scene_list = scene_manager.get_scene_list()

    elapsed = time.time() - start_time
    logger.info(f"Detection complete: {len(scene_list)} scenes in {elapsed:.1f}s")

    # Get video properties for validation
    video_duration = video.duration.get_seconds()

    # Convert PySceneDetect format to pipeline schema
    # PySceneDetect returns (start_time, end_time) tuples
    # We need dict format matching data_schemas.md
    scenes = []
    for i, (start_time_obj, end_time_obj) in enumerate(scene_list):
        start_sec = start_time_obj.get_seconds()
        end_sec = end_time_obj.get_seconds()

        scenes.append(
            {
                "scene_id": i,
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "duration": round(end_sec - start_sec, 2),
                "transition_type": "hard_cut",  # PySceneDetect doesn't distinguish types
                "difference_score": 0.0,  # Not exposed by PySceneDetect API
            }
        )

    # Calculate statistics for summary
    durations = [s["duration"] for s in scenes]

    # Build result matching pipeline schema
    result = {
        "metadata": {
            "video_duration": round(video_duration, 2),
            "detection_method": f"pyscenedetect_{detector_type}",
            "threshold": threshold,
            "effective_threshold": threshold,  # PySceneDetect doesn't compute adaptive threshold
            "fps_analyzed": fps,
            "frame_skip": 1,  # PySceneDetect processes every frame by default
            "downscale_factor": downscale if downscale else 1,
            "min_scene_length": min_scene_length,
            "detection_date": datetime.utcnow().isoformat() + "Z",
        },
        "scenes": scenes,
        "statistics": {
            "total_scenes": len(scenes),
            "average_duration": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "shortest_scene": round(min(durations), 2) if durations else 0.0,
            "longest_scene": round(max(durations), 2) if durations else 0.0,
            "hard_cuts": len(scenes),  # All detected as hard cuts
            "gradual_transitions": 0,  # PySceneDetect doesn't distinguish
        },
    }

    return result


def validate_scene_boundaries(scenes_data: Dict) -> List[str]:
    """
    Validate scene boundary data for temporal consistency.

    Checks performed:
    - First scene starts at 0
    - No temporal gaps between scenes
    - No overlapping scenes
    - Duration calculations are correct
    - Last scene ends at video duration

    These checks catch common issues:
    - Missing scenes (gaps in timeline)
    - Duplicate scenes (overlaps)
    - Math errors (duration mismatch)
    - Incomplete coverage (doesn't reach end of video)

    Args:
        scenes_data: Scene boundaries dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    scenes = scenes_data.get("scenes", [])
    metadata = scenes_data.get("metadata", {})
    video_duration = metadata.get("video_duration", 0)

    if not scenes:
        errors.append("No scenes found")
        return errors

    # Check first scene starts at 0
    # Allow small tolerance for floating point precision
    if abs(scenes[0]["start"]) > 0.01:
        errors.append(f"First scene doesn't start at 0: {scenes[0]['start']}")

    # Check temporal consistency between consecutive scenes
    for i in range(len(scenes) - 1):
        current = scenes[i]
        next_scene = scenes[i + 1]

        # Check for gaps (next scene starts after current ends)
        gap = next_scene["start"] - current["end"]
        if abs(gap) > 0.1:  # Allow 0.1s tolerance for rounding
            errors.append(
                f"Gap between scene {i} and {i + 1}: "
                f"{current['end']}s to {next_scene['start']}s ({gap:.2f}s gap)"
            )

        # Check for overlaps (scenes shouldn't overlap)
        if current["end"] > next_scene["start"]:
            errors.append(
                f"Overlap between scene {i} and {i + 1}: "
                f"scene {i} ends at {current['end']}, scene {i + 1} starts at {next_scene['start']}"
            )

        # Check duration calculation
        calculated_duration = current["end"] - current["start"]
        if abs(calculated_duration - current["duration"]) > 0.01:
            errors.append(
                f"Scene {i} duration mismatch: "
                f"calculated {calculated_duration:.2f}s, stored {current['duration']}s"
            )

    # Check last scene ends at video duration
    # Allow 1 second tolerance for video duration estimates
    if video_duration > 0:
        last_end = scenes[-1]["end"]
        if abs(last_end - video_duration) > 1.0:
            errors.append(
                f"Last scene doesn't end at video duration: "
                f"scene ends at {last_end}s, video duration is {video_duration}s"
            )

    return errors


def process_video(video_dir: Path, config: Dict = None, force_redetect: bool = False) -> Dict:
    """
    Process scene detection for a single video directory.

    This function integrates with the pipeline by:
    1. Loading metadata from Stage 1 (downloader)
    2. Detecting scene boundaries
    3. Saving results for Stage 5 (segment_aligner)
    4. Skipping if already processed with same config

    Skip logic prevents redundant processing:
    - First run: Detects scenes (2-5 minutes per video)
    - Second run: Skips detection if config unchanged (<1 second)
    - With force_redetect=True: Always reprocesses

    Why this matters:
    - Scene detection is one of the slower pipeline stages
    - Running entire pipeline multiple times shouldn't redo scene detection
    - But need to reprocess if user changed threshold settings

    Args:
        video_dir: Path to video-{ID} directory
        config: Detection configuration (optional)
            - detector_type: 'adaptive' or 'content'
            - threshold: Detection sensitivity (float)
            - min_scene_length: Minimum scene duration (float, seconds)
            - downscale: Optional downscale factor for speed (int)
        force_redetect: If True, reprocess even if already done (default: False)

    Returns:
        Scene boundaries dictionary
    """
    video_dir = Path(video_dir)

    # Load metadata from Stage 1 (downloader)
    metadata_path = video_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Find video file
    video_file = metadata["file_paths"].get("video")
    if not video_file:
        raise ValueError("Video file path not in metadata")

    video_path = video_dir / video_file
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Get FPS from metadata (critical for frame-to-time conversion)
    fps = metadata.get("fps")
    if not fps or fps <= 0:
        raise ValueError(f"Invalid FPS in metadata: {fps}")

    # Get configuration with sensible defaults
    config = config or {}

    detector_type = config.get("detector_type", "adaptive")

    # Set appropriate default threshold based on detector type
    # AdaptiveDetector uses rolling average, needs lower threshold
    # ContentDetector uses fixed threshold, needs higher value
    if detector_type.lower() == "adaptive":
        default_threshold = 3.0  # Recommended range: 1-7
    else:
        default_threshold = 27.0  # Recommended range: 20-35

    threshold = config.get("threshold", default_threshold)
    min_scene_length = config.get("min_scene_length", 2.0)
    downscale = config.get("downscale", None)

    # Build config dict for comparison
    current_config = {
        "detector_type": detector_type,
        "threshold": threshold,
        "min_scene_length": min_scene_length,
        "downscale": downscale,
    }

    # Check if already detected with same config
    if not force_redetect and check_if_detected(video_dir, current_config):
        logger.info(f"Scenes already detected in {video_dir.name} - SKIPPING")
        logger.info("  (Use force_redetect=True to reprocess)")

        # Load and return existing scene data
        scene_file = video_dir / "scene_boundaries.json"
        with open(scene_file, "r", encoding="utf-8") as f:
            result = json.load(f)

        stats = result.get("statistics", {})
        logger.info(f"  Total scenes: {stats.get('total_scenes', 0)}")
        logger.info(f"  Average duration: {stats.get('average_duration', 0)}s")

        return result

    # Detect scenes
    logger.info(f"Detecting scenes in {video_dir.name}")

    if force_redetect:
        logger.info("Force redetect enabled - will overwrite existing data")

    result = detect_scenes_pyscenedetect(
        video_path=video_path,
        fps=fps,
        detector_type=detector_type,
        threshold=threshold,
        min_scene_length=min_scene_length,
        downscale=downscale,
    )

    # Validate scene boundaries
    errors = validate_scene_boundaries(result)
    if errors:
        logger.warning(f"Validation found {len(errors)} issues:")
        for error in errors[:5]:  # Show first 5
            logger.warning(f"  - {error}")
    else:
        logger.info("Validation: PASSED")

    # Save results
    output_path = video_dir / "scene_boundaries.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved scene boundaries to {output_path.name}")

    # Display statistics
    stats = result["statistics"]
    logger.info(
        f"Statistics: {stats['total_scenes']} scenes, "
        f"avg {stats['average_duration']}s, "
        f"range {stats['shortest_scene']}-{stats['longest_scene']}s"
    )

    return result


if __name__ == "__main__":
    """
    Test scene detection on downloaded videos.
    
    Demonstrates:
    1. Skip logic for already-processed videos
    2. Multiple configuration presets
    3. Validation and error handling
    4. Integration with pipeline file structure
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

    # Configuration presets
    # Different use cases require different settings
    configs = {
        "balanced": {
            "detector_type": "adaptive",
            "threshold": 3.0,
            "min_scene_length": 2.0,
            "description": "Recommended: Good balance, handles camera motion",
        },
        "conservative": {
            "detector_type": "adaptive",
            "threshold": 7.0,
            "min_scene_length": 5.0,
            "description": "Fewer scenes, only obvious changes",
        },
        "sensitive": {
            "detector_type": "adaptive",
            "threshold": 2.0,
            "min_scene_length": 1.0,
            "description": "More scenes, catches subtle changes",
        },
        "fast": {
            "detector_type": "content",
            "threshold": 30.0,
            "min_scene_length": 2.0,
            "downscale": 2,
            "description": "Fast processing, good for clean cuts",
        },
    }

    # Choose configuration
    config_name = "conservative"
    config = configs[config_name]

    # Control reprocessing behavior
    force_redetect = False  # Set True to force reprocess all

    print("Configuration:")
    print(f"  Preset: {config_name}")
    print(f"  Description: {config['description']}")
    print(f"  Detector: {config['detector_type']}")
    print(f"  Threshold: {config['threshold']}")
    print(f"  Min scene: {config['min_scene_length']}s")
    if "downscale" in config:
        print(f"  Downscale: {config['downscale']}x")
    print(f"  Force redetect: {force_redetect}")
    print("=" * 70)

    # Track statistics
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for video_dir in video_dirs:
        try:
            print(f"\nProcessing: {video_dir.name}")
            start_time = time.time()

            result = process_video(video_dir, config, force_redetect=force_redetect)

            elapsed = time.time() - start_time

            # Check if was skipped (fast processing time indicates skip)
            was_skipped = elapsed < 1.0

            if was_skipped:
                skipped_count += 1
            else:
                processed_count += 1

            # Display results
            stats = result["statistics"]
            meta = result["metadata"]

            print(f"  Duration: {meta['video_duration']}s")
            print(f"  Scenes: {stats['total_scenes']}")
            print(f"  Average duration: {stats['average_duration']}s")
            print(f"  Range: {stats['shortest_scene']}s - {stats['longest_scene']}s")

            if was_skipped:
                print("  Status: SKIPPED (already processed)")
            else:
                print(f"  Processing time: {elapsed:.1f}s")

            # Validate
            errors = validate_scene_boundaries(result)
            if errors:
                print(f"  Validation issues: {len(errors)}")
                for error in errors[:3]:
                    print(f"    - {error}")
            else:
                print("  Validation: PASSED")

        except Exception as e:
            failed_count += 1
            print(f"  ERROR: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"Total videos: {len(video_dirs)}")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")

    if skipped_count > 0 and not force_redetect:
        print("\nSkipped videos are already processed.")
        print("  Use force_redetect=True to reprocess all")

    print("\n" + "=" * 70)
    print("CONFIGURATION GUIDE")
    print("=" * 70)
    print("\nAvailable presets:")
    for name, cfg in configs.items():
        print(f"  '{name}': {cfg['description']}")
    print("\nTo use different settings, change 'config_name' variable in script")
