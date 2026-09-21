import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yt_dlp

from vea.config import extract_youtube_id

# Logging is configured by the CLI (vea.cli) or by the calling application.
# Library modules must not call logging.basicConfig at import time.
logger = logging.getLogger(__name__)


def extract_video_id_from_url(url: str) -> Optional[str]:
    # The patterns moved to vea.config so `vea run` can validate a URL without
    # importing this module, which pulls in yt_dlp. One pattern, not two that
    # drift. The warning stays here: this caller logs and falls back to asking
    # yt-dlp directly, while the CLI refuses to start.
    video_id = extract_youtube_id(url)
    if video_id is None:
        logger.warning(f"Could not extract video ID from URL: {url}")
    return video_id


def check_if_downloaded(video_id: str, base_output: str = "downloads") -> bool:

    video_dir = Path(base_output) / f"video-{video_id}"

    if not video_dir.exists():
        return False

    # Check for required files
    metadata_path = video_dir / "metadata.json"
    if not metadata_path.exists():
        logger.warning(f"Directory {video_id} exists but missing metadata.json")
        return False

    # Load metadata to find actual filenames
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        video_file = metadata["file_paths"].get("video")
        audio_file = metadata["file_paths"].get("audio")

        video_exists = video_file and (video_dir / video_file).exists()
        audio_exists = audio_file and (video_dir / audio_file).exists()

        if video_exists and audio_exists:
            return True
        else:
            logger.warning(f"Directory {video_id} missing required files")
            return False

    except Exception as e:
        logger.warning(f"Error checking {video_id}: {e}")
        return False


def download_youtube_content(
    urls: List[str], base_output: str = "downloads", force_redownload: bool = False
) -> List[Dict]:

    base_path = Path(base_output)
    base_path.mkdir(exist_ok=True)

    results = []

    logger.info(f"Processing {len(urls)} URL(s)")
    logger.info(f"Force redownload: {force_redownload}")
    logger.info("=" * 60)

    for idx, url in enumerate(urls, start=1):
        logger.info(f"\n[{idx}/{len(urls)}] Processing: {url}")

        try:
            # Step 1: Extract video ID from URL
            video_id = extract_video_id_from_url(url)

            if not video_id:
                # Fallback: Try to get ID from yt-dlp
                # This is slower but more reliable for unusual URLs
                logger.info("Extracting video ID via yt-dlp...")
                try:
                    ydl_opts = {"quiet": True, "no_warnings": True}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        video_id = info.get("id")
                except Exception as e:
                    raise ValueError(f"Could not extract video ID: {e}")

            if not video_id:
                raise ValueError("Failed to extract video ID from URL")

            logger.info(f"Video ID: {video_id}")

            video_dir = base_path / f"video-{video_id}"

            # Step 2: Check if already downloaded
            if not force_redownload and check_if_downloaded(video_id, base_output):
                logger.info(f"Video {video_id} already downloaded - SKIPPING")
                logger.info(f"  Location: {video_dir}")
                logger.info("  (Use force_redownload=True to re-download)")

                # Load existing metadata for result
                with open(video_dir / "metadata.json", "r", encoding="utf-8") as f:
                    existing_metadata = json.load(f)

                results.append(
                    {
                        "status": "skipped",
                        "video_id": video_id,
                        "video_dir": str(video_dir),
                        "metadata": existing_metadata,
                        "skipped_reason": "already_downloaded",
                    }
                )
                continue

            # Step 3: Create/recreate video directory
            if video_dir.exists() and force_redownload:
                logger.warning(f"Force redownload enabled - will overwrite {video_id}")

            video_dir.mkdir(exist_ok=True)

            # Step 4: Extract metadata first (fast, validates URL)
            metadata = extract_metadata(url, video_dir, video_id)

            # Step 5: Download video stream
            video_path = download_video_stream(url, video_dir, video_id)

            # Step 6: Download and convert audio stream
            audio_path = download_audio_stream(url, video_dir, video_id)

            # Step 7: Validate downloads
            if validate_downloads(video_path, audio_path, metadata):
                logger.info(f"Successfully processed video {video_id}")
                results.append(
                    {
                        "status": "success",
                        "video_id": video_id,
                        "video_dir": str(video_dir),
                        "metadata": metadata,
                    }
                )
            else:
                raise ValueError("Download validation failed")

        except Exception as e:
            logger.error(f"Failed to process URL {idx}: {str(e)}")

            # Try to get video_id for error reporting
            try:
                video_id = extract_video_id_from_url(url) or "unknown"
            except:
                video_id = "unknown"

            results.append(
                {
                    "status": "failed",
                    "video_id": video_id,
                    "video_dir": str(base_path / f"video-{video_id}")
                    if video_id != "unknown"
                    else None,
                    "url": url,
                    "error": str(e),
                }
            )

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("BATCH SUMMARY")
    logger.info("=" * 60)

    successful = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")

    logger.info(f"Total URLs: {len(urls)}")
    logger.info(f"  Success: {successful}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Failed: {failed}")

    if skipped > 0:
        logger.info("\nSkipped videos are already downloaded.")
        logger.info(f"  Location: {base_output}/")
        logger.info("  Use force_redownload=True to re-download")

    return results


def extract_metadata(url: str, output_dir: Path, video_id: str) -> Dict:

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,  # Need full info, not just playlist data
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Build metadata dict matching our schema
            metadata = {
                "url": url,
                "video_id": info.get("id"),
                "title": info.get("title"),
                "uploader": info.get("uploader"),
                "duration": info.get("duration"),  # seconds, float
                "fps": info.get("fps"),  # Critical for scene detection
                "width": info.get("width"),
                "height": info.get("height"),
                "video_codec": info.get("vcodec"),
                "audio_codec": info.get("acodec"),
                "upload_date": info.get("upload_date"),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "download_date": datetime.utcnow().isoformat() + "Z",
                "file_paths": {
                    "video": None,  # Will be set after download
                    "audio": None,
                },
            }

            # Save metadata immediately
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Extracted metadata: {metadata['title']} ({metadata['duration']}s)")
            return metadata

    except Exception as e:
        logger.error(f"Failed to extract metadata from {url}: {str(e)}")
        raise


def download_video_stream(url: str, output_dir: Path, video_id: str) -> Path:

    output_template = str(output_dir / "video.%(ext)s")

    ydl_opts = {
        # Format priority: H.264 > VP9 > non-AV1 > any
        # This ensures maximum compatibility while still getting good quality
        "format": "bestvideo[vcodec^=avc1]/bestvideo[vcodec^=vp9]/bestvideo[vcodec!=av01]/bestvideo",
        "outtmpl": output_template,
        "quiet": False,
        "no_warnings": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            ext = info.get("ext", "mp4")
            video_path = output_dir / f"video.{ext}"

            _update_metadata(output_dir, "video", f"video.{ext}")

            # Log which codec was actually downloaded
            codec = info.get("vcodec", "unknown")
            logger.info(f"Downloaded video stream: {video_path.name} (codec: {codec})")
            return video_path

    except Exception as e:
        logger.error(f"Failed to download video stream: {str(e)}")
        raise


def download_audio_stream(url: str, output_dir: Path, video_id: str) -> Path:

    output_template = str(output_dir / "audio")

    format_options = [
        "bestaudio",
        "bestaudio*",
        "best",
    ]

    for format_str in format_options:
        ydl_opts = {
            "format": format_str,
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "postprocessor_args": [
                "-ar",
                "16000",
                "-ac",
                "1",
            ],
            "quiet": False,
            "no_warnings": False,
        }

        try:
            logger.info(f"Attempting audio download with format: {format_str}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)

                audio_path = output_dir / "audio.wav"

                if audio_path.exists():
                    # Update metadata with audio file info
                    _update_metadata(output_dir, "audio", "audio.wav")
                    _add_file_sizes(output_dir)

                    logger.info(f"Downloaded and converted audio: {audio_path.name}")
                    return audio_path
                else:
                    # File wasn't created, try next format
                    continue

        except Exception as e:
            logger.warning(f"Format '{format_str}' failed: {str(e)}")
            continue

    # All formats failed
    error_msg = "All audio download attempts failed. Video may have audio restrictions."
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def validate_downloads(video_path: Path, audio_path: Path, metadata: Dict) -> bool:

    issues = []

    if not video_path.exists():
        issues.append(f"Video file not found: {video_path}")

    if not audio_path.exists():
        issues.append(f"Audio file not found: {audio_path}")

    if issues:
        logger.error(f"Validation failed: {', '.join(issues)}")
        return False

    video_size_mb = video_path.stat().st_size / (1024 * 1024)
    audio_size_mb = audio_path.stat().st_size / (1024 * 1024)

    if video_size_mb < 0.1:
        issues.append(f"Video file too small: {video_size_mb:.2f}MB")

    if audio_size_mb < 0.1:
        issues.append(f"Audio file too small: {audio_size_mb:.2f}MB")

    expected_duration = metadata.get("duration", 0)
    if expected_duration > 0:
        expected_size_mb = expected_duration / 60 * 15
        actual_size_mb = video_size_mb + audio_size_mb

        if actual_size_mb < expected_size_mb / 10:
            issues.append(
                f"Files may be truncated: expected ~{expected_size_mb:.1f}MB, "
                f"got {actual_size_mb:.1f}MB"
            )

    if issues:
        for issue in issues:
            logger.warning(issue)
        return False

    logger.info(f"Validation passed: video={video_size_mb:.1f}MB, audio={audio_size_mb:.1f}MB")
    return True


def _update_metadata(output_dir: Path, stream_type: str, filename: str) -> None:

    metadata_path = output_dir / "metadata.json"

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        metadata["file_paths"][stream_type] = filename

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.warning(f"Failed to update metadata: {str(e)}")


def _add_file_sizes(output_dir: Path) -> None:

    metadata_path = output_dir / "metadata.json"

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        video_file = metadata["file_paths"].get("video")
        audio_file = metadata["file_paths"].get("audio")

        file_sizes = {}

        if video_file:
            video_path = output_dir / video_file
            if video_path.exists():
                file_sizes["video_mb"] = round(video_path.stat().st_size / (1024 * 1024), 1)

        if audio_file:
            audio_path = output_dir / audio_file
            if audio_path.exists():
                file_sizes["audio_mb"] = round(audio_path.stat().st_size / (1024 * 1024), 1)

        metadata["file_sizes"] = file_sizes

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.warning(f"Failed to add file sizes to metadata: {str(e)}")


def list_downloaded_videos(base_output: str = "downloads") -> List[Dict]:
    base_path = Path(base_output)

    if not base_path.exists():
        return []

    videos = []

    # Find all directories that look like video IDs (11 characters)
    for video_dir in base_path.iterdir():
        if not video_dir.is_dir():
            continue

        metadata_path = video_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            videos.append(
                {
                    "video_id": video_dir.name,
                    "title": metadata.get("title"),
                    "duration": metadata.get("duration"),
                    "download_date": metadata.get("download_date"),
                    "has_video": (video_dir / metadata["file_paths"].get("video", "")).exists(),
                    "has_audio": (video_dir / metadata["file_paths"].get("audio", "")).exists(),
                }
            )
        except Exception as e:
            logger.warning(f"Could not read metadata for {video_dir.name}: {e}")

    return videos


# Testing and demonstration
if __name__ == "__main__":
    # Same test URLs
    test_urls = [
        "https://www.youtube.com/watch?v=mqGSkDFeLEo",
        "https://www.youtube.com/watch?v=bvoF_lFC_gg",
        "https://www.youtube.com/watch?v=3HZCDlSmWZs",
        "https://www.youtube.com/watch?v=q7EKQsR57uk&t=157s",
    ]

    print("=" * 70)
    print("EXAMPLE")
    print("=" * 70)

    # Show existing downloads
    existing = list_downloaded_videos("downloads")
    if existing:
        print(f"\nFound {len(existing)} existing download(s):")
        for video in existing:
            print(f"  {video['video_id']}: {video['title']}")
            print(f"    Duration: {video['duration']}s, Downloaded: {video['download_date']}")
    else:
        print("\nNo existing downloads found")

    print("\n" + "=" * 70)
    print("Running with test URLs")
    print("=" * 70)
    print("Expected behavior: Skip already-downloaded videos, only download new ones")
    print("")

    # Run downloader
    results = download_youtube_content(test_urls, base_output="downloads")

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Video ID: {result['video_id']}")
        print(f"    Status: {result['status'].upper()}")

        if result["status"] == "success":
            print(f"    Title: {result['metadata']['title']}")
            print(f"    Duration: {result['metadata']['duration']}s")
            print(f"    Location: {result['video_dir']}")
        elif result["status"] == "skipped":
            print(f"    Title: {result['metadata']['title']}")
            print(f"    Reason: {result.get('skipped_reason')}")
            print(f"    Location: {result['video_dir']}")
            print("    Note: All existing processing data preserved!")
        elif result["status"] == "failed":
            print(f"    Error: {result['error']}")

    # Show directory structure
    print("\n" + "=" * 70)
    print("DIRECTORY STRUCTURE")
    print("=" * 70)
    print("\ndownloads/")

    existing = list_downloaded_videos("downloads")
    for video in sorted(existing, key=lambda x: x["video_id"]):
        print(f"  {video['video_id']}/  # {video['title'][:50]}")
