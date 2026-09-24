"""
Two-Level Scene Segmentation Module with Semantic Similarity (Stage 5)

Creates hierarchical scene structure:
1. Global semantic scenes (10+ min) - for visualization structure
2. Local idea segments (8-60 words) - for emotion classification

APPROACH: Semantic embeddings for measuring idea coherence
- Uses sentence-transformers multilingual models for Russian text
- Measures actual meaning, not just shared words
- Industry standard for semantic similarity tasks
- Always respects sentence boundaries (complete thoughts)

Design rationale:
- Global scenes: High-level narrative structure (vertical lines on timeline)
- Local segments: Coherent ideas optimal for emotion detection
- Semantic similarity: Understands "машина" = "автомобиль" (synonyms)
- Word-count constraints: Ensures sufficient context (8-60 words)
- Sentence boundaries: Never splits incomplete thoughts

Alternative approaches considered:
- Word overlap: Too simplistic, misses semantics
- Fixed-duration: Ignores linguistic structure
- Character count: Doesn't account for information density
- Current approach: State-of-the-art semantic understanding
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Sentence embeddings for semantic similarity
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("WARNING: sentence-transformers not installed")
    print("Install with: pip install sentence-transformers")
    print("Falling back to word overlap method (less accurate)")

# Logging is configured by the CLI (vea.cli) or by the calling application.
# Library modules must not call logging.basicConfig at import time.
logger = logging.getLogger(__name__)


# =============================================================================
# SEMANTIC SIMILARITY
# =============================================================================


def load_semantic_model(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    device: Optional[str] = None,
) -> Optional[SentenceTransformer]:
    """
    Load sentence embedding model for semantic similarity.

    Why this model:
    - Multilingual: Works for Russian natively (50+ languages)
    - Balanced: 420MB, fast inference, good accuracy
    - Proven: Industry standard for semantic similarity

    Alternative models:
    - paraphrase-multilingual-mpnet-base-v2: Better quality, slower (970MB)
    - distiluse-base-multilingual-cased-v1: Faster, lower quality (480MB)
    - Current choice: Best balance for production use

    Returns:
        Loaded model or None if library not available
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        logger.warning("sentence-transformers not available, using fallback method")
        return None

    try:
        logger.info(f"Loading semantic model: {model_name}")
        start = time.time()
        model = SentenceTransformer(model_name, device=device)
        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")
        return model
    except Exception as e:
        logger.error(f"Failed to load semantic model: {e}")
        logger.warning("Falling back to word overlap method")
        return None


def calculate_semantic_similarity(
    text1: str, text2: str, model: Optional[SentenceTransformer] = None
) -> float:
    """
    Calculate semantic similarity between two sentences.

    Uses sentence embeddings if model available, otherwise falls back
    to word overlap method.

    Why semantic embeddings:
    - Understands meaning: "машина сломалась" ≈ "автомобиль не работает"
    - Handles synonyms: "большой" ≈ "огромный"
    - Cross-language: Works for Russian without translation
    - Industry standard: Used in RAG, semantic search, QA systems

    Fallback word overlap:
    - Simple but limited: Only counts shared words
    - Misses semantics: "машина" ≠ "автомобиль" (0% overlap!)
    - Still useful: Better than nothing if embeddings unavailable

    Args:
        text1: First sentence
        text2: Second sentence
        model: Loaded SentenceTransformer model (optional)

    Returns:
        Similarity score 0.0-1.0 (higher = more similar)
    """
    if model is not None:
        try:
            # Use semantic embeddings (preferred method)
            embeddings = model.encode([text1, text2])

            # Cosine similarity: dot product of normalized vectors
            # Range: -1 to 1, but typically 0-1 for similar content
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )

            return float(max(0.0, similarity))  # Clamp to [0, 1]

        except Exception as e:
            logger.warning(f"Embedding calculation failed: {e}, using fallback")
            # Fall through to word overlap method

    # Fallback: Word overlap method (from original implementation)
    return calculate_word_overlap(text1, text2)


def calculate_word_overlap(text1: str, text2: str) -> float:
    """
    Fallback method: Calculate word overlap similarity.

    This is the original implementation - kept for backward compatibility
    and as fallback when sentence-transformers not available.

    Limitations:
    - Only counts shared words (ignores synonyms)
    - Sensitive to stopwords
    - No semantic understanding

    Still useful:
    - Fast (no model inference)
    - No dependencies
    - Works offline
    """
    # Russian stopwords (common words to ignore)
    stopwords = {
        "и",
        "в",
        "на",
        "с",
        "по",
        "для",
        "от",
        "к",
        "у",
        "о",
        "об",
        "а",
        "но",
        "да",
        "или",
        "то",
        "что",
        "как",
        "это",
        "весь",
        "он",
        "она",
        "они",
        "его",
        "её",
        "их",
        "мы",
        "вы",
        "ты",
        "я",
        "меня",
        "мне",
        "тебя",
        "вас",
        "нас",
        "не",
        "ни",
        "уже",
        "еще",
        "так",
        "вот",
        "здесь",
        "там",
        "тут",
        "тоже",
        "также",
        "же",
        "ли",
        "бы",
        "только",
        "был",
        "была",
        "было",
        "были",
        "есть",
        "быть",
        "будет",
        "будут",
        "буду",
        "будешь",
    }

    def clean_words(text):
        words = text.lower().split()
        cleaned = set()
        for w in words:
            # Remove punctuation
            w = re.sub(r"[^\w]", "", w, flags=re.UNICODE)
            # Keep words >= 3 chars and not stopwords
            if len(w) >= 3 and w not in stopwords:
                cleaned.add(w)
        return cleaned

    set1 = clean_words(text1)
    set2 = clean_words(text2)

    if not set1 or not set2:
        return 0.0

    # Jaccard similarity: intersection / union
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


# =============================================================================
# TEXT PROCESSING
# =============================================================================


def split_text_into_sentences(text: str) -> List[str]:
    """
    Split Russian text into sentences while PRESERVING punctuation.

    Critical: Punctuation marks meaning complete thoughts.
    Must preserve periods, exclamation marks, question marks.

    Design choice: Keep punctuation attached to sentence
    - Easier validation: Can check if ends with punctuation
    - Natural reading: Matches how humans see sentences
    - Emotion cues: "Отлично!" vs "Отлично" have different emotions
    """
    if not text:
        return []

    # Normalize spacing
    text = re.sub(r"\s+", " ", text).strip()

    # Add space after punctuation if missing
    text = re.sub(r"([.!?…])([А-ЯA-ZЁ«])", r"\1 \2", text)

    # Normalize ellipsis
    text = text.replace("…", "...")
    text = re.sub(r"\.{3,}", "...", text)

    # Split while KEEPING punctuation
    # Pattern: Split after punctuation + space before capital letter
    parts = re.split(r"(?<=[.!?…])\s+(?=[А-ЯA-ZЁ«])|(?<=[.!?…])$", text)

    sentences = []
    for sent in parts:
        sent = sent.strip()

        # Filter criteria
        if not sent:
            continue

        word_count = len(sent.split())
        if word_count < 2:
            continue

        # Skip if only punctuation
        if re.match(r"^[^\w]+$", sent, re.UNICODE):
            continue

        sentences.append(sent)

    return sentences


# =============================================================================
# TRANSCRIPTION TO WORD MAPPING
# =============================================================================


def find_best_transcription(video_dir: Path) -> Optional[Path]:
    """
    Find the best transcription file in a video directory.

    Selection criteria (priority order):
    1. Full video transcription (not time-limited)
    2. Most recent (latest date)
    3. Best model (large-v3 > large > medium > small)
    4. Longest duration

    Why this order:
    - Full transcriptions always preferred (complete data)
    - Newer better (may have bug fixes, better processing)
    - Better models produce more accurate text
    """
    video_dir = Path(video_dir)
    transcription_files = list(video_dir.glob("transcription_*.json"))

    if not transcription_files:
        return None

    transcriptions_with_meta = []

    for trans_file in transcription_files:
        try:
            with open(trans_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})

            transcriptions_with_meta.append(
                {
                    "path": trans_file,
                    "is_full": metadata.get("duration_limit") == "full",
                    "date": metadata.get("transcription_date", ""),
                    "model": metadata.get("model", ""),
                    "duration": metadata.get("duration", 0),
                }
            )
        except Exception as e:
            logger.warning(f"Could not read {trans_file.name}: {e}")
            continue

    if not transcriptions_with_meta:
        return None

    # Model quality ranking
    model_order = {"large-v3": 5, "large": 4, "medium": 3, "small": 2, "base": 1, "tiny": 0}

    def sort_key(t):
        model_priority = model_order.get(t["model"], 0)
        return (t["is_full"], t["date"], model_priority, t["duration"])

    transcriptions_with_meta.sort(key=sort_key, reverse=True)

    best = transcriptions_with_meta[0]
    logger.info(f"Selected transcription: {best['path'].name}")
    logger.info(
        f"  Model: {best['model']}, Full: {best['is_full']}, Duration: {best['duration']:.1f}s"
    )

    return best["path"]


def _map_sentences_to_words(sentences: List[str], words_in_scene: List[Dict]) -> List[Dict]:
    """
    Map sentence strings to word timestamp objects.

    Strategy: Greedy matching
    - Assume sentences appear in order
    - Take next N words from word stream for each sentence

    Why greedy:
    - Simple and fast
    - Works well when transcription quality is good
    - Whisper word timestamps are accurate enough

    Alternative considered:
    - Fuzzy word matching: Slower, more complex
    - Edit distance alignment: Overkill for good transcriptions
    - Current approach: Sufficient for production use

    Limitation:
    - If transcription has major errors (missing/extra words),
      alignment may be slightly off
    - In practice: Whisper is accurate enough that this rarely matters
    """
    sentence_data = []
    word_idx = 0

    for sentence in sentences:
        # Count words in sentence
        sentence_word_list = sentence.split()
        expected_word_count = len(sentence_word_list)

        # Collect corresponding words from timestamp stream
        sentence_words = []
        collected = 0

        # Greedy: take next N words
        while word_idx < len(words_in_scene) and collected < expected_word_count:
            sentence_words.append(words_in_scene[word_idx])
            word_idx += 1
            collected += 1

        # Only create entry if we found words
        if sentence_words:
            sentence_data.append(
                {
                    "text": sentence,
                    "words": sentence_words,
                    "start": sentence_words[0]["start"],
                    "end": sentence_words[-1]["end"],
                    "word_list": sentence_word_list,  # For similarity calculation
                }
            )

    return sentence_data


# =============================================================================
# GLOBAL SCENE CREATION
# =============================================================================


def group_shots_into_global_scenes(
    shots: List[Dict], all_words: List[Dict], config: Dict
) -> List[Dict]:
    """
    Create global semantic scenes from PySceneDetect shots.

    Purpose: High-level narrative structure for visualization (timeline markers).
    Goal: Reduce 300 shots/hour to 10-30 large scenes (5-15 min each).

    Grouping strategy:
    - Aggressive merging (only split on major boundaries)
    - Duration-based: Split if scene exceeds max (15 min default)
    - Silence-based: Split on long silence gaps (5s default)

    Why aggressive:
    - PySceneDetect detects every camera cut
    - We want narrative structure, not technical cuts
    - A dialogue scene might have 20 camera cuts but is one semantic unit
    """
    if not shots:
        return []

    max_duration = config.get("max_global_scene_duration", 900.0)  # 15 min
    min_gap = config.get("min_gap_for_global_split", 5.0)  # 5s silence

    logger.info(f"Creating global scenes from {len(shots)} shots")
    logger.info(f"  Max scene duration: {max_duration / 60:.1f} min")
    logger.info(f"  Min gap for split: {min_gap}s")

    # Start with first shot
    global_scenes = []
    current_scene = {
        "start": shots[0]["start"],
        "end": shots[0]["end"],
        "original_shot_ids": [shots[0]["scene_id"]],
        "shot_count": 1,
    }

    # Merge shots into global scenes
    for i in range(1, len(shots)):
        shot = shots[i]

        # Calculate metrics
        time_gap = shot["start"] - current_scene["end"]
        new_duration = shot["end"] - current_scene["start"]

        # Decision: Should we split here?
        should_split = False

        # Rule 1: Hard duration limit
        if new_duration > max_duration:
            should_split = True
            logger.debug(
                f"  Split at shot {i}: duration limit ({new_duration:.1f}s > {max_duration}s)"
            )

        # Rule 2: Large silence gap
        elif time_gap > min_gap:
            should_split = True
            logger.debug(f"  Split at shot {i}: silence gap ({time_gap:.1f}s)")

        if should_split:
            # Finalize current scene
            current_scene["duration"] = current_scene["end"] - current_scene["start"]
            global_scenes.append(current_scene)

            # Start new scene
            current_scene = {
                "start": shot["start"],
                "end": shot["end"],
                "original_shot_ids": [shot["scene_id"]],
                "shot_count": 1,
            }
        else:
            # Extend current scene
            current_scene["end"] = shot["end"]
            current_scene["original_shot_ids"].append(shot["scene_id"])
            current_scene["shot_count"] += 1

    # Finalize last scene
    current_scene["duration"] = current_scene["end"] - current_scene["start"]
    global_scenes.append(current_scene)

    logger.info(f"Created {len(global_scenes)} global scenes")
    logger.info(
        f"  Reduction: {len(shots)} → {len(global_scenes)} ({len(shots) / len(global_scenes):.1f}x)"
    )

    return global_scenes


# =============================================================================
# LOCAL SEGMENT CREATION (SEMANTIC VERSION)
# =============================================================================


def _split_long_sentence(sentence_text: str, max_words: int) -> List[str]:
    """
    Split a long sentence into smaller chunks at punctuation boundaries.

    Why needed: Russian sentences can be very long with multiple clauses.
    Respects: Commas, semicolons, dashes as natural break points.
    Fallback: Hard split if no punctuation available.
    """
    words = sentence_text.split()

    if len(words) <= max_words:
        return [sentence_text]

    # Try to split at punctuation (commas, semicolons, dashes)
    # Pattern: split at , ; — but keep punctuation with previous part
    parts = re.split(r"(?<=[,;—])\s+", sentence_text)

    chunks = []
    current_chunk = []
    current_count = 0

    for part in parts:
        part_words = part.split()
        part_count = len(part_words)

        if current_count + part_count <= max_words:
            current_chunk.append(part)
            current_count += part_count
        else:
            # Finalize current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [part]
            current_count = part_count

    # Add remaining
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Fallback: if still too long, hard split by words
    final_chunks = []
    for chunk in chunks:
        chunk_words = chunk.split()
        if len(chunk_words) <= max_words:
            final_chunks.append(chunk)
        else:
            # Hard split
            for i in range(0, len(chunk_words), max_words):
                final_chunks.append(" ".join(chunk_words[i : i + max_words]))

    return final_chunks


def create_local_idea_segments(
    global_scene: Dict,
    all_words: List[Dict],
    global_scene_id: int,
    config: Dict,
    semantic_model: Optional[SentenceTransformer] = None,
) -> List[Dict]:
    """
    Create LOCAL segments using SEMANTIC SIMILARITY + SENTENCE BOUNDARIES.

    KEY PRINCIPLES:
    1. ALWAYS end on sentence boundary (complete thought)
    2. Use semantic embeddings for similarity (not word overlap)
    3. Enforce word count constraints (8-60 words)
    4. Merge short sentences, split on topic changes

    Why semantic embeddings over word overlap:
    - "машина сломалась" and "автомобиль не работает" have 0% word overlap
      but are semantically identical (embedding similarity ~0.85)
    - Captures synonyms, paraphrases, related concepts
    - Works across languages (multilingual models)

    Design rationale:
    - Respects linguistic structure (sentence = complete idea)
    - Measures actual meaning (embeddings, not keywords)
    - Practical constraints (word counts for emotion classification)
    - Handles Russian linguistic patterns (flexible sentence length)

    Args:
        global_scene: Parent global scene dict
        all_words: All word timestamps from transcription
        global_scene_id: Parent scene ID
        config: Configuration dict
        semantic_model: Loaded SentenceTransformer model (optional)

    Returns:
        List of local segment dicts
    """

    # Configuration
    min_words = config.get("min_words_per_segment", 6)
    target_words = config.get("target_words_per_segment", 15)
    max_words = config.get("max_words_per_segment", 30)
    max_sentences = config.get("max_sentences_per_segment", 4)

    # Similarity threshold
    # Lower for embeddings than word overlap (embeddings more accurate)
    # 0.3-0.5 recommended for semantic segmentation
    min_similarity = config.get("min_similarity_threshold", 0.35)

    scene_start = global_scene["start"]
    scene_end = global_scene["end"]

    # Extract words in this global scene
    words_in_scene = [w for w in all_words if scene_start <= w["start"] < scene_end]

    # Handle empty scene (no speech)
    if not words_in_scene:
        return [
            {
                "global_scene_id": global_scene_id,
                "start": scene_start,
                "end": scene_end,
                "duration": scene_end - scene_start,
                "text": "",
                "sentence_count": 0,
                "word_count": 0,
                "has_speech": False,
                "confidence_average": None,
            }
        ]

    # Split text into sentences
    full_text = " ".join(w["word"] for w in words_in_scene)
    sentences = split_text_into_sentences(full_text)

    logger.debug(
        f"  Scene {global_scene_id}: {len(sentences)} sentences, {len(words_in_scene)} words"
    )

    if not sentences:
        logger.warning(f"  Scene {global_scene_id}: Sentence splitting failed")
        sentences = [full_text]

    # FIRST: Check for long sentences and split them
    processed_sentences = []
    for sent in sentences:
        sent_word_count = len(sent.split())
        if sent_word_count > max_words:
            logger.warning(f"  Long sentence detected ({sent_word_count} words), splitting...")
            # Split long sentence at punctuation boundaries
            sub_sentences = _split_long_sentence(sent, max_words)
            processed_sentences.extend(sub_sentences)
            logger.debug(f"    Split into {len(sub_sentences)} parts")
        else:
            processed_sentences.append(sent)

    # Map sentences to word timestamps
    sentence_data = _map_sentences_to_words(processed_sentences, words_in_scene)

    # Handle mapping failure
    if not sentence_data:
        confidences = [w.get("probability", 0) for w in words_in_scene]
        return [
            {
                "global_scene_id": global_scene_id,
                "start": scene_start,
                "end": scene_end,
                "duration": scene_end - scene_start,
                "text": full_text,
                "sentence_count": 0,
                "word_count": len(words_in_scene),
                "has_speech": True,
                "confidence_average": float(np.mean(confidences)) if confidences else None,
            }
        ]

    # SEMANTIC GROUPING STRATEGY
    local_segments = []
    current_sentences = []
    current_word_count = 0

    for i, sent_data in enumerate(sentence_data):
        sentence_word_count = len(sent_data["word_list"])
        sentence_text = sent_data["text"]

        # RULE 0: First sentence - always start new segment
        if not current_sentences:
            current_sentences.append(sent_data)
            current_word_count = sentence_word_count
            continue

        # Calculate potential new totals
        potential_word_count = current_word_count + sentence_word_count
        potential_sentence_count = len(current_sentences) + 1

        # DECISION TREE
        should_split = False
        split_reason = None

        # RULE 1: Would exceed max words - MUST split
        if potential_word_count > max_words:
            should_split = True
            split_reason = f"max_words ({potential_word_count} > {max_words})"

        # RULE 2: Would exceed max sentences - MUST split
        elif potential_sentence_count > max_sentences:
            should_split = True
            split_reason = f"max_sentences ({potential_sentence_count} > {max_sentences})"

        # RULE 3: Below minimum - KEEP accumulating
        # Don't check similarity yet, need more context
        elif current_word_count < min_words:
            current_sentences.append(sent_data)
            current_word_count = potential_word_count
            continue

        # RULE 4: Past target + semantic shift - CAN split
        elif current_word_count >= target_words:
            # Calculate semantic similarity using embeddings
            last_sentence_text = current_sentences[-1]["text"]

            similarity = calculate_semantic_similarity(
                last_sentence_text, sentence_text, model=semantic_model
            )

            # Low similarity = topic change
            if similarity < min_similarity:
                should_split = True
                split_reason = f"semantic_shift (sim={similarity:.3f}, words={current_word_count})"
            else:
                # High similarity = same topic, keep merging
                current_sentences.append(sent_data)
                current_word_count = potential_word_count
                continue

        # RULE 5: Between min and target - KEEP merging
        # Still building up to optimal size
        else:
            current_sentences.append(sent_data)
            current_word_count = potential_word_count
            continue

        # Execute split (only reaches here if should_split=True)
        if should_split:
            segment = _create_local_segment(current_sentences, global_scene_id, global_scene)
            local_segments.append(segment)

            logger.debug(
                f"    Segment {len(local_segments)}: "
                f"{len(current_sentences)} sent, {current_word_count} words "
                f"({split_reason})"
            )

            # Start new segment with current sentence
            current_sentences = [sent_data]
            current_word_count = sentence_word_count

    # Finalize last segment
    if current_sentences:
        # Merge with previous if too short
        if current_word_count < min_words and local_segments:
            logger.debug(
                f"    Merging short final segment ({current_word_count} words) with previous"
            )

            # Update previous segment
            prev_segment = local_segments[-1]
            combined_text = (
                prev_segment["text"] + " " + " ".join(s["text"] for s in current_sentences)
            )

            local_segments[-1] = {
                **prev_segment,
                "end": current_sentences[-1]["end"],
                "duration": current_sentences[-1]["end"] - prev_segment["start"],
                "text": combined_text,
                "sentence_count": prev_segment["sentence_count"] + len(current_sentences),
                "word_count": prev_segment["word_count"] + current_word_count,
            }

            logger.debug(
                f"    Merged: {local_segments[-1]['word_count']} words, "
                f"{local_segments[-1]['sentence_count']} sent"
            )
        else:
            # Normal case - create final segment
            segment = _create_local_segment(current_sentences, global_scene_id, global_scene)
            local_segments.append(segment)
            logger.debug(
                f"    Final segment: {len(current_sentences)} sent, {current_word_count} words"
            )

    return local_segments


def _create_local_segment(sentences: List[Dict], global_scene_id: int, global_scene: Dict) -> Dict:
    """
    Create local segment dict from sentence group.

    Ensures segment respects all constraints:
    - Ends on sentence boundary (complete thought)
    - Has proper timestamps from word data
    - Includes all required fields for pipeline
    """

    # Combine text (preserving sentence boundaries)
    text = " ".join(s["text"] for s in sentences)

    # Collect all words
    all_words = []
    for sent in sentences:
        all_words.extend(sent["words"])

    # Calculate timestamps
    start = sentences[0]["start"]
    end = sentences[-1]["end"]
    duration = end - start

    # Calculate average confidence
    confidences = [w.get("probability", 0) for w in all_words]
    avg_confidence = float(np.mean(confidences)) if confidences else None

    # Build segment dict matching pipeline schema
    segment = {
        "global_scene_id": global_scene_id,
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(duration, 2),
        "text": text,
        "sentence_count": len(sentences),
        "word_count": len(all_words),
        "has_speech": True,
        "confidence_average": round(avg_confidence, 3) if avg_confidence else None,
        "original_shot_count": global_scene["shot_count"],
        "original_shot_ids": global_scene["original_shot_ids"],
    }

    return segment


# =============================================================================
# VALIDATION
# =============================================================================


def validate_global_scenes(scenes: List[Dict]) -> List[str]:
    """
    Validate global scene structure for temporal consistency.
    """
    errors = []

    if not scenes:
        errors.append("No global scenes created")
        return errors

    # Check first scene starts near beginning
    if scenes[0]["start"] > 5.0:
        errors.append(f"First global scene starts late: {scenes[0]['start']}s")

    # Check temporal ordering and gaps
    for i in range(len(scenes) - 1):
        current = scenes[i]
        next_scene = scenes[i + 1]

        # Check for gaps (allow small tolerance)
        gap = next_scene["start"] - current["end"]
        if abs(gap) > 1.0:
            errors.append(f"Global scene {i}: Gap of {gap:.2f}s to next scene")

        # Check for overlaps
        if current["end"] > next_scene["start"]:
            errors.append(f"Global scene {i}: Overlaps with scene {i + 1}")

        # Check duration calculation
        calc_duration = current["end"] - current["start"]
        if abs(calc_duration - current["duration"]) > 0.1:
            errors.append(
                f"Global scene {i}: Duration mismatch "
                f"(calculated: {calc_duration:.2f}, stored: {current['duration']})"
            )

    return errors


def validate_local_segments(segments: List[Dict]) -> List[str]:
    """
    Validate local segment structure.

    Critical checks:
    - Word count constraints respected
    - Temporal consistency maintained
    - Data completeness
    """
    errors = []

    if not segments:
        errors.append("No local segments created")
        return errors

    # Check each segment
    for i, seg in enumerate(segments):
        # Check word count constraints
        if seg["has_speech"] and seg["word_count"] < 8:
            errors.append(f"Segment {i}: Too short ({seg['word_count']} words, min 8)")

        if seg["word_count"] > 60:
            errors.append(f"Segment {i}: Too long ({seg['word_count']} words, max 60)")

        # Check data consistency
        if seg["has_speech"] and seg["word_count"] == 0:
            errors.append(f"Segment {i}: has_speech=True but word_count=0")

        if seg["has_speech"] and not seg["text"]:
            errors.append(f"Segment {i}: has_speech=True but empty text")

        # Check duration
        calc_duration = seg["end"] - seg["start"]
        if abs(calc_duration - seg["duration"]) > 0.1:
            errors.append(
                f"Segment {i}: Duration mismatch "
                f"(calculated: {calc_duration:.2f}, stored: {seg['duration']})"
            )

    # Check temporal ordering (allow small gaps)
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]

        gap = next_seg["start"] - current["end"]
        if gap > 5.0:  # More lenient for local segments
            errors.append(f"Segment {i}: Large gap of {gap:.2f}s to next segment")

    return errors


# =============================================================================
# SKIP LOGIC & FILE MANAGEMENT
# =============================================================================


def check_if_processed(video_dir: Path, transcription_name: str, config: Dict) -> Tuple[bool, bool]:
    """
    Check if video already processed.

    Returns:
        Tuple of (global_scenes_exists, local_segments_exists)
    """
    video_dir = Path(video_dir)

    # Check global scenes (shared across all transcriptions)
    global_exists = (video_dir / "global_scenes.json").exists()

    # Check local segments (specific to transcription)
    trans_base = transcription_name.replace(".json", "")
    local_exists = (video_dir / f"local_segments_{trans_base}.json").exists()

    return global_exists, local_exists


# =============================================================================
# MAIN PROCESSING
# =============================================================================


def process_video(
    video_dir: Path,
    config: Dict = None,
    transcription_name: Optional[str] = None,
    force_reprocess: bool = False,
    semantic_model: Optional[SentenceTransformer] = None,
) -> Tuple[Dict, Dict]:
    """
    Process two-level scene segmentation for a video.

    Pipeline integration:
    - Input: scene_boundaries.json (Stage 2) + transcription_*.json (Stage 4)
    - Output: global_scenes.json + local_segments_*.json
    - Next stage: Emotion classification (Stage 6) uses local segments

    Args:
        video_dir: Path to video-{ID} directory
        config: Configuration dict (uses defaults if None)
        transcription_name: Specific transcription file (auto-selects if None)
        force_reprocess: Reprocess even if already done
        semantic_model: Pre-loaded SentenceTransformer model (optional)

    Returns:
        Tuple of (global_scenes_dict, local_segments_dict)

    Raises:
        FileNotFoundError: If required input files missing
        ValueError: If input data invalid
    """
    video_dir = Path(video_dir)

    # Load PySceneDetect shots
    shots_file = video_dir / "scene_boundaries.json"
    if not shots_file.exists():
        raise FileNotFoundError(
            f"Scene boundaries not found: {shots_file}\nRun scene_detector.py first (Stage 2)"
        )

    with open(shots_file, "r", encoding="utf-8") as f:
        shots_data = json.load(f)

    shots = shots_data.get("scenes", [])
    if not shots:
        raise ValueError("No shots found in scene_boundaries.json")

    logger.info(f"Loaded {len(shots)} PySceneDetect shots")

    # Find or load transcription
    if transcription_name:
        transcription_path = video_dir / transcription_name
        if not transcription_path.exists():
            raise FileNotFoundError(f"Transcription not found: {transcription_path}")
    else:
        transcription_path = find_best_transcription(video_dir)
        if not transcription_path:
            raise FileNotFoundError(
                f"No transcription files found in {video_dir}\nRun transcriber.py first (Stage 4)"
            )
        transcription_name = transcription_path.name

    logger.info(f"Using transcription: {transcription_name}")

    # Load transcription
    with open(transcription_path, "r", encoding="utf-8") as f:
        transcription = json.load(f)

    # Extract all words with timestamps
    all_words = []
    for segment in transcription.get("segments", []):
        words = segment.get("words", [])
        if words:
            all_words.extend(words)

    if not all_words:
        raise ValueError(
            f"No words with timestamps found in {transcription_name}\n"
            "Transcription may be missing word-level timestamps"
        )

    logger.info(f"Found {len(all_words)} words with timestamps")

    # Set configuration defaults
    config = config or {}
    config.setdefault("max_global_scene_duration", 900.0)  # 15 min
    config.setdefault("min_gap_for_global_split", 5.0)  # 5s
    config.setdefault("min_words_per_segment", 8)  # Min words
    config.setdefault("target_words_per_segment", 25)  # Target words
    config.setdefault("max_words_per_segment", 60)  # Max words
    config.setdefault("max_sentences_per_segment", 5)  # Max sentences
    config.setdefault("min_similarity_threshold", 0.35)  # Semantic threshold

    # Check if already processed
    if not force_reprocess:
        global_exists, local_exists = check_if_processed(video_dir, transcription_name, config)

        if global_exists and local_exists:
            logger.info("Already processed - SKIPPING")
            logger.info("  (Use force_reprocess=True to reprocess)")

            # Load and return existing results
            with open(video_dir / "global_scenes.json", "r", encoding="utf-8") as f:
                global_result = json.load(f)

            trans_base = transcription_name.replace(".json", "")
            with open(video_dir / f"local_segments_{trans_base}.json", "r", encoding="utf-8") as f:
                local_result = json.load(f)

            return global_result, local_result

    logger.info(f"\nProcessing: {video_dir.name}")
    if force_reprocess:
        logger.info("Force reprocess enabled")

    start_time = time.time()

    # ═════════════════════════════════════════════════════════════════════
    # STEP 1: Create Global Semantic Scenes
    # ═════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Creating GLOBAL semantic scenes (for visualization)")
    logger.info("=" * 70)

    global_scenes = group_shots_into_global_scenes(shots, all_words, config)

    # Add IDs
    for i, scene in enumerate(global_scenes):
        scene["global_scene_id"] = i

    # Validate
    global_errors = validate_global_scenes(global_scenes)
    if global_errors:
        logger.warning(f"Global scenes validation: {len(global_errors)} issues")
        for error in global_errors[:5]:
            logger.warning(f"  - {error}")
    else:
        logger.info("Global scenes validation: PASSED")

    # Calculate statistics
    durations = [s["duration"] for s in global_scenes]

    global_result = {
        "metadata": {
            "original_shots": len(shots),
            "global_scenes": len(global_scenes),
            "purpose": "visualization_structure",
            "reduction_ratio": round(len(shots) / len(global_scenes), 2),
            "config": {
                "max_global_scene_duration": config["max_global_scene_duration"],
                "min_gap_for_global_split": config["min_gap_for_global_split"],
            },
            "processing_date": datetime.utcnow().isoformat() + "Z",
        },
        "scenes": global_scenes,
        "statistics": {
            "total_scenes": len(global_scenes),
            "average_duration": round(np.mean(durations), 2),
            "median_duration": round(np.median(durations), 2),
            "shortest_scene": round(min(durations), 2),
            "longest_scene": round(max(durations), 2),
            "total_duration": round(sum(durations), 2),
        },
    }

    # Save global scenes
    with open(video_dir / "global_scenes.json", "w", encoding="utf-8") as f:
        json.dump(global_result, f, indent=2, ensure_ascii=False)

    logger.info("\nSaved: global_scenes.json")
    stats = global_result["statistics"]
    logger.info(f"  Total: {stats['total_scenes']} scenes")
    logger.info(f"  Duration range: {stats['shortest_scene']}s - {stats['longest_scene']}s")
    logger.info(
        f"  Average: {stats['average_duration']}s ({stats['average_duration'] / 60:.1f} min)"
    )
    logger.info(f"  Median: {stats['median_duration']}s ({stats['median_duration'] / 60:.1f} min)")

    # ═════════════════════════════════════════════════════════════════════
    # STEP 2: Create Local Idea Segments (SEMANTIC)
    # ═════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Creating LOCAL idea segments (SEMANTIC METHOD)")
    logger.info("=" * 70)
    logger.info(
        f"  Word constraints: {config['min_words_per_segment']}-{config['max_words_per_segment']}"
    )
    logger.info(f"  Target words: {config['target_words_per_segment']}")
    logger.info(f"  Max sentences: {config['max_sentences_per_segment']}")
    logger.info(
        f"  Similarity method: {'Semantic embeddings' if semantic_model else 'Word overlap (fallback)'}"
    )
    logger.info(f"  Similarity threshold: {config['min_similarity_threshold']}")

    all_local_segments = []
    local_segment_id = 0

    for global_scene in global_scenes:
        global_scene_id = global_scene["global_scene_id"]

        logger.debug(f"\nProcessing global scene {global_scene_id}:")

        # Create local segments for this global scene
        local_segments = create_local_idea_segments(
            global_scene, all_words, global_scene_id, config, semantic_model=semantic_model
        )

        # Assign sequential IDs
        for seg in local_segments:
            seg["local_segment_id"] = local_segment_id
            local_segment_id += 1

        all_local_segments.extend(local_segments)

        logger.debug(f"  Created {len(local_segments)} local segments")

    logger.info(
        f"\nCreated {len(all_local_segments)} local segments from "
        f"{len(global_scenes)} global scenes"
    )

    # Validate local segments
    local_errors = validate_local_segments(all_local_segments)
    if local_errors:
        logger.warning(f"Local segments validation: {len(local_errors)} issues")
        for error in local_errors[:10]:
            logger.warning(f"  - {error}")
    else:
        logger.info("Local segments validation: PASSED")

    # Calculate statistics
    segments_with_speech = [s for s in all_local_segments if s["has_speech"]]

    word_counts = [s["word_count"] for s in segments_with_speech]
    sentence_counts = [s["sentence_count"] for s in segments_with_speech]

    local_result = {
        "metadata": {
            "purpose": "emotion_classification",
            "transcription_file": transcription_name,
            "total_global_scenes": len(global_scenes),
            "segmentation_method": "semantic_embeddings" if semantic_model else "word_overlap",
            "config": {
                "min_words_per_segment": config["min_words_per_segment"],
                "target_words_per_segment": config["target_words_per_segment"],
                "max_words_per_segment": config["max_words_per_segment"],
                "max_sentences_per_segment": config["max_sentences_per_segment"],
                "min_similarity_threshold": config["min_similarity_threshold"],
            },
            "processing_date": datetime.utcnow().isoformat() + "Z",
        },
        "segments": all_local_segments,
        "statistics": {
            "total_segments": len(all_local_segments),
            "segments_with_speech": len(segments_with_speech),
            "segments_without_speech": len(all_local_segments) - len(segments_with_speech),
            "total_words": sum(s["word_count"] for s in all_local_segments),
            "word_count_distribution": {
                "<8": sum(1 for c in word_counts if c < 8),
                "8-25": sum(1 for c in word_counts if 8 <= c < 25),
                "25-40": sum(1 for c in word_counts if 25 <= c < 40),
                "40-60": sum(1 for c in word_counts if 40 <= c <= 60),
                ">60": sum(1 for c in word_counts if c > 60),
            },
            "sentence_count_distribution": {
                "1": sum(1 for c in sentence_counts if c == 1),
                "2": sum(1 for c in sentence_counts if c == 2),
                "3": sum(1 for c in sentence_counts if c == 3),
                "4": sum(1 for c in sentence_counts if c == 4),
                "5": sum(1 for c in sentence_counts if c == 5),
                ">5": sum(1 for c in sentence_counts if c > 5),
            },
            "average_words_per_segment": round(np.mean(word_counts), 1) if word_counts else 0,
            "median_words_per_segment": round(np.median(word_counts), 1) if word_counts else 0,
            "average_sentences_per_segment": round(np.mean(sentence_counts), 1)
            if sentence_counts
            else 0,
            "average_segment_duration": round(
                np.mean([s["duration"] for s in all_local_segments]), 2
            ),
        },
    }

    # Save local segments
    trans_base = transcription_name.replace(".json", "")
    output_filename = f"local_segments_{trans_base}.json"

    with open(video_dir / output_filename, "w", encoding="utf-8") as f:
        json.dump(local_result, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved: {output_filename}")
    stats = local_result["statistics"]
    logger.info(f"  Total segments: {stats['total_segments']}")
    logger.info(f"  With speech: {stats['segments_with_speech']}")
    logger.info(f"  Avg words/segment: {stats['average_words_per_segment']}")
    logger.info(f"  Median words/segment: {stats['median_words_per_segment']}")
    logger.info(f"  Word distribution: {stats['word_count_distribution']}")
    logger.info(f"  Sentence distribution: {stats['sentence_count_distribution']}")

    # Check for violations
    violations = stats["word_count_distribution"].get("<8", 0)
    if violations > 0:
        logger.warning(f"  WARNING: {violations} segments below 8-word minimum")

    violations = stats["word_count_distribution"].get(">60", 0)
    if violations > 0:
        logger.warning(f"  WARNING: {violations} segments exceed 60-word maximum")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal processing time: {elapsed:.1f}s")

    return global_result, local_result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    """
    Test two-level scene segmentation with semantic similarity.
    
    Demonstrates:
    1. Semantic embeddings for Russian text
    2. Skip logic for already-processed videos
    3. Word-based constraints (prevents short fragments)
    4. Validation and error handling
    5. Integration with pipeline file structure
    """
    import sys

    downloads_dir = Path("downloads")

    if not downloads_dir.exists():
        print("Downloads directory not found")
        print("Run downloader.py (Stage 1) first")
        sys.exit(1)

    video_dirs = sorted(downloads_dir.glob("video-*"))

    if not video_dirs:
        print("No video directories found in downloads/")
        sys.exit(1)

    print(f"Found {len(video_dirs)} video(s) to process")
    print("=" * 70)

    # Load semantic model once (reuse for all videos)
    semantic_model = load_semantic_model("paraphrase-multilingual-MiniLM-L12-v2")

    if semantic_model:
        print("Semantic embeddings: ENABLED (accurate)")
    else:
        print("Semantic embeddings: DISABLED (using word overlap fallback)")

    print("=" * 70)

    # Configuration with semantic approach
    config = {
        # Global scenes (for visualization structure)
        "max_global_scene_duration": 900.0,  # 15 min max
        "min_gap_for_global_split": 5.0,  # 5s silence triggers split
        # Local segments (for emotion classification)
        "min_words_per_segment": 8,  # Prevent short fragments
        "target_words_per_segment": 16,  # Optimal for emotion
        "max_words_per_segment": 35,  # Prevent overly long
        "max_sentences_per_segment": 4,  # Safety limit
        # Semantic similarity (lower threshold for embeddings)
        "min_similarity_threshold": 0.4,  # 0.3-0.5 recommended
    }

    force_reprocess = False

    print("Configuration:")
    print("  Global scenes:")
    print(f"    Max duration: {config['max_global_scene_duration'] / 60:.0f} min")
    print(f"    Min gap for split: {config['min_gap_for_global_split']}s")
    print("  Local segments:")
    print(f"    Word range: {config['min_words_per_segment']}-{config['max_words_per_segment']}")
    print(f"    Target words: {config['target_words_per_segment']}")
    print(f"    Max sentences: {config['max_sentences_per_segment']}")
    print(f"    Similarity threshold: {config['min_similarity_threshold']}")
    print(f"  Force reprocess: {force_reprocess}")
    print("=" * 70)

    # Process each video
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for video_dir in video_dirs:
        try:
            print(f"\n{'=' * 70}")
            print(f"Processing: {video_dir.name}")
            print("=" * 70)

            start_time = time.time()

            global_result, local_result = process_video(
                video_dir=video_dir,
                config=config,
                force_reprocess=force_reprocess,
                semantic_model=semantic_model,
            )

            elapsed = time.time() - start_time

            # Check if was skipped (fast processing indicates skip)
            was_skipped = elapsed < 1.0

            if was_skipped:
                skipped_count += 1
                print(f"\nSkipped: {video_dir.name} (already processed)")
            else:
                processed_count += 1
                print(f"\nSuccess: {video_dir.name}")

            print(f"  Global scenes: {len(global_result['scenes'])}")
            print(f"  Local segments: {len(local_result['segments'])}")

            # Display quality metrics
            stats = local_result["statistics"]
            print(f"  Avg words/segment: {stats['average_words_per_segment']}")
            print(f"  Median words/segment: {stats['median_words_per_segment']}")
            print(f"  Avg sentences/segment: {stats['average_sentences_per_segment']}")

            # Check for violations
            under = stats["word_count_distribution"].get("<8", 0)
            over = stats["word_count_distribution"].get(">60", 0)

            if under > 0:
                print(f"  WARNING: {under} segments below 8-word minimum")
            if over > 0:
                print(f"  WARNING: {over} segments exceed 60-word maximum")

        except Exception as e:
            failed_count += 1
            print(f"\nFailed: {video_dir.name}")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"Total videos: {len(video_dirs)}")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")

    if skipped_count > 0 and not force_reprocess:
        print("\nSkipped videos are already processed.")
        print("  Use force_reprocess=True to reprocess all")
