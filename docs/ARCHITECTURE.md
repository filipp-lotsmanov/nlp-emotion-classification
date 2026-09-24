# Russian Video Emotion Analysis Pipeline

A modular 9-stage pipeline for analyzing emotional content in Russian YouTube videos, featuring speech recognition, multilingual emotion classification, and cross-language validation.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Module Documentation](#module-documentation)
- [Output Files](#output-files)
- [Web Interface](#web-interface)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline processes Russian-language video content to extract and visualize emotional patterns. It combines speech recognition (Whisper), scene detection (PySceneDetect), semantic segmentation, multilingual translation (NLLB), and emotion classification to provide comprehensive emotional analysis of video content.

**Key Features:**
- **Hierarchical scene structure**: Global scenes for visualization + local segments for classification
- **Cross-language validation**: Parallel emotion analysis in Russian and English
- **Ensemble emotion detection**: 3-model consensus for improved accuracy
- **Anti-hallucination measures**: Optimized Whisper configuration for long videos
- **Continuous intensity values**: Preserves full arousal-valence information
- **Smart skip logic**: Reuses processed data to avoid redundant computation

**Primary Use Cases:**
- TV show and interview emotion analysis
- Content moderation and quality assessment
- Emotion timeline visualization
- Cross-language emotion validation research

---

## Architecture

### Pipeline Flow

```
Stage 1: YouTube Download (yt-dlp)
    ↓
Stage 2: Scene Detection (PySceneDetect)
    ↓
Stage 3: Audio Preprocessing (RMS normalization)
    ↓
Stage 4: Transcription (faster-whisper large-v3)
    ↓
Stage 5A: Scene Alignment (semantic embeddings)
    ↓
Stage 5B: Translation (NLLB Russian→English)
    ↓
Stage 6A: Russian Intensity (XLM-RoBERTa valence-arousal)
    ↓
Stage 6B: English Intensity (XLM-RoBERTa valence-arousal)
    ↓
Stage 7A: Russian Emotion (ruBERT 7-class)
    ↓
Stage 7B: English Emotion (DistilRoBERTa + DeBERTa 7-class)
    ↓
Stage 8: Visualization (emotion timeline)
    ↓
Stage 9: CSV Export (consolidated results)
```

### Design Principles

**Modularity**: Each stage lives in its own module under `src/vea/stages/` and exposes a `process_video` (or equivalent) entry point, so it can be run separately. Stages validate previous outputs and fail gracefully.

**Skip Logic**: Every module checks if output already exists with matching configuration. This enables fast iteration and prevents redundant processing.

**Error Isolation**: Each stage wrapper in `src/vea/pipeline.py` catches its own exceptions and returns a success flag. The orchestrator stops at the first failed stage from 1 to 7; a failure in stage 8 or 9 is logged and the run continues. If the NLLB translator fails to initialise, stages 5B, 6B and 7B are skipped and the Russian branch still runs.

**Configuration Transparency**: All parameters are logged and saved to output files for reproducibility.

**Resource Management**: The sentence-embedding model (stage 5A) and the NLLB translator (stage 5B) are loaded once, before stage 1, and reused. The stage 6 and 7 classifiers are loaded inside their own stage.

**No import-time side effects**: Library modules only call `logging.getLogger(__name__)`; logging is configured by the CLI (`vea.config.configure_logging`). No module pins `CUDA_VISIBLE_DEVICES`. Both are enforced by `tests/test_source_hygiene.py`.

---

## Installation

### Requirements

- Python >=3.11,<3.14 (`requires-python` in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) for environment management
- CUDA-capable GPU (recommended for faster processing)
- Ubuntu/Linux (tested on Ubuntu)
- ~50GB disk space for models and data

### Setup

Dependencies are declared in `pyproject.toml`; `uv.lock` pins the transitive set.

```bash
uv sync                              # pipeline and CLI
uv sync --extra api                  # plus `vea serve` (FastAPI + uvicorn)
uv sync --extra api --extra train    # plus retraining dependencies
```

Other extras: `baselines`, `xai`, `tensorflow` (used under `training/`). `uv sync` is exact, so any extra you leave off is removed.

Two stage models are local checkpoints that are not downloaded automatically: `va-xlmroberta-large` (stages 6A/6B, directory `models/xlmroberta-base-va/`) and `emotion-en-deberta` (stage 7B, directory `models/emotion-en-deberta/`). Fetch them with:

```bash
./scripts/fetch_va_checkpoint.sh
./scripts/fetch_emotion_en_checkpoint.sh
uv run vea models                    # preflight: which weights are present or missing
```

Hub models download on first use:
- Whisper large-v3 (~3GB)
- NLLB-200-3.3B (~17GB; see `translation_model_ref` in `src/vea/config.py`)
- Sentence embeddings, ruBERT-tiny2 and DistilRoBERTa emotion classifiers

### GPU Configuration

No GPU is hardcoded. `vea.config.resolve_device()` resolves a device from `VEA_DEVICE` (`auto`, `cpu`, `cuda`, `cuda:N`; default `auto`), and `uv run vea config` shows the resolved settings and visible GPUs. An explicit value (`cpu`, `cuda`, `cuda:N`) is written into every stage config whose `device` is unset (`apply_device_setting`, called by `pipeline.main`), and into the stage 5A embedder and the stage 5B translator; `vea run` refuses to start if it names a device torch cannot see. Under `auto`, each stage chooses for itself: stages 4, 5A, 6 and 7 pick `cuda` when torch reports it available and `cpu` otherwise, and stage 5B's `select_gpu()` picks the visible GPU with the most free memory. To choose a card while keeping that behaviour, restrict what is visible:

```bash
export CUDA_VISIBLE_DEVICES="$(scripts/pick_free_gpu.sh)"   # GPU with the most free memory
```

---

## Quick Start

### Basic Usage

```bash
# Process a single video
uv run vea run "https://www.youtube.com/watch?v=VIDEO_ID"
```

The `vea` command is defined in `src/vea/cli.py`; `vea run` calls `main` in `src/vea/pipeline.py`. Before importing the pipeline it refuses to start if a checkpoint a stage needs is missing (override with `--allow-missing-models`, in which case the dependent stages fail) or if the URL is not a YouTube URL it can parse. `--log-level` overrides `VEA_LOG_LEVEL`.

Other subcommands: `vea config` (resolved settings and devices), `vea models` (checkpoint preflight), `vea serve` (HTTP API, see [Web Interface](#web-interface)).

### Example

```bash
uv run vea run "https://www.youtube.com/watch?v=mqGSkDFeLEo"
```

### Output Location

All files are saved to `downloads/video-{VIDEO_ID}/`, relative to the working directory (`base_output` in the stage 1 config):
- Metadata, audio, video files
- Scene boundaries, transcriptions
- Intensity and emotion predictions
- Visualization and CSV export

---

## Pipeline Stages

### Stage 1: YouTube Download

**Module**: `src/vea/stages/download.py`

Downloads video and audio streams from YouTube using yt-dlp.

**Features:**
- Automatic video ID extraction
- Format fallback strategies (H.264 → VP9 → any available)
- Audio conversion to 16kHz mono WAV (optimal for Whisper)
- Comprehensive metadata extraction
- Skip logic to avoid re-downloading

**Output:**
- `video.{ext}` - Video file (mp4/webm)
- `audio.wav` - Audio file (16kHz mono)
- `metadata.json` - Video metadata and file paths

**Design Notes:**
- Uses `video-{ID}` directory naming to prevent overwrites
- Saves both video and audio for scene detection and transcription
- Metadata includes fps (critical for scene detection accuracy)

---

### Stage 2: Scene Detection

**Module**: `src/vea/stages/scenes.py`

Detects visual scene boundaries using PySceneDetect with adaptive thresholding.

**Features:**
- Adaptive detector (automatically adjusts to video characteristics)
- Configurable threshold and minimum scene length
- Scene statistics and quality metrics
- Skip logic with configuration validation

**Output:**
- `scene_boundaries.json` - Scene timestamps and metadata

**Design Notes:**
- Adaptive detection handles varying content types better than fixed thresholds
- Default threshold (3.0) balances sensitivity with false positives
- Minimum scene length (2.0s) prevents micro-scenes from camera flickers
- These are "camera shots" - later merged into semantic scenes in Stage 5A

**Alternative Approaches:**
- Content detector: More conservative, misses subtle transitions
- Fixed threshold: Less adaptable to different video types
- Current approach: Best balance for TV shows and interviews

---

### Stage 3: Audio Preprocessing

**Module**: `src/vea/stages/audio.py`

Normalizes audio levels and optionally reduces background noise.

**Features:**
- RMS normalization to -12dB (optimal for speech recognition)
- Optional noise reduction (noisereduce library)
- Audio quality metrics (silence detection, clipping warnings)
- Peak limiting to prevent distortion

**Output:**
- `preprocessed_audio.wav` - Normalized audio
- `audio_quality.json` - Quality metrics and warnings

**Design Notes:**
- -12dB target provides consistent loudness without clipping
- Noise reduction disabled by default (can harm speech clarity)
- Preserves original audio for debugging/comparison
- Quality flags help identify problematic audio

---

### Stage 4: Speech Transcription

**Module**: `src/vea/stages/transcribe.py`

Transcribes Russian speech using faster-whisper (CTranslate2 implementation of OpenAI Whisper).

**Features:**
- faster-whisper for 2-4x speedup over openai-whisper
- VAD filtering to remove silence (reduces hallucinations)
- Word-level timestamps for precise alignment
- Anti-hallucination measures for long videos
- Quality monitoring and hallucination detection

**Output:**
- `transcription_{model}.json` - Transcription with word timestamps (`transcription_large-v3.json` for a full run with the defaults). When `duration_limit` is a number of seconds N, the name gets an `_{N}s` suffix, e.g. `transcription_large-v3_300s.json` (`generate_transcription_filename`).

**Critical Anti-Hallucination Strategy:**
1. `vad_filter=True` - Removes silence where hallucinations occur
2. `condition_on_previous_text=False` - Prevents repetitive loops
3. `temperature=0.0` - Deterministic output
4. Conservative VAD parameters - Preserves speech edges
5. Quality metrics - Flags suspicious segments

**Design Notes:**
- Large-v3 model provides best accuracy for Russian
- Word timestamps critical for downstream alignment
- Compression ratio and no_speech_prob detect hallucinations
- VAD parameters tuned for speech preservation over aggressive filtering (`min_silence_duration_ms=2000`, `speech_pad_ms=400`, `threshold=0.5`, set in `transcribe_audio`)
- On CPU, `compute_type="float16"` is replaced with `int8` automatically

**Why faster-whisper over openai-whisper:**
- Same accuracy, 2-4x faster inference
- Built-in Silero VAD support
- Lower memory footprint
- CTranslate2 optimizations

---

### Stage 5A: Scene Alignment

**Module**: `src/vea/stages/align.py`

Creates hierarchical scene structure using semantic embeddings.

**Features:**
- **Global scenes**: Large units of up to 15 min for visualization structure (split on the duration cap or on a 5s silence gap)
- **Local segments**: 8-60 word coherent ideas for emotion classification
- Semantic similarity using multilingual sentence embeddings
- Word-count constraints to ensure sufficient context
- Always respects sentence boundaries (complete thoughts)

**Output:**
- `global_scenes.json` - High-level scene structure
- `local_segments_{transcription}.json` - Emotion-ready segments, e.g. `local_segments_transcription_large-v3.json`. When several transcriptions exist, `find_best_transcription` prefers a full (not time-limited) one, then the most recent, then the larger model.

**Design Rationale:**
- **Two-level hierarchy**: Global for structure, local for classification
- **Semantic embeddings**: Understands "машина" = "автомобиль" (synonyms)
- **Word-based segmentation**: 8-60 words provides optimal context for emotion
- **Sentence boundaries**: Never splits incomplete thoughts

**Alternative Approaches:**
- Word overlap: Too simplistic, misses semantic relationships
- Fixed duration: Ignores linguistic structure
- Character count: Doesn't account for information density
- Current approach: State-of-the-art semantic understanding

**Technical Details:**
- Uses `paraphrase-multilingual-MiniLM-L12-v2` for Russian embeddings
- Similarity threshold (`min_similarity_threshold`, default 0.35) balances coherence with granularity
- Segment size: minimum 8, target 25, maximum 60 words, at most 5 sentences
- Fallback to word overlap if embeddings unavailable
- Design goal (`group_shots_into_global_scenes`): reduce ~300 camera shots per hour to 10-30 global scenes

---

### Stage 5B: Translation

**Module**: `src/vea/stages/translate.py`

Translates Russian segments to English using NLLB (No Language Left Behind).

**Features:**
- NLLB-200-3.3B model by default; `VEA_TRANSLATION_MODEL` selects `nllb-3.3b`, `nllb-1.3b` or `nllb-600m` (`TRANSLATION_MODELS` in `src/vea/config.py`)
- Batch processing for efficiency
- Automatic GPU memory management: `select_gpu()` picks the visible GPU with the most free memory, and the batch size halves on out-of-memory
- Preserves segment structure and metadata

**Output:**
- `local_segments_{transcription}_translated.json` - English translations

**Design Notes:**
- NLLB chosen for multilingual quality (better than M2M-100)
- 3.3B parameter model balances quality and speed; it is roughly 17 GB of weights and wants about 8 GB of VRAM
- Beam search (num_beams=4) improves translation quality
- NLLB-200 is licensed CC-BY-NC-4.0 (non-commercial); see `docs/LICENSING.md`
- Enables cross-language emotion validation

---

### Stage 6A: Russian Intensity Classification

**Module**: `src/vea/stages/intensity_ru.py`

Predicts arousal and valence for Russian text using XLM-RoBERTa.

**Features:**
- Continuous arousal/valence values (0-1 scale)
- Video-adaptive emotion threshold (uses median arousal)
- Multilingual model processes Russian directly (no translation)
- Binary emotion significance flag for downstream filtering

**Output:**
- `arousal_{model}_local_{transcription}.json` - Arousal-valence predictions, e.g. `arousal_xlmroberta-large_local_transcription_large-v3.json` (`{model}` is the stage config's `name`, which still says `xlmroberta-large`; see below)

**Design Rationale:**
- **Continuous values**: Preserves all information (no arbitrary discretization)
- **Video-adaptive threshold**: Adjusts to content baseline automatically
- **Binary filter**: Simpler than 5-level intensity scale
- **Academic best practice**: Matches dimensional emotion theory

**Alternative Approaches:**
- 5-point intensity scale: Loses information, arbitrary thresholds
- Fixed thresholds: Don't adapt to content type
- Translation required: Adds error and latency
- Current approach: Research-backed, preserves data

**Technical Details:**
- Registry key `va-xlmroberta-large`, loaded from the local checkpoint directory `xlmroberta-base-va`. Despite the key name this is the **base-size** XLM-RoBERTa checkpoint published by Mendes & Martins (gmendes9/multilingual_va_prediction); their large checkpoint exceeds GitHub's release-asset limit. The key keeps the archive's name (see the comment in `src/vea/config.py`)
- Trained on 34 psycho-linguistic valence-arousal datasets across 100 languages, so it reads Russian without translation
- Fetched and checksum-verified by `scripts/fetch_va_checkpoint.sh`
- Its arousal output separates this project's classes at AUC 0.5734; read `docs/PROVENANCE.md` section 11 before reporting anything derived from it
- Threshold methods: `median` (default), `mean`, `percentile_60`, `mad` (`calculate_emotion_threshold`)
- Batch processing for GPU efficiency

---

### Stage 6B: English Intensity Classification

**Module**: `src/vea/stages/intensity_en.py`

Predicts arousal and valence for English translations.

**Features:**
- Same XLM-RoBERTa model as Russian classifier
- Processes translated segments for cross-language comparison
- Enables validation of translation quality impact

**Output:**
- `arousal_{model}_en_local_{transcription}.json` - English arousal-valence

**Design Notes:**
- Parallel processing enables Russian vs English comparison
- Validates whether translation affects emotion perception
- Uses multilingual model for consistency with Russian predictions

---

### Stage 7A: Russian Emotion Classification

**Module**: `src/vea/stages/emotion_ru.py`

Classifies emotions in Russian segments using ruBERT.

**Features:**
- 7 emotion classes, `EMOTION_CLASSES` in `src/vea/config.py`: anger, disgust, fear, joy, neutral, sadness, surprise. Model labels are mapped onto them by `canonicalize_emotion` (the Russian model emits `enthusiasm`, which maps to joy); an unrecognised label raises rather than defaulting to neutral
- Only processes emotionally significant segments (from Stage 6A)
- Fast inference with small model (rubert-tiny2)
- Assigns neutral to low-arousal segments without inference

**Output:**
- `emotion_{model}_local_{transcription}.json` - Russian emotion predictions, e.g. `emotion_rubert-tiny2_local_transcription_large-v3.json`

**Design Rationale:**
- **Two-stage approach**: Intensity filtering → emotion classification
- **Realistic class distribution**: 60-70% neutral in TV content
- **Efficiency**: Only classifies high-arousal segments
- **Native Russian**: Better accuracy than translated text

**Technical Details:**
- `Djacon/rubert-tiny2-russian-emotion-detection` (registry key `emotion-ru-rubert-tiny2`), a third-party Hub model used as-is; small model, fast inference
- The group's own Russian model (`emotion-ru-finetuned`) is declared in the registry for provenance but not wired in
- Confidence scores and quality metrics

---

### Stage 7B: English Emotion Classification

**Module**: `src/vea/stages/emotion_en.py`

Classifies emotions in English translations using two models.

**Features:**
- **Dual-model ensemble**: DistilRoBERTa + DeBERTa for validation
- Only processes emotionally significant segments (from Stage 6B)
- Same 7 classes as the Russian branch (`EMOTION_CLASSES`)
- English-language models applied to the NLLB translations, so translation errors carry into this branch

**Output:**
- `emotion_distilroberta_en_local_{transcription}.json`
- `emotion_deberta-finetuned_en_local_{transcription}.json`

**Design Notes:**
- Two models enable ensemble validation
- DistilRoBERTa: `j-hartmann/emotion-english-distilroberta-base`, fast, efficient (~82M parameters), fed raw text
- DeBERTa: `microsoft/deberta-v3-base` (~183M parameters) retrained for this project on the cleaned super-emotion set, loaded from the local checkpoint `emotion-en-deberta` (`scripts/fetch_emotion_en_checkpoint.sh`). Its inputs go through the `super_emotion_v1` text cleaner (`vea.text_clean`) because its training data did; DistilRoBERTa's do not
- The `deberta-finetuned` label in the filename predates this checkpoint: outputs generated before the switch came from `tae898/emoberta-large` and must be regenerated before they are reported (`docs/PROVENANCE.md` sections 1 and 10)
- Enables cross-language emotion comparison

**Why Two English Models:**
- Cross-validation improves reliability
- Detects translation-induced emotion shifts
- Provides confidence through agreement
- Research shows ensemble > single model

---

### Stage 8: Visualization

**Module**: `src/vea/stages/visualize.py`

Creates emotion timeline visualization with ensemble validation.

**Features:**
- **Two-panel layout**: Arousal (top) + Valence (bottom)
- Emotion-colored arousal timeline
- Agreement-based opacity (more opaque = higher confidence)
- Scene markers and boundaries
- Smooth interpolation for visual clarity
- Quality analysis and statistics

**Output:**
- `emotion_timeline.png` - Visualization image

**Design Elements:**
- **Arousal panel**: Shows emotional intensity with emotion colors
- **Valence panel**: Shows positive vs negative direction
- **Opacity**: Reflects model agreement (transparent = uncertain)
- **Scene markers**: Vertical lines showing global scene boundaries
- **Color smoothing**: Prevents jarring transitions

**Interpretation:**
- High arousal + high valence = Joy/Excitement
- High arousal + low valence = Fear/Anger
- Low arousal + high valence = Calm contentment
- Low arousal + low valence = Sadness/Boredom

---

### Stage 9: CSV Export

**Module**: `src/vea/stages/export.py`

Consolidates all pipeline results into a single CSV for analysis.

**Features:**
- One row per segment
- Combines transcription, translation, VA, and emotions
- Ensemble emotion calculation with agreement levels (`calculate_final_emotion`)
- Handles missing data gracefully

**Output:**
- `emotion_analysis_data.csv` - Consolidated results

**CSV Columns** (in this order, as written by `create_emotion_csv`):
- Segment metadata: `segment_id`, `start_time`, `end_time`, `duration`
- Text: `text_ru`, `text_en`
- Russian VA: `ru_arousal`, `ru_valence`
- English VA: `en_arousal`, `en_valence`
- Russian emotion: `emotion_ru`, `emotion_ru_confidence`
- English emotion: `emotion_en_distil`, `emotion_en_distil_confidence`, `emotion_en_deberta`, `emotion_en_deberta_confidence`
- Ensemble: `emotion_final`, `emotion_agreement`

Emotion cells hold display names (`EMOTION_DISPLAY_NAMES`: `Anger`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness`, `Surprise`; joy is written as `Happiness`).

`emotion_agreement` is one of:
- `full` - all available models agree (at least two)
- `majority` - two of three agree
- `confidence` - all disagree; the highest-confidence prediction is used
- `single` - only one model produced a prediction

(`no_data` is returned when no model produced a prediction for the segment.)

**Usage Examples:**
```python
import pandas as pd

# Load results
df = pd.read_csv("downloads/video-ID/emotion_analysis_data.csv")

# Filter high-confidence predictions
confident = df[df["emotion_agreement"] == "full"]

# Analyze emotion distribution
df["emotion_final"].value_counts()

# Compare Russian vs English emotions
disagreements = df[df["emotion_ru"] != df["emotion_en_distil"]]
```

---

## Module Documentation

### Core Modules

All paths are relative to the repository root; the package is `vea` (`src/vea/`).

#### `src/vea/cli.py`
The `vea` command (`[project.scripts]` in `pyproject.toml`): `run`, `config`, `models`, `serve`. `config` and `models` do not import the pipeline, so they work without torch installed.

#### `src/vea/pipeline.py`
Pipeline orchestrator (`main(youtube_url, configs=None)`) that runs all 9 stages sequentially, plus `DEFAULT_CONFIGS` and one `run_stage_*` wrapper per stage. Loads the shared models once, validates each stage output, and handles errors gracefully.

#### `src/vea/config.py`
Runtime settings from `VEA_*` environment variables (`Settings`, `get_settings`), the model registry (`MODEL_REGISTRY`, `resolve_model`, `missing_checkpoints`), `resolve_device`, `apply_device_setting`, `configure_logging`, and the emotion vocabulary (`EMOTION_CLASSES`, `EMOTION_ALIASES`, `canonicalize_emotion`). Imports neither torch nor transformers at module level.

#### `src/vea/text_clean.py`
The `super_emotion_v1` text cleaner applied to the DeBERTa member's inputs in stage 7B.

#### `src/vea/stages/download.py`
YouTube downloader using yt-dlp with format fallback and metadata extraction.

#### `src/vea/stages/scenes.py`
Scene boundary detection using PySceneDetect with adaptive thresholding.

#### `src/vea/stages/audio.py`
Audio normalization and optional noise reduction.

#### `src/vea/stages/transcribe.py`
Russian speech transcription using faster-whisper with anti-hallucination measures.

#### `src/vea/stages/align.py`
Hierarchical scene segmentation using semantic embeddings.

#### `src/vea/stages/translate.py`
Russian to English translation using NLLB (`NLLBTranslator`).

#### `src/vea/stages/intensity_ru.py`
Russian arousal-valence prediction using XLM-RoBERTa.

#### `src/vea/stages/intensity_en.py`
English arousal-valence prediction using XLM-RoBERTa.

#### `src/vea/stages/emotion_ru.py`
Russian emotion classification using ruBERT.

#### `src/vea/stages/emotion_en.py`
English emotion classification using DistilRoBERTa and DeBERTa.

#### `src/vea/stages/visualize.py`
Emotion timeline visualization with ensemble validation.

#### `src/vea/stages/export.py`
CSV export consolidating all pipeline results.

#### `src/vea/api/`
The HTTP API behind `vea serve` (`app.py` routes, `jobs.py` job queue, `runs.py` reading finished runs). See [Web Interface](#web-interface).

---

## Output Files

### Directory Structure

```
downloads/
└── video-{VIDEO_ID}/
    ├── video.{ext}                                  # Downloaded video (mp4/webm)
    ├── audio.wav                                    # Original audio
    ├── metadata.json                                # Video metadata
    ├── preprocessed_audio.wav                       # Normalized audio
    ├── audio_quality.json                           # Quality metrics
    ├── scene_boundaries.json                        # Scene detection
    ├── transcription_large-v3.json                  # Transcription (full run)
    ├── global_scenes.json                           # Global scenes
    ├── local_segments_transcription_large-v3.json  # Russian segments
    ├── local_segments_*_translated.json            # English segments
    ├── arousal_xlmroberta-large_local_*.json       # Russian VA
    ├── arousal_xlmroberta-large_en_local_*.json    # English VA
    ├── emotion_rubert-tiny2_local_*.json           # Russian emotion
    ├── emotion_distilroberta_en_local_*.json       # English emotion 1
    ├── emotion_deberta-finetuned_en_local_*.json   # English emotion 2
    ├── emotion_timeline.png                         # Visualization
    └── emotion_analysis_data.csv                    # Final results
```

A time-limited run writes `transcription_large-v3_{N}s.json` instead, and every downstream name carries that base (e.g. `local_segments_transcription_large-v3_{N}s.json`).

### Key File Formats

All JSON files follow consistent schemas:
- `metadata` section with configuration
- `statistics` section with processing stats
- `predictions`/`segments`/`scenes` arrays with results

---

## Web Interface

A browser front end ships with the pipeline: a FastAPI service in `src/vea/api/` started by `vea serve`, and a Next.js app in `frontend/`.

```bash
uv sync --extra api
uv run vea serve                              # API on http://127.0.0.1:8000
cd frontend && npm install && npm run dev     # UI on http://localhost:3000
```

- `POST /api/jobs` with `{"url": ...}` queues a run. Jobs run one at a time in submission order; each is a subprocess executing `python -m vea.cli run <url>`, so it goes through the same preflight checks as the CLI. `GET /api/jobs/{id}/events` streams job state and log lines (SSE); `DELETE /api/jobs/{id}` cancels.
- `GET /api/runs`, `/api/runs/{video_id}`, `/api/runs/{video_id}/segments` and `/api/runs/{video_id}/timeline` read finished runs from `downloads/` (`vea serve --downloads` changes the directory). Emotion labels are canonicalised through `vea.config` before they leave the API.
- `GET /api/meta` and `GET /api/health` report the stage list, emotion vocabulary and whether checkpoints are present.

The API has no authentication and fetches caller-supplied URLs; it binds to `127.0.0.1` by default and must not be exposed. The full route list is in the `src/vea/api/app.py` docstring.

---

## Configuration

### Default Configuration

The pipeline uses optimized defaults in `DEFAULT_CONFIGS` in `src/vea/pipeline.py`. Key parameters (excerpt):

```python
DEFAULT_CONFIGS = {
    "stage_4_transcription": {
        "model_size": "large-v3",
        "language": "ru",
        "duration_limit": "full",
        "device": None,
        "compute_type": "float16",
    },
    "stage_5a_scene_alignment": {
        "max_global_scene_duration": 900.0,
        "min_gap_for_global_split": 5.0,
        "min_words_per_segment": 8,
        "target_words_per_segment": 25,
        "max_words_per_segment": 60,
        "max_sentences_per_segment": 5,
        "min_similarity_threshold": 0.35,
        "force_reprocess": False,
    },
    "stage_5b_translation": {
        "model_name": translation_model_ref(),  # from VEA_TRANSLATION_MODEL
        "num_beams": 4,
        "batch_size": 32,
        ...
    },
    "stage_6a_russian_intensity": {
        "name": "xlmroberta-large",
        "model": "va-xlmroberta-large",  # key in vea.config.MODEL_REGISTRY
        "config": {"batch_size": 32, "device": None,
                   "skip_advertisements": True, "threshold_method": "median"},
        ...
    },
    ...
}
```

Stages 6 and 7 name models by their `MODEL_REGISTRY` key rather than by path; `_resolve_stage_model` turns the key into a loadable path when the stage starts, so a missing checkpoint fails that stage with a named error. `stage_5b_translation.model_name` is evaluated when `vea.pipeline` is imported, so `VEA_TRANSLATION_MODEL` must be set before that.

### Environment Variables

Read by `Settings.from_env` in `src/vea/config.py`; `uv run vea config` prints the resolved values.

| Variable | Default | Meaning |
|---|---|---|
| `VEA_MODELS_DIR` | `models/` | where local checkpoints are looked up |
| `VEA_DATA_DIR` | `data/` | holds the API's job store (`$VEA_DATA_DIR/jobs`); pipeline output still goes to `downloads/` |
| `VEA_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`; see [GPU Configuration](#gpu-configuration) for what each does |
| `VEA_LOG_LEVEL` | `INFO` | logging level |
| `VEA_TRANSLATION_MODEL` | `nllb-3.3b` | `nllb-3.3b`, `nllb-1.3b` or `nllb-600m` |

### Custom Configuration

`main(youtube_url, configs=None)` uses `configs` as given; it does **not** merge a partial dict with the defaults, so every stage key must be present. Start from a deep copy of the defaults and change what you need:

```python
import copy

from vea.config import configure_logging
from vea.pipeline import DEFAULT_CONFIGS, main

configure_logging("INFO")  # the CLI does this for you; library code does not

configs = copy.deepcopy(DEFAULT_CONFIGS)
configs["stage_4_transcription"]["model_size"] = "medium"  # Faster, less accurate
configs["stage_4_transcription"]["duration_limit"] = 300  # First 5 minutes only

ok = main(youtube_url, configs=configs)  # True on success
```

This bypasses the checkpoint and URL checks that `vea run` performs. Use `copy.deepcopy`: the stage dicts are nested, so a shallow copy would modify `DEFAULT_CONFIGS` itself.

---

## Technical Details

### Hardware Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 16GB
- GPU: 8GB VRAM (GTX 1080 or better)
- Storage: 50GB free space

**Recommended:**
- CPU: 8+ cores
- RAM: 32GB
- GPU: 24GB VRAM (RTX 3090/4090 or A5000)
- Storage: 100GB+ for multiple videos

### Processing Times

Approximate times for 1-hour video on RTX 6000 Ada:
- Stage 1 (Download): 2-5 minutes
- Stage 2 (Scene Detection): 3-5 minutes
- Stage 3 (Audio Preprocessing): 1 minute
- Stage 4 (Transcription): 5-10 minutes
- Stage 5A (Scene Alignment): 1-2 minutes
- Stage 5B (Translation): 2-5 minutes
- Stage 6A+6B (Intensity): 2-3 minutes
- Stage 7A+7B (Emotion): 2-3 minutes
- Stage 8 (Visualization): 30 seconds
- Stage 9 (CSV Export): 10 seconds

**Total: ~20-35 minutes per hour of video**

### Model Storage

Where each model is stored, as far as the code determines it:
- Whisper (stage 4): `WhisperModel(model_size, ...)` is called without `download_root`, so faster-whisper downloads the model through the Hugging Face Hub cache (`HF_HOME`, default under `~/.cache/huggingface/`)
- NLLB (stage 5B): `NLLBTranslator` passes `cache_dir="./models_cache"`, so the translator is cached in `models_cache/` under the working directory, not in `HF_HOME`
- Hub classifiers (stages 7A, 7B DistilRoBERTa): `from_pretrained` without `cache_dir`, i.e. the Hugging Face cache
- Sentence transformers (stage 5A): `SentenceTransformer(model_name)` without `cache_folder`, i.e. the library's default cache
- Local checkpoints (stages 6A/6B, 7B DeBERTa): `VEA_MODELS_DIR` (default `models/`), directories `xlmroberta-base-va/` and `emotion-en-deberta/`

Total storage: NLLB-200-3.3B alone is roughly 17 GB and Whisper large-v3 about 3 GB, so plan for more than 20 GB of model weights with the default translator. `VEA_TRANSLATION_MODEL=nllb-600m` reduces the translator to about 2.5 GB.

---

## Troubleshooting

### Common Issues

**Issue: Whisper hallucinations (repetitive text)**
- Solution: Already implemented VAD filtering and `condition_on_previous_text=False`
- If still occurring: Adjust the VAD parameters in `transcribe_audio` (`src/vea/stages/transcribe.py`; they are not exposed in the stage config) or use a shorter `duration_limit`

**Issue: `vea run` refuses to start ("checkpoints that a stage needs are missing")**
- Solution: Run `uv run vea models`, then fetch the missing weights with `scripts/fetch_va_checkpoint.sh` / `scripts/fetch_emotion_en_checkpoint.sh`
- Or set `VEA_MODELS_DIR` if the checkpoints live elsewhere

**Issue: CUDA out of memory**
- Solution: Reduce batch sizes in configs
- Stage 4: Use `compute_type='int8'` instead of `float16`
- Stages 6-7: Reduce `batch_size` (under each stage's `config`) to 16 or 8
- Pick a less busy card: `export CUDA_VISIBLE_DEVICES="$(scripts/pick_free_gpu.sh)"`

**Issue: Missing translations (Stage 5B fails)**
- Solution: Check GPU memory and NLLB model download
- Try a smaller NLLB model: `VEA_TRANSLATION_MODEL=nllb-1.3b` or `nllb-600m`

**Issue: Low emotion prediction quality**
- Check: Are intensity thresholds appropriate?
- Solution: Adjust `threshold_method` in Stage 6 config (`median`, `mean`, `percentile_60`, `mad`)
- Review: CSV export to analyze confidence scores

**Issue: Slow processing**
- Solution: Use GPU (`uv run vea config` shows whether torch sees CUDA; check `CUDA_VISIBLE_DEVICES`)
- Reduce model sizes: medium Whisper, a smaller NLLB via `VEA_TRANSLATION_MODEL`
- Process shorter videos or use `duration_limit`

### Logging

All modules log through `logging.getLogger(__name__)`; only the entry point configures handlers. Increase verbosity:

```bash
uv run vea run "<url>" --log-level DEBUG     # or: export VEA_LOG_LEVEL=DEBUG
```

From Python, call `vea.config.configure_logging("DEBUG")` before `main`.

### Validation

Each module includes validation functions. Check for errors:

```python
# Validate transcription
from vea.stages.transcribe import validate_transcription

result = validate_transcription(transcription_data)
print(result)  # List of validation errors
```

---

## Future Improvements

### Planned Enhancements

1. **Speaker diarization**: Separate emotions by speaker
2. **Visual emotion analysis**: Facial expression detection
3. **Temporal smoothing**: Reduce emotion jitter in timeline
4. **Multi-language support**: Extend beyond Russian
5. **Real-time processing**: Streaming pipeline for live content

The web interface that used to be listed here has shipped; see [Web Interface](#web-interface).

### Research Opportunities

- Cross-language emotion perception differences
- Translation impact on emotion detection accuracy
- Optimal segment length for emotion classification
- Multi-modal fusion (audio + visual + text)

---

## Citation

If you use this pipeline in research, please cite:

```
Required Citations:

1. Emotion English DistilRoBERTa-base:
   Hartmann, J. (2022). Emotion English DistilRoBERTa-base.
   https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/

2. Super Emotion Dataset:
   Junqué de Fortuny, E. (2025). The Super Emotion Dataset.
   https://huggingface.co/datasets/cirimus/super-emotion
   Licensed under CC BY-SA 4.0

3. Russian Izard Emotions Dataset:
   Djacon (2023). ru-izard-emotions dataset.
   https://huggingface.co/datasets/Djacon/ru-izard-emotions
   Licensed under MIT

4. Multilingual Valence-Arousal Prediction:
   Mendes, G., & Martins, B. Quantifying Valence and Arousal in Text 
   with Multilingual Pre-trained Transformers.
   https://github.com/gmendes9/multilingual_va_prediction

5. Microsoft DeBERTa:
   He, P., Gao, J., & Chen, W. (2021). DeBERTaV3.
   https://github.com/microsoft/DeBERTa
   Licensed under MIT
```

---

## License

This work is licensed under Creative Commons Attribution-ShareAlike 4.0 
International (CC BY-SA 4.0) to comply with the most restrictive upstream 
dataset license (super-emotion dataset).

### Components and Their Licenses

- Pipeline Code: covered by the repository licence, CC BY-SA 4.0 (`LICENSE`); it could be MIT on its own only if no CC BY-SA 4.0 data or weights shipped with it (see `docs/LICENSING.md`)
- Trained Models: CC BY-SA 4.0 (derived from super-emotion dataset)
- NLLB-200 (stage 5B): CC-BY-NC-4.0, non-commercial; see `docs/LICENSING.md`
