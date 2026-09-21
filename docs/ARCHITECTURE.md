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

**Modularity**: Each stage is independent and can be run separately. Stages validate previous outputs and fail gracefully.

**Skip Logic**: Every module checks if output already exists with matching configuration. This enables fast iteration and prevents redundant processing.

**Error Isolation**: One stage failure doesn't crash the pipeline. Each stage returns clear success/failure status.

**Configuration Transparency**: All parameters are logged and saved to output files for reproducibility.

**Resource Management**: Models are loaded once and reused across all videos for efficiency.

---

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- Ubuntu/Linux (tested on Ubuntu)
- ~50GB disk space for models and data

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download required models (first run will auto-download)
# - Whisper large-v3 (~3GB)
# - NLLB-200-3.3B (~6GB)
# - XLM-RoBERTa models (~1GB)
# - Emotion classifiers (~500MB)
```

### GPU Configuration

The pipeline uses GPU 5 by default. Modify `os.environ['CUDA_VISIBLE_DEVICES']` in module files to use different GPU:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 instead
```

---

## Quick Start

### Basic Usage

```bash
# Process a single video
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Example

```bash
python main.py "https://www.youtube.com/watch?v=mqGSkDFeLEo"
```

### Output Location

All files are saved to `downloads/video-{VIDEO_ID}/`:
- Metadata, audio, video files
- Scene boundaries, transcriptions
- Intensity and emotion predictions
- Visualization and CSV export

---

## Pipeline Stages

### Stage 1: YouTube Download

**Module**: `download_module.py`

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

**Module**: `scene_detector_module.py`

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

**Module**: `audio_preprocess_module.py`

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

**Module**: `transcribe_module.py`

Transcribes Russian speech using faster-whisper (CTranslate2 implementation of OpenAI Whisper).

**Features:**
- faster-whisper for 2-4x speedup over openai-whisper
- VAD filtering to remove silence (reduces hallucinations)
- Word-level timestamps for precise alignment
- Anti-hallucination measures for long videos
- Quality monitoring and hallucination detection

**Output:**
- `transcription_{model}_{duration}.json` - Transcription with word timestamps

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
- VAD parameters tuned for speech preservation over aggressive filtering

**Why faster-whisper over openai-whisper:**
- Same accuracy, 2-4x faster inference
- Built-in Silero VAD support
- Lower memory footprint
- CTranslate2 optimizations

---

### Stage 5A: Scene Alignment

**Module**: `scene_align_module.py`

Creates hierarchical scene structure using semantic embeddings.

**Features:**
- **Global scenes**: 10-15 min semantic units for visualization structure
- **Local segments**: 8-60 word coherent ideas for emotion classification
- Semantic similarity using multilingual sentence embeddings
- Word-count constraints to ensure sufficient context
- Always respects sentence boundaries (complete thoughts)

**Output:**
- `global_scenes.json` - High-level scene structure
- `local_segments_{transcription}.json` - Emotion-ready segments

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
- Similarity threshold (0.35-0.4) balances coherence with granularity
- Fallback to word overlap if embeddings unavailable
- Aggregates 300+ camera shots into 30-50 semantic scenes per hour

---

### Stage 5B: Translation

**Module**: `translate_module.py`

Translates Russian segments to English using NLLB (No Language Left Behind).

**Features:**
- NLLB-200-3.3B model for high-quality translation
- Batch processing for efficiency
- Automatic GPU memory management
- Preserves segment structure and metadata

**Output:**
- `local_segments_{transcription}_translated.json` - English translations

**Design Notes:**
- NLLB chosen for multilingual quality (better than M2M-100)
- 3.3B parameter model balances quality and speed
- Beam search (num_beams=4) improves translation quality
- Enables cross-language emotion validation

---

### Stage 6A: Russian Intensity Classification

**Module**: `intensity_classifier_ru.py`

Predicts arousal and valence for Russian text using XLM-RoBERTa.

**Features:**
- Continuous arousal/valence values (0-1 scale)
- Video-adaptive emotion threshold (uses median arousal)
- Multilingual model processes Russian directly (no translation)
- Binary emotion significance flag for downstream filtering

**Output:**
- `arousal_{model}_local_{transcription}.json` - Arousal-valence predictions

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
- XLM-RoBERTa large (560M parameters)
- Trained on valence-arousal datasets
- Works with Russian, English, and 100+ languages
- Batch processing for GPU efficiency

---

### Stage 6B: English Intensity Classification

**Module**: `intensity_classifier_en.py`

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

**Module**: `emotion_classifier_ru.py`

Classifies emotions in Russian segments using ruBERT.

**Features:**
- 7-class emotion model (anger, fear, joy, love, neutral, sadness, surprise)
- Only processes emotionally significant segments (from Stage 6A)
- Fast inference with small model (rubert-tiny2)
- Assigns neutral to low-arousal segments without inference

**Output:**
- `emotion_{model}_local_{transcription}.json` - Russian emotion predictions

**Design Rationale:**
- **Two-stage approach**: Intensity filtering → emotion classification
- **Realistic class distribution**: 60-70% neutral in TV content
- **Efficiency**: Only classifies high-arousal segments
- **Native Russian**: Better accuracy than translated text

**Technical Details:**
- ruBERT-tiny2: 29M parameters, fast inference
- Trained on Russian emotion datasets
- 7 super-emotion categories from academic literature
- Confidence scores and quality metrics

---

### Stage 7B: English Emotion Classification

**Module**: `emotion_classifier_en.py`

Classifies emotions in English translations using two models.

**Features:**
- **Dual-model ensemble**: DistilRoBERTa + DeBERTa for validation
- Only processes emotionally significant segments (from Stage 6B)
- 7-class emotion model matching Russian categories
- Native English processing (no translation artifacts)

**Output:**
- `emotion_distilroberta_en_local_{transcription}.json`
- `emotion_deberta_en_local_{transcription}.json`

**Design Notes:**
- Two models enable ensemble validation
- DistilRoBERTa: Fast, efficient (82M parameters)
- DeBERTa: More accurate (larger model)
- Enables cross-language emotion comparison

**Why Two English Models:**
- Cross-validation improves reliability
- Detects translation-induced emotion shifts
- Provides confidence through agreement
- Research shows ensemble > single model

---

### Stage 8: Visualization

**Module**: `visualization.py`

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

**Module**: `csv_generator.py`

Consolidates all pipeline results into a single CSV for analysis.

**Features:**
- One row per segment
- Combines transcription, translation, VA, and emotions
- Ensemble emotion calculation with agreement levels
- Handles missing data gracefully

**Output:**
- `emotion_analysis_data.csv` - Consolidated results

**CSV Columns:**
- Segment metadata: `segment_id`, `start_time`, `end_time`, `duration`
- Text: `text_ru`, `text_en`
- Russian VA: `ru_arousal`, `ru_valence`
- English VA: `en_arousal`, `en_valence`
- Russian emotion: `emotion_ru`, `confidence_ru`
- English emotion: `emotion_en_distil`, `emotion_en_deberta`
- Ensemble: `emotion_final`, `emotion_agreement`

**Usage Examples:**
```python
import pandas as pd

# Load results
df = pd.read_csv("downloads/video-ID/emotion_analysis_data.csv")

# Filter high-confidence predictions
confident = df[df["emotion_agreement"] == "full_agreement"]

# Analyze emotion distribution
df["emotion_final"].value_counts()

# Compare Russian vs English emotions
disagreements = df[df["emotion_ru"] != df["emotion_en_distil"]]
```

---

## Module Documentation

### Core Modules

#### `main.py`
Pipeline orchestrator that runs all 9 stages sequentially. Loads models once, validates each stage output, and handles errors gracefully.

#### `download_module.py`
YouTube downloader using yt-dlp with format fallback and metadata extraction.

#### `scene_detector_module.py`
Scene boundary detection using PySceneDetect with adaptive thresholding.

#### `audio_preprocess_module.py`
Audio normalization and optional noise reduction.

#### `transcribe_module.py`
Russian speech transcription using faster-whisper with anti-hallucination measures.

#### `scene_align_module.py`
Hierarchical scene segmentation using semantic embeddings.

#### `translate_module.py`
Russian to English translation using NLLB.

#### `intensity_classifier_ru.py`
Russian arousal-valence prediction using XLM-RoBERTa.

#### `intensity_classifier_en.py`
English arousal-valence prediction using XLM-RoBERTa.

#### `emotion_classifier_ru.py`
Russian emotion classification using ruBERT.

#### `emotion_classifier_en.py`
English emotion classification using DistilRoBERTa and DeBERTa.

#### `visualization.py`
Emotion timeline visualization with ensemble validation.

#### `csv_generator.py`
CSV export consolidating all pipeline results.

---

## Output Files

### Directory Structure

```
downloads/
└── video-{VIDEO_ID}/
    ├── video.mp4                                    # Downloaded video
    ├── audio.wav                                    # Original audio
    ├── metadata.json                                # Video metadata
    ├── preprocessed_audio.wav                       # Normalized audio
    ├── audio_quality.json                           # Quality metrics
    ├── scene_boundaries.json                        # Scene detection
    ├── transcription_large-v3_full.json            # Transcription
    ├── global_scenes.json                           # Global scenes
    ├── local_segments_transcription_large-v3.json  # Russian segments
    ├── local_segments_*_translated.json            # English segments
    ├── arousal_xlmroberta-large_local_*.json       # Russian VA
    ├── arousal_xlmroberta-large_en_local_*.json    # English VA
    ├── emotion_rubert-tiny2_local_*.json           # Russian emotion
    ├── emotion_distilroberta_en_local_*.json       # English emotion 1
    ├── emotion_deberta_en_local_*.json             # English emotion 2
    ├── emotion_timeline.png                         # Visualization
    └── emotion_analysis_data.csv                    # Final results
```

### Key File Formats

All JSON files follow consistent schemas:
- `metadata` section with configuration
- `statistics` section with processing stats
- `predictions`/`segments`/`scenes` arrays with results

---

## Configuration

### Default Configuration

The pipeline uses optimized defaults in `main.py`. Key parameters:

```python
DEFAULT_CONFIGS = {
    "stage_4_transcription": {
        "model_size": "large-v3",
        "language": "ru",
        "compute_type": "float16",
    },
    "stage_5a_scene_alignment": {
        "min_words_per_segment": 8,
        "target_words_per_segment": 16,
        "min_similarity_threshold": 0.4,
    },
    "stage_6a_russian_intensity": {"threshold_method": "median", "batch_size": 32},
}
```

### Custom Configuration

Modify configurations in `main.py` or pass custom config dict:

```python
custom_config = {
    "stage_4_transcription": {
        "model_size": "medium",  # Faster, less accurate
        "duration_limit": 300,  # Process first 5 minutes only
    }
}

main(youtube_url, configs=custom_config)
```

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

Models are cached in default locations:
- Whisper: `~/.cache/whisper/`
- Transformers: `~/.cache/huggingface/`
- Sentence transformers: `~/.cache/torch/sentence_transformers/`

Total storage: ~15GB for all models

---

## Troubleshooting

### Common Issues

**Issue: Whisper hallucinations (repetitive text)**
- Solution: Already implemented VAD filtering and `condition_on_previous_text=False`
- If still occurring: Increase VAD threshold or use shorter duration_limit

**Issue: CUDA out of memory**
- Solution: Reduce batch sizes in configs
- Stage 4: Use `compute_type='int8'` instead of `float16`
- Stages 6-7: Reduce `batch_size` to 16 or 8

**Issue: Missing translations (Stage 5B fails)**
- Solution: Check GPU memory and NLLB model download
- Try smaller NLLB model: `facebook/nllb-200-1.3B`

**Issue: Low emotion prediction quality**
- Check: Are intensity thresholds appropriate?
- Solution: Adjust `threshold_method` in Stage 6 config
- Review: CSV export to analyze confidence scores

**Issue: Slow processing**
- Solution: Use GPU (check `CUDA_VISIBLE_DEVICES`)
- Reduce model sizes: medium Whisper, base XLM-RoBERTa
- Process shorter videos or use `duration_limit`

### Logging

All modules use Python logging. Increase verbosity:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Validation

Each module includes validation functions. Check for errors:

```python
# Validate transcription
from transcribe_module import validate_transcription

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
6. **Web interface**: User-friendly UI for video upload and analysis

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

- Pipeline Code: MIT License (but models/data are CC BY-SA 4.0)
- Trained Models: CC BY-SA 4.0 (derived from super-emotion dataset)
