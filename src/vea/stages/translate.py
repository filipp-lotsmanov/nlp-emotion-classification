"""
NLLB-200 Translation Module

Design approach:
- Uses num_beams=4 (research shows this is optimal for NLLB quality vs speed)
- Proper GPU memory detection using torch.cuda.mem_get_info()
- Simple segment-by-segment translation (context windows are unreliable)
- Auto batch size reduction on OOM
- Clean input/output format without irrelevant metadata

Why this approach:
1. Beam search (num_beams=4) provides better quality than greedy (num_beams=1)
2. Context windows often fail because markers get translated/lost
3. Per-segment translation is more reliable and easier to debug
4. Batch processing with OOM handling maximizes throughput

Alternative approaches considered:
- Context windows: Too unreliable, markers often disappear in translation
- Temperature/sampling: Not compatible with beam search, lower quality
- Pipeline API: Less control over batch processing and error handling
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Logging is configured by the CLI (vea.cli) or by the calling application.
# Library modules must not call logging.basicConfig at import time.
logger = logging.getLogger(__name__)


# ============================================================================
# GPU SELECTION
# ============================================================================


def get_actual_gpu_memory() -> List[Dict[str, float]]:
    """
    Get actual free memory for each GPU (includes all processes).

    Uses torch.cuda.mem_get_info() which queries the GPU driver directly,
    unlike memory_reserved() which only shows current process allocation.
    """
    if not torch.cuda.is_available():
        return []

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)

        props = torch.cuda.get_device_properties(i)

        gpu_info.append(
            {
                "id": i,
                "name": props.name,
                "free_gb": free_bytes / 1e9,
                "total_gb": total_bytes / 1e9,
                "used_gb": (total_bytes - free_bytes) / 1e9,
            }
        )

    return gpu_info


def select_gpu(min_memory_gb: float = 8.0, preferred_gpu: Optional[int] = None) -> str:
    """
    Select best available GPU with sufficient memory.

    Args:
        min_memory_gb: Minimum free memory required
        preferred_gpu: Preferred GPU index (will use if available)

    Returns:
        Device string ('cuda:N' or 'cpu')
    """
    gpu_info = get_actual_gpu_memory()

    if not gpu_info:
        logger.warning("No CUDA GPUs available, using CPU")
        return "cpu"

    # Log all GPU status
    logger.info("GPU memory status:")
    for info in gpu_info:
        logger.info(
            f"  GPU {info['id']} ({info['name']}): "
            f"{info['free_gb']:.1f}GB free / {info['total_gb']:.1f}GB total"
        )

    # Check preferred GPU first
    if preferred_gpu is not None:
        if preferred_gpu < len(gpu_info):
            info = gpu_info[preferred_gpu]
            if info["free_gb"] >= min_memory_gb:
                device = f"cuda:{preferred_gpu}"
                logger.info(f"Using preferred GPU {preferred_gpu}")
                return device
            else:
                logger.warning(
                    f"Preferred GPU {preferred_gpu} has insufficient memory "
                    f"({info['free_gb']:.1f}GB < {min_memory_gb:.1f}GB)"
                )

    # Find GPU with most free memory
    available = [g for g in gpu_info if g["free_gb"] >= min_memory_gb]

    if not available:
        logger.error(
            f"No GPU has {min_memory_gb:.1f}GB free memory. "
            "Consider reducing batch size or using CPU."
        )
        return "cpu"

    best = max(available, key=lambda x: x["free_gb"])
    device = f"cuda:{best['id']}"
    logger.info(f"Selected GPU {best['id']} with {best['free_gb']:.1f}GB free")

    return device


# ============================================================================
# NLLB TRANSLATOR
# ============================================================================


class NLLBTranslator:
    """
    NLLB-200 translator with proper beam search and batch processing.

    Key parameters:
    - num_beams=4: Optimal quality/speed trade-off for NLLB
    - no_repeat_ngram_size=3: Prevents repetition loops
    - max_length=200: Conservative limit for output
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-3.3B",
        device: Optional[str] = None,
        preferred_gpu: Optional[int] = None,
    ):
        self.model_name = model_name

        # Estimate memory requirements
        memory_map = {
            "facebook/nllb-200-3.3B": 8.0,
            "facebook/nllb-200-1.3B": 4.0,
            "facebook/nllb-200-distilled-1.3B": 4.0,
            "facebook/nllb-200-distilled-600M": 2.5,
        }
        required_memory = memory_map.get(model_name, 10.0)

        # Select device
        if device is None:
            self.device = select_gpu(required_memory, preferred_gpu)
        else:
            self.device = device

        logger.info(f"Initializing {model_name}")
        logger.info(f"Device: {self.device}")

        # Load model
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer from HuggingFace."""
        cache_dir = Path("./models_cache")
        cache_dir.mkdir(exist_ok=True)

        logger.info("Loading model (may take a few minutes on first run)...")

        # Load tokenizer
        # Important: Don't set src_lang/tgt_lang here, set per-batch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=str(cache_dir))

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, cache_dir=str(cache_dir)
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters()) / 1e9
        logger.info(f"Model loaded: {params:.1f}B parameters")

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str = "rus_Cyrl",
        tgt_lang: str = "eng_Latn",
        batch_size: int = 32,
        num_beams: int = 4,
        show_progress: bool = False,
    ) -> List[str]:
        """
        Translate batch of texts with auto batch size reduction on OOM.

        Why num_beams=4:
        - Research shows 4-5 beams optimal for NLLB quality
        - Greedy (num_beams=1) is faster but significantly lower quality
        - Higher beams (>5) give minimal improvement at high cost

        Args:
            texts: List of texts to translate
            src_lang: Source language code (e.g., 'rus_Cyrl')
            tgt_lang: Target language code (e.g., 'eng_Latn')
            batch_size: Initial batch size (auto-reduces on OOM)
            num_beams: Beam search width (4 recommended)
            show_progress: Log progress updates

        Returns:
            List of translated texts
        """
        if not texts:
            return []

        # Set source language on tokenizer
        self.tokenizer.src_lang = src_lang

        # Get target language token for forced BOS
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        translations = []
        current_batch_size = batch_size
        i = 0

        while i < len(texts):
            batch_texts = texts[i : i + current_batch_size]

            try:
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,  # NLLB max input length
                ).to(self.device)

                # Generate with beam search
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        num_beams=num_beams,
                        max_length=200,
                        no_repeat_ngram_size=3,  # Prevent repetition
                        early_stopping=True,  # Stop when all beams finish
                    )

                # Decode
                batch_translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                translations.extend(batch_translations)
                i += current_batch_size

                if show_progress and len(translations) % (5 * batch_size) < current_batch_size:
                    logger.info(
                        f"Progress: {len(translations)}/{len(texts)} "
                        f"(batch_size={current_batch_size})"
                    )

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()

                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        logger.warning(f"OOM error, reducing batch_size to {current_batch_size}")
                        # Don't increment i - retry same batch
                    else:
                        logger.error("OOM even at batch_size=1")
                        raise
                else:
                    raise

        return translations

    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, "model"):
            del self.model
            if "cuda" in self.device:
                torch.cuda.empty_cache()
            logger.info("Model removed from memory")


# ============================================================================
# FILE PROCESSING
# ============================================================================


def translate_segments_file(
    segments_path: Path,
    translator: NLLBTranslator,
    src_lang: str = "rus_Cyrl",
    tgt_lang: str = "eng_Latn",
    batch_size: int = 32,
    num_beams: int = 4,
) -> Dict:
    """
    Translate segments file.

    Input format expected:
    {
        "metadata": {...},
        "segments": [
            {
                "text": "Russian text",
                "start": 0.0,
                "end": 1.0,
                ...
            }
        ]
    }

    Output adds 'text_en' field to each segment.
    Removes irrelevant fields like original_shot_count.
    """
    logger.info(f"Translating: {segments_path.name}")

    with open(segments_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])

    if not segments:
        logger.warning("No segments found")
        return data

    # Extract texts to translate
    texts_to_translate = []
    translate_indices = []

    for i, segment in enumerate(segments):
        text = segment.get("text", "").strip()
        if text and segment.get("has_speech", True):
            texts_to_translate.append(text)
            translate_indices.append(i)

    if not texts_to_translate:
        logger.warning("No text to translate")
        return data

    logger.info(f"Translating {len(texts_to_translate)} segments...")

    # Translate
    start_time = time.time()
    translations = translator.translate_batch(
        texts_to_translate,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        batch_size=batch_size,
        num_beams=num_beams,
        show_progress=True,
    )
    elapsed = time.time() - start_time

    logger.info(f"Translation complete: {elapsed:.1f}s, {len(translations) / elapsed:.1f} seg/s")

    # Update segments (remove irrelevant fields)
    relevant_fields = {
        "start",
        "end",
        "duration",
        "text",
        "has_speech",
        "sentence_count",
        "word_count",
        "confidence_average",
    }

    translated_segments = []
    translation_idx = 0

    for i, segment in enumerate(segments):
        # Keep only relevant fields
        clean_segment = {k: v for k, v in segment.items() if k in relevant_fields}

        # Add translation if available
        if i in translate_indices:
            clean_segment["text_en"] = translations[translation_idx]
            translation_idx += 1
        else:
            clean_segment["text_en"] = ""

        translated_segments.append(clean_segment)

    # Build output
    output = {
        "metadata": {
            **data.get("metadata", {}),
            "translation": {
                "source_language": src_lang.split("_")[0],  # 'rus'
                "target_language": tgt_lang.split("_")[0],  # 'eng'
                "model": translator.model_name,
                "num_beams": num_beams,
                "batch_size": batch_size,
                "date": datetime.utcnow().isoformat() + "Z",
                "processing_time_seconds": elapsed,
            },
        },
        "segments": translated_segments,
        "stats": {
            "total_segments": len(translated_segments),
            "translated": len(translations),
            "skipped": len(translated_segments) - len(translations),
            "speed_segments_per_second": len(translations) / elapsed,
        },
    }

    # Save
    output_path = segments_path.parent / segments_path.name.replace(".json", "_translated.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved: {output_path.name}")

    return output


def process_directory(
    data_dir: Path,
    translator: NLLBTranslator,
    file_pattern: str = "local_segments_*.json",
    skip_existing: bool = True,
    **kwargs,
):
    """
    Process all matching files in directory.

    Args:
        data_dir: Directory to search
        translator: NLLB translator instance
        file_pattern: Glob pattern for files to process
        skip_existing: Skip files that already have _translated.json
        **kwargs: Additional args passed to translate_segments_file
    """
    data_dir = Path(data_dir)

    # Find files to process
    files = sorted(data_dir.glob(file_pattern))
    files = [f for f in files if not f.name.endswith("_translated.json")]

    if not files:
        logger.warning(f"No files matching '{file_pattern}' found in {data_dir}")
        return []

    logger.info(f"Found {len(files)} file(s) to process")

    results = []
    for file_path in files:
        # Check if already translated
        output_path = file_path.parent / file_path.name.replace(".json", "_translated.json")

        if skip_existing and output_path.exists():
            logger.info(f"Skipping {file_path.name} (already translated)")
            continue

        try:
            result = translate_segments_file(file_path, translator, **kwargs)
            results.append({"status": "success", "file": file_path.name})
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            results.append({"status": "failed", "file": file_path.name, "error": str(e)})

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """
    Example usage:
    
    1. Single file:
        translator = NLLBTranslator(preferred_gpu=0)
        translate_segments_file('data/segments.json', translator)
    
    2. Directory:
        translator = NLLBTranslator()
        process_directory('downloads/video-001', translator)
    
    Configuration:
    - MODEL: 3.3B for best quality, distilled-600M for speed
    - NUM_BEAMS: 4 recommended (quality/speed balance)
    - BATCH_SIZE: 32 for 3.3B, higher for smaller models
    - PREFERRED_GPU: None for auto-select, or specific GPU index
    """

    # Configuration
    MODEL = "facebook/nllb-200-3.3B"
    NUM_BEAMS = 4  # 4-5 optimal for NLLB
    BATCH_SIZE = 32
    PREFERRED_GPU = None  # None for auto-select

    logger.info("=" * 70)
    logger.info("NLLB Translation")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL}")
    logger.info(f"Beam search: {NUM_BEAMS} beams")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info("=" * 70)

    # Initialize translator
    translator = NLLBTranslator(model_name=MODEL, preferred_gpu=PREFERRED_GPU)

    # Process files
    downloads_dir = Path("downloads")

    if not downloads_dir.exists():
        logger.error(f"Directory not found: {downloads_dir}")
    else:
        video_dirs = sorted(downloads_dir.glob("video-*"))

        if not video_dirs:
            logger.warning("No video-* directories found")
        else:
            logger.info(f"Found {len(video_dirs)} video directory(s)")

            for video_dir in video_dirs:
                logger.info(f"\nProcessing: {video_dir.name}")
                logger.info("-" * 70)

                results = process_directory(
                    video_dir, translator, batch_size=BATCH_SIZE, num_beams=NUM_BEAMS
                )

                success = sum(1 for r in results if r["status"] == "success")
                failed = sum(1 for r in results if r["status"] == "failed")

                logger.info(f"Results: {success} success, {failed} failed")

    # Cleanup
    translator.cleanup()

    logger.info("\n" + "=" * 70)
    logger.info("Translation complete!")
    logger.info("=" * 70)
