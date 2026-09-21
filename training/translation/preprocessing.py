import re
import json
import logging
from pathlib import Path
from collections import Counter
import unicodedata
import pandas as pd
from iteration_config import ITER_CONFIG


class OptimizedCorpusProcessor:
    def __init__(self, output_dir="data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        self.cleaning_stats = {
            "script_fixes": 0,
            "encoding_fixes": 0,
            "punctuation_fixes": 0,
            "duplicates_removed": 0,
        }

    def process_tsv_files(self, en_tsv, ru_tsv, min_chars=66, max_chars=313, quality_threshold=0.5):
        self.logger.info("Loading TSV files...")

        try:
            en_df = pd.read_csv(
                en_tsv, sep="\t", header=None, names=["text"], encoding="utf-8", on_bad_lines="skip"
            )
            ru_df = pd.read_csv(
                ru_tsv, sep="\t", header=None, names=["text"], encoding="utf-8", on_bad_lines="skip"
            )
        except Exception as e:
            self.logger.error(f"Error reading TSV files: {e}")
            return None

        en_texts = en_df["text"].dropna().astype(str).tolist()
        ru_texts = ru_df["text"].dropna().astype(str).tolist()

        min_len = min(len(en_texts), len(ru_texts))
        pairs = [(en_texts[i], ru_texts[i]) for i in range(min_len)]

        self.logger.info(f"Loaded {len(pairs):,} pairs from TSV")

        return self._process_pairs(pairs, min_chars, max_chars, quality_threshold)

    def _process_pairs(self, pairs, min_chars, max_chars, quality_threshold):
        seen_pairs = set()
        processed_pairs = []
        stats = {"total": len(pairs), "kept": 0, "filtered": 0}

        for i, (en, ru) in enumerate(pairs):
            if i % 10000 == 0:
                self.logger.info(f"Processing {i:,}/{len(pairs):,}...")

            en_clean = self._deep_clean(en, "en")
            ru_clean = self._deep_clean(ru, "ru")

            if not en_clean or not ru_clean:
                stats["filtered"] += 1
                continue

            pair_key = (en_clean.lower(), ru_clean.lower())
            if pair_key in seen_pairs:
                self.cleaning_stats["duplicates_removed"] += 1
                continue
            seen_pairs.add(pair_key)

            if not self._passes_filter(en_clean, ru_clean, min_chars, max_chars):
                stats["filtered"] += 1
                continue

            score = self._quality_score(en_clean, ru_clean)

            if score >= quality_threshold:
                processed_pairs.append((en_clean, ru_clean, score))
                stats["kept"] += 1
            else:
                stats["filtered"] += 1

        self.logger.info(f"Kept {stats['kept']:,} / {len(pairs):,}")

        splits = self._create_splits(processed_pairs)
        self._save_splits(splits)

        metadata = {
            "total_pairs": len(pairs),
            "kept_pairs": stats["kept"],
            "filtered_pairs": stats["filtered"],
            "duplicates_removed": self.cleaning_stats["duplicates_removed"],
            "train_size": len(splits["train"]),
            "valid_size": len(splits["valid"]),
            "test_size": len(splits["test"]),
            "cleaning_stats": self.cleaning_stats,
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _deep_clean(self, text, lang):
        if not text or not str(text).strip():
            return ""

        text = str(text)

        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
        text = text.replace("\u00a0", " ")
        self.cleaning_stats["encoding_fixes"] += 1

        # Script contamination
        if lang == "en":
            cyrillic_map = {
                "а": "a",
                "А": "A",
                "е": "e",
                "Е": "E",
                "о": "o",
                "О": "O",
                "р": "p",
                "Р": "P",
                "с": "c",
                "С": "C",
                "у": "y",
                "У": "Y",
                "х": "x",
                "Х": "X",
                "і": "i",
                "І": "I",
                "м": "m",
                "М": "M",
                "в": "v",
                "В": "V",
                "к": "k",
                "К": "K",
                "н": "h",
                "Н": "H",
                "т": "t",
                "Т": "T",
            }
            for cyr, lat in cyrillic_map.items():
                if cyr in text:
                    text = text.replace(cyr, lat)
                    self.cleaning_stats["script_fixes"] += 1

        text = re.sub(r'[""„‟]', '"', text)
        text = re.sub(r"[" "‚]", "'", text)
        text = re.sub(r"[—–]", "-", text)
        text = re.sub(r"\s+", " ", text)
        self.cleaning_stats["punctuation_fixes"] += 1

        math_map = {
            "α": "alpha",
            "β": "beta",
            "γ": "gamma",
            "δ": "delta",
            "θ": "theta",
            "λ": "lambda",
            "μ": "mu",
            "π": "pi",
            "≤": "<=",
            "≥": ">=",
            "≠": "!=",
            "±": "+/-",
            "×": "x",
            "÷": "/",
            "²": "2",
            "³": "3",
            "°": " deg ",
        }
        for symbol, replacement in math_map.items():
            text = text.replace(symbol, replacement)

        return text.strip()

    def _passes_filter(self, en, ru, min_c, max_c):
        en_len, ru_len = len(en), len(ru)

        if en_len < min_c or ru_len < min_c or en_len > max_c or ru_len > max_c:
            return False

        en_words = len(en.split())
        ru_words = len(ru.split())

        if en_words < 3 or ru_words < 3:
            return False

        ratio = ru_len / en_len if en_len > 0 else 0
        return 0.3 <= ratio <= 3.0

    def _quality_score(self, en, ru):
        score = 1.0

        en_words = len(en.split())
        ru_words = len(ru.split())

        if en_words == 0 or ru_words == 0:
            return 0.0

        ratio = ru_words / en_words
        if ratio < 0.5 or ratio > 2.0:
            score -= 0.2

        en_nums = set(re.findall(r"\d+", en))
        ru_nums = set(re.findall(r"\d+", ru))
        if en_nums and en_nums != ru_nums:
            score -= 0.15

        if re.search(r"[а-яё]", en):
            score -= 0.3

        if en_words > 0:
            word_counts = Counter(en.lower().split())
            most_common = word_counts.most_common(1)[0][1] if word_counts else 0
            if most_common / en_words > 0.5:
                score -= 0.3

        return max(0.0, score)

    def _create_splits(self, pairs, train_r=0.8, valid_r=0.15):
        import random

        random.shuffle(pairs)

        n = len(pairs)
        train_n = int(n * train_r)
        valid_n = int(n * valid_r)

        return {
            "train": pairs[:train_n],
            "valid": pairs[train_n : train_n + valid_n],
            "test": pairs[train_n + valid_n :],
        }

    def _save_splits(self, splits):
        for name, pairs in splits.items():
            with (
                open(self.output_dir / f"{name}.en", "w", encoding="utf-8") as f_en,
                open(self.output_dir / f"{name}.ru", "w", encoding="utf-8") as f_ru,
            ):
                for en, ru, _ in pairs:
                    f_en.write(en + "\n")
                    f_ru.write(ru + "\n")


if __name__ == "__main__":
    processor = OptimizedCorpusProcessor(output_dir=ITER_CONFIG["data_dir"])

    print(f"Processing: {ITER_CONFIG['name']}")
    print("Input files:")
    print(f"  EN: {ITER_CONFIG['en_file']}")
    print(f"  RU: {ITER_CONFIG['ru_file']}")
    print(f"Output directory: {ITER_CONFIG['data_dir']}")

    metadata = processor.process_tsv_files(
        en_tsv=ITER_CONFIG["en_file"], ru_tsv=ITER_CONFIG["ru_file"], quality_threshold=0.6
    )

    if metadata:
        print(f"  Kept: {metadata['kept_pairs']:,} pairs")
        print(f"  Duplicates removed: {metadata['duplicates_removed']:,}")
        print(f"  Output: {ITER_CONFIG['data_dir']}")
