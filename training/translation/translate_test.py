# translate.py
import torch
import sentencepiece as spm
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from transformer_model import Transformer
from iteration_config import ITER_CONFIG


class Translator:
    def __init__(self, checkpoint_path, ru_tokenizer_path, en_tokenizer_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.ru_sp = spm.SentencePieceProcessor()
        self.ru_sp.load(ru_tokenizer_path)

        self.en_sp = spm.SentencePieceProcessor()
        self.en_sp.load(en_tokenizer_path)

        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        config = checkpoint.get("config", {})

        self.model = Transformer(
            src_vocab_size=16000,
            tgt_vocab_size=16000,
            d_model=config.get("d_model", 256),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 3),
            d_ff=config.get("d_ff", 1024),
            max_seq_length=512,
            dropout=config.get("dropout", 0.3),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print("Model loaded successfully!")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

        self.bos_id = 2
        self.eos_id = 3
        self.pad_id = 0

    def translate(
        self,
        russian_text,
        max_length=100,
        temperature=0.7,
        repetition_penalty=1.5,
        top_k=50,
        top_p=0.9,
    ):
        self.model.eval()

        with torch.no_grad():
            src_tokens = self.ru_sp.encode(russian_text, out_type=int)
            src = torch.tensor([src_tokens], dtype=torch.long).to(self.device)

            tgt_tokens = [self.bos_id]

            token_counts = {}
            consecutive_repeats = 0
            last_token = None

            for step in range(max_length):
                tgt = torch.tensor([tgt_tokens], dtype=torch.long).to(self.device)

                output = self.model(src, tgt)
                logits = output[0, -1, :].clone()

                for token_id, count in token_counts.items():
                    if count > 0:
                        logits[token_id] = logits[token_id] / (repetition_penalty**count)

                logits = logits / temperature

                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token == self.eos_id or next_token == self.pad_id:
                    break

                if next_token == last_token:
                    consecutive_repeats += 1
                    if consecutive_repeats >= 3:
                        break
                else:
                    consecutive_repeats = 0

                token_counts[next_token] = token_counts.get(next_token, 0) + 1

                if token_counts[next_token] > 5:
                    break

                tgt_tokens.append(next_token)
                last_token = next_token

            translation = self.en_sp.decode(tgt_tokens[1:])

        return translation

    def translate_batch(self, russian_sentences, max_length=100):
        """Translate multiple sentences"""
        translations = []

        for sentence in tqdm(russian_sentences, desc="Translating"):
            translation = self.translate(sentence, max_length)
            translations.append(translation)

        return translations


def main():
    data_dir = Path(ITER_CONFIG["data_dir"])
    checkpoint_dir = Path(ITER_CONFIG["checkpoint_dir"])

    checkpoint_path = checkpoint_dir / "best_model.pt"
    ru_tokenizer_path = data_dir / "ru_spm.model"
    en_tokenizer_path = data_dir / "en_spm.model"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python train.py")
        return

    if not ru_tokenizer_path.exists() or not en_tokenizer_path.exists():
        print(f"Error: Tokenizer models not found in {data_dir}")
        return

    translator = Translator(
        checkpoint_path=str(checkpoint_path),
        ru_tokenizer_path=str(ru_tokenizer_path),
        en_tokenizer_path=str(en_tokenizer_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    test_file = "russian_200_test.tsv"

    if not Path(test_file).exists():
        print(f"\nError: Test file '{test_file}' not found!")
        return

    df = pd.read_csv(test_file, sep="\t")
    russian_sentences = df["text"].tolist()

    print(f"Loaded {len(russian_sentences)} sentences from {test_file}")

    print("\nFirst 3 sentences:")
    for i, sent in enumerate(russian_sentences[:3], 1):
        print(f"{i}. {sent[:80]}{'...' if len(sent) > 80 else ''}")

    translations = translator.translate_batch(russian_sentences, max_length=100)

    results_df = pd.DataFrame({"russian": russian_sentences, "english_translation": translations})

    output_file = "translations_russian_200_test.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("SAMPLE TRANSLATIONS")
    print("=" * 60)

    for i in range(min(10, len(results_df))):
        print(f"\n{i + 1}.")
        print(
            f"RU: {results_df.iloc[i]['russian'][:100]}{'...' if len(results_df.iloc[i]['russian']) > 100 else ''}"
        )
        print(f"EN: {results_df.iloc[i]['english_translation']}")

    print(f"Total sentences translated: {len(translations)}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
