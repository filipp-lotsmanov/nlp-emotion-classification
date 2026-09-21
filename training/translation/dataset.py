import torch
from torch.utils.data import Dataset
import torch.nn as nn
import sentencepiece as spm


class TranslationDataset(Dataset):
    def __init__(self, en_file, ru_file, ru_tokenizer_path, en_tokenizer_path, max_len=512):
        with open(en_file, "r", encoding="utf-8") as f:
            self.en_lines = [line.strip() for line in f]
        with open(ru_file, "r", encoding="utf-8") as f:
            self.ru_lines = [line.strip() for line in f]

        self.ru_sp = spm.SentencePieceProcessor()
        self.ru_sp.load(ru_tokenizer_path)

        self.en_sp = spm.SentencePieceProcessor()
        self.en_sp.load(en_tokenizer_path)

        self.max_len = max_len

        self.src_vocab_size = self.ru_sp.get_piece_size()
        self.tgt_vocab_size = self.en_sp.get_piece_size()

    def __len__(self):
        return len(self.en_lines)

    def __getitem__(self, idx):
        en_text = self.en_lines[idx]
        ru_text = self.ru_lines[idx]

        ru_tokens = self.ru_sp.encode(ru_text, out_type=int)
        en_tokens = self.en_sp.encode(en_text, out_type=int)

        ru_tokens = ru_tokens[: self.max_len]
        en_tokens = en_tokens[: self.max_len]

        return {
            "src": torch.tensor(ru_tokens, dtype=torch.long),
            "tgt": torch.tensor(en_tokens, dtype=torch.long),
        }


def collate_fn(batch):
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_padded, tgt_padded
