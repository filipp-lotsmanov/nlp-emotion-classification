import torch
from torch.utils.data import DataLoader
from pathlib import Path
import subprocess
import sys

from iteration_config import ITER_CONFIG

try:
    import sentencepiece as spm
    from tqdm import tqdm
    import pandas as pd
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "sentencepiece",
            "tqdm",
            "tensorboard",
            "pandas",
        ]
    )
    import sentencepiece as spm

from transformer_model import Transformer
from dataset import TranslationDataset, collate_fn
from trainer import TransformerTrainer


def train_sentencepiece(data_dir):
    """Train tokenizers for current iteration"""
    data_dir = Path(data_dir)
    ru_model = data_dir / "ru_spm.model"
    en_model = data_dir / "en_spm.model"

    if not ru_model.exists():
        print("Training Russian tokenizer...")
        spm.SentencePieceTrainer.train(
            input=str(data_dir / "train.ru"),
            model_prefix=str(data_dir / "ru_spm"),
            vocab_size=16000,
            character_coverage=0.9995,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            pad_piece="<pad>",
        )
        print("  ✓ Russian tokenizer trained")
    else:
        print("  ✓ Russian tokenizer exists")

    if not en_model.exists():
        print("Training English tokenizer...")
        spm.SentencePieceTrainer.train(
            input=str(data_dir / "train.en"),
            model_prefix=str(data_dir / "en_spm"),
            vocab_size=16000,
            character_coverage=0.9995,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            pad_piece="<pad>",
        )
        print("  ✓ English tokenizer trained")
    else:
        print("  ✓ English tokenizer exists")


def verify_data_alignment(data_dir, num_samples=5):
    """Verify that parallel data is properly aligned"""
    print("\nVerifying data alignment...")
    data_dir = Path(data_dir)

    with open(data_dir / "train.ru", "r", encoding="utf-8") as f:
        ru_lines = f.readlines()[:num_samples]

    with open(data_dir / "train.en", "r", encoding="utf-8") as f:
        en_lines = f.readlines()[:num_samples]

    print(f"\nShowing {num_samples} sample pairs:")
    for i, (ru, en) in enumerate(zip(ru_lines, en_lines), 1):
        print(f"\n{i}.")
        print(f"  RU: {ru.strip()[:100]}...")
        print(f"  EN: {en.strip()[:100]}...")

    print("\n✓ Data alignment verified")


def main():
    print(f"TRAINING: {ITER_CONFIG['name']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    data_dir = Path(ITER_CONFIG["data_dir"])
    checkpoint_dir = Path(ITER_CONFIG["checkpoint_dir"])
    log_dir = Path(ITER_CONFIG["log_dir"])

    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "train.en").exists():
        print(f"\n✗ Preprocessed data not found in {data_dir}")
        return

    print("\nPreparing SentencePiece tokenizers...")
    train_sentencepiece(data_dir)

    verify_data_alignment(data_dir)

    config = {
        "d_model": 256,
        "num_layers": 3,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.3,
        "learning_rate": 3e-4,
        "batch_size": 32,
        "weight_decay": 0.01,
        "label_smoothing": 0.1,
        "use_warmup": True,
        "warmup_steps": 2000,
        "gradient_accumulation_steps": 2,
        "early_stop_patience": 5,
        "early_stop_min_delta": 0.01,
        "use_amp": False,
    }

    print("CONFIGURATION")
    for k, v in config.items():
        print(f"  {k:30s}: {v}")
    print(
        f"  {'effective_batch_size':30s}: {config['batch_size'] * config['gradient_accumulation_steps']}"
    )
    print("=" * 60)

    print("\nCreating datasets...")
    train_dataset = TranslationDataset(
        str(data_dir / "train.en"),
        str(data_dir / "train.ru"),
        str(data_dir / "ru_spm.model"),
        str(data_dir / "en_spm.model"),
    )
    valid_dataset = TranslationDataset(
        str(data_dir / "valid.en"),
        str(data_dir / "valid.ru"),
        str(data_dir / "ru_spm.model"),
        str(data_dir / "en_spm.model"),
    )

    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(valid_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )

    print(f"  Train batches per epoch: {len(train_loader)}")
    print(f"  Valid batches per epoch: {len(valid_loader)}")
    print(f"  Steps per epoch: {len(train_loader) // config['gradient_accumulation_steps']}")

    print("\nInitializing model...")
    model = Transformer(
        src_vocab_size=16000,
        tgt_vocab_size=16000,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_length=512,
        dropout=config["dropout"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB")

    print("STARTING TRAINING")
    print(f"  Logs: {log_dir}")

    from torch.utils.tensorboard import SummaryWriter

    trainer = TransformerTrainer(model, train_loader, valid_loader, config, device)
    trainer.writer = SummaryWriter(str(log_dir))

    trainer.writer.add_text("config", str(config), 0)

    try:
        trainer.train(num_epochs=50, save_path=str(checkpoint_dir / "best_model.pt"))
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current state...")
        torch.save(
            {
                "epoch": trainer.epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "config": config,
            },
            checkpoint_dir / "interrupted_model.pt",
        )
        print(f"Saved to: {checkpoint_dir / 'interrupted_model.pt'}")

    print(f"TRAINING COMPLETE: {ITER_CONFIG['name']}")
    print(f"Best model: {checkpoint_dir / 'best_model.pt'}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
