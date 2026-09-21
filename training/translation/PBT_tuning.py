import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sentencepiece as spm

from transformer_model import Transformer
from dataset import TranslationDataset, collate_fn
from trainer import TransformerTrainer
from iteration_config import ITER_CONFIG


CONFIGS_TO_TEST = [
    # Baseline (lower LR)
    {
        "name": "baseline",
        "d_model": 512,
        "num_layers": 6,
        "lr": 5e-5,
        "dropout": 0.1,
        "batch_size": 32,
    },
    # Smaller, faster
    {
        "name": "small",
        "d_model": 256,
        "num_layers": 4,
        "lr": 1e-4,
        "dropout": 0.1,
        "batch_size": 64,
    },
    # More regularization
    {
        "name": "regularized",
        "d_model": 512,
        "num_layers": 6,
        "lr": 3e-5,
        "dropout": 0.2,
        "batch_size": 32,
    },
    # Deeper model
    {
        "name": "deep",
        "d_model": 512,
        "num_layers": 8,
        "lr": 3e-5,
        "dropout": 0.15,
        "batch_size": 16,
    },
    # Wider model
    {"name": "wide", "d_model": 768, "num_layers": 4, "lr": 5e-5, "dropout": 0.1, "batch_size": 16},
    # Conservative
    {
        "name": "conservative",
        "d_model": 512,
        "num_layers": 6,
        "lr": 2e-5,
        "dropout": 0.15,
        "batch_size": 32,
    },
]


def setup_sentencepiece_models(data_dir, vocab_size=16000):
    data_path = Path(data_dir)
    en_model = data_path / "en_spm.model"
    ru_model = data_path / "ru_spm.model"

    if en_model.exists() and ru_model.exists():
        print("✓ SentencePiece models found:")
        print(f"  - {en_model}")
        print(f"  - {ru_model}")
        return True

    train_en = data_path / "train.en"
    train_ru = data_path / "train.ru"

    if not train_en.exists() or not train_ru.exists():
        print("\n✗ ERROR: Training data files not found:")
        print(f"  Looking for: {train_en}")
        print(f"  Looking for: {train_ru}")
        print("\nPlease ensure your data files are in the correct location.")
        return False

    print("\nTraining data found:")
    print(f"  - {train_en}")
    print(f"  - {train_ru}")

    print("\nTraining English SentencePiece model...")
    try:
        spm.SentencePieceTrainer.train(
            input=str(train_en),
            model_prefix=str(data_path / "en_spm"),
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=[],
        )
        print(f"  ✓ Created: {en_model}")
    except Exception as e:
        print(f"  ✗ Failed to train English model: {e}")
        return False

    print("\nTraining Russian SentencePiece model...")
    try:
        spm.SentencePieceTrainer.train(
            input=str(train_ru),
            model_prefix=str(data_path / "ru_spm"),
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=[],
        )
        print(f"  ✓ Created: {ru_model}")
    except Exception as e:
        print(f"  ✗ Failed to train Russian model: {e}")
        return False

    print("\n✓ All SentencePiece models ready!")
    return True


def train_single_config(config_dict, gpu_id=0):
    device = f"cuda:{gpu_id}"

    print(f"\n{'=' * 60}")
    print(f"Testing: {config_dict['name']}")
    print(f"{'=' * 60}")
    print(f"  d_model: {config_dict['d_model']}")
    print(f"  num_layers: {config_dict['num_layers']}")
    print(f"  learning_rate: {config_dict['lr']}")
    print(f"  dropout: {config_dict['dropout']}")
    print(f"  batch_size: {config_dict['batch_size']}")

    data_dir = Path(ITER_CONFIG["data_dir"])

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_dict["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config_dict["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = Transformer(
        src_vocab_size=16000,
        tgt_vocab_size=16000,
        d_model=config_dict["d_model"],
        num_heads=8,
        num_layers=config_dict["num_layers"],
        d_ff=config_dict["d_model"] * 4,
        max_seq_length=512,
        dropout=config_dict["dropout"],
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    train_config = {
        "d_model": config_dict["d_model"],
        "num_heads": 8,
        "num_layers": config_dict["num_layers"],
        "d_ff": config_dict["d_model"] * 4,
        "dropout": config_dict["dropout"],
        "learning_rate": config_dict["lr"],
        "batch_size": config_dict["batch_size"],
        "weight_decay": 0.0001,
        "label_smoothing": 0.1,
        "use_warmup": True,
        "warmup_steps": 4000,
        "gradient_accumulation_steps": 1,
        "early_stop_patience": 3,
        "early_stop_min_delta": 0.001,
        "use_amp": False,
    }

    trainer = TransformerTrainer(model, train_loader, valid_loader, train_config, device)
    trainer.writer = None

    best_val_loss = float("inf")

    for epoch in range(15):
        print(f"\nEpoch {epoch + 1}/15")
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()

        print(f"  Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if trainer.early_stopping(val_loss, epoch):
            print(f"  Early stopped at epoch {epoch + 1}")
            break

    result = {
        "name": config_dict["name"],
        "config": config_dict,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_loss,
        "epochs_trained": epoch + 1,
        "parameters": params,
    }

    print(f"\n  Result: {best_val_loss:.4f}")

    return result


def run_simple_search(gpu_id=6):

    print("=" * 60)
    print("SIMPLE HYPERPARAMETER SEARCH")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  GPU: {gpu_id}")
    print(f"  Configs to test: {len(CONFIGS_TO_TEST)}")
    print("  Max epochs per config: 15")
    print("  Early stopping: Yes (patience=3)")
    print("  Estimated time: 3-4 hours")

    if not torch.cuda.is_available():
        print("\n✗ CUDA not available!")
        return None

    print(f"\n  Using: {torch.cuda.get_device_name(gpu_id)}")

    if not setup_sentencepiece_models(ITER_CONFIG["data_dir"]):
        print("\n✗ Failed to setup SentencePiece models. Exiting.")
        return None

    results = []

    for i, config in enumerate(CONFIGS_TO_TEST, 1):
        print(f"\n{'#' * 60}")
        print(f"CONFIG {i}/{len(CONFIGS_TO_TEST)}")
        print(f"{'#' * 60}")

        try:
            result = train_single_config(config, gpu_id=gpu_id)
            results.append(result)

            with open("search_progress.json", "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"\n✗ Config '{config['name']}' failed: {e}")
            results.append(
                {"name": config["name"], "config": config, "best_val_loss": None, "error": str(e)}
            )
            continue

    valid_results = [r for r in results if r.get("best_val_loss") is not None]

    if not valid_results:
        print("\n✗ No configurations completed successfully!")
        return results

    best = min(valid_results, key=lambda x: x["best_val_loss"])

    print("\n" + "=" * 60)
    print("SEARCH COMPLETE")
    print("=" * 60)

    print("\nAll results:")
    for r in sorted(valid_results, key=lambda x: x["best_val_loss"]):
        print(
            f"  {r['name']:15s}: Val Loss = {r['best_val_loss']:.4f}, Params = {r['parameters']:,}"
        )

    print(f"\n{'=' * 60}")
    print(f"BEST: {best['name']}")
    print(f"{'=' * 60}")
    print(f"  Validation Loss: {best['best_val_loss']:.4f}")
    print(f"  Parameters: {best['parameters']:,}")
    print("\n  Configuration:")
    for key, value in best["config"].items():
        print(f"    {key}: {value}")

    with open("best_hyperparameters.json", "w") as f:
        json.dump({"best_config": best, "all_results": results}, f, indent=2)

    print("\n✓ Results saved: best_hyperparameters.json")

    try:
        import matplotlib.pyplot as plt

        names = [r["name"] for r in valid_results]
        losses = [r["best_val_loss"] for r in valid_results]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            names, losses, color=["green" if r == best else "blue" for r in valid_results]
        )
        plt.xlabel("Configuration")
        plt.ylabel("Best Validation Loss")
        plt.title("Hyperparameter Search Results")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig("search_comparison.png", dpi=150)
        print("✓ Plot saved: search_comparison.png")
    except:
        pass

    return results


if __name__ == "__main__":
    results = run_simple_search(gpu_id=6)
    if results:
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print("1. Open best_hyperparameters.json")
        print("2. Copy best config to train.py")
        print("3. Run full training: python train.py")
