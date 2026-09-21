# compare_results.py
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_iteration_results(iteration_name):
    """Load results from one iteration"""
    checkpoint_path = Path(f"checkpoints/{iteration_name}/best_model.pt")

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found for {iteration_name}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    return {
        "name": iteration_name,
        "val_loss": checkpoint["val_loss"],
        "epoch": checkpoint["epoch"],
        "config": checkpoint.get("config", {}),
        "history": checkpoint.get("history", {}),
    }


def compare_iterations():

    iter1 = load_iteration_results("iteration_1")
    iter2 = load_iteration_results("iteration_2")

    if not iter1 or not iter2:
        print("\n✗ Both iterations must be trained first")
        print("Train them by:")
        print("  1. Edit iteration_config.py: CURRENT_ITERATION = 'iteration_1'")
        print("  2. Run: python preprocessing.py")
        print("  3. Run: python train.py")
        print("  4. Edit iteration_config.py: CURRENT_ITERATION = 'iteration_2'")
        print("  5. Run: python preprocessing.py")
        print("  6. Run: python train.py")
        print("  7. Run: python compare_results.py")
        return

    print(f"\n{iter1['name'].upper()}:")
    print(f"  Best Val Loss: {iter1['val_loss']:.4f}")
    print(f"  Epochs Trained: {iter1['epoch'] + 1}")
    if iter1["history"].get("train_loss"):
        print(f"  Final Train Loss: {iter1['history']['train_loss'][-1]:.4f}")

    print(f"\n{iter2['name'].upper()}:")
    print(f"  Best Val Loss: {iter2['val_loss']:.4f}")
    print(f"  Epochs Trained: {iter2['epoch'] + 1}")
    if iter2["history"].get("train_loss"):
        print(f"  Final Train Loss: {iter2['history']['train_loss'][-1]:.4f}")

    # Determine winner
    diff = iter1["val_loss"] - iter2["val_loss"]
    if diff < 0:
        winner = iter1["name"]
        improvement = abs(diff)
        improvement_pct = (improvement / iter2["val_loss"]) * 100
    else:
        winner = iter2["name"]
        improvement = diff
        improvement_pct = (improvement / iter1["val_loss"]) * 100

    print(f"\n{'=' * 60}")
    print(f"WINNER: {winner}")
    print(f"Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
    print(f"{'=' * 60}")

    # Create comparison plots
    if iter1["history"].get("val_loss") and iter2["history"].get("val_loss"):
        print("\nGenerating comparison plots...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Validation loss
        axes[0].plot(iter1["history"]["val_loss"], label=iter1["name"], linewidth=2)
        axes[0].plot(iter2["history"]["val_loss"], label=iter2["name"], linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Validation Loss")
        axes[0].set_title("Validation Loss Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Training loss
        axes[1].plot(iter1["history"]["train_loss"], label=iter1["name"], linewidth=2)
        axes[1].plot(iter2["history"]["train_loss"], label=iter2["name"], linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Training Loss")
        axes[1].set_title("Training Loss Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("iteration_comparison.png", dpi=150)
        print("  ✓ Plot saved: iteration_comparison.png")

    # Save comparison JSON
    comparison = {
        "iteration_1": {"val_loss": iter1["val_loss"], "epochs": iter1["epoch"] + 1},
        "iteration_2": {"val_loss": iter2["val_loss"], "epochs": iter2["epoch"] + 1},
        "winner": winner,
        "improvement": improvement,
        "improvement_percentage": improvement_pct,
    }

    with open("comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("  ✓ Results saved: comparison_results.json")


if __name__ == "__main__":
    compare_iterations()
