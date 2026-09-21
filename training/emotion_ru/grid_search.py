import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
import os
import json
from datetime import datetime
from datasets import load_dataset

# Avoid TensorFlow issues
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# Grid search configuration
# Approach: Test combinations of models and hyperparameters systematically
# Alternative 1: Use Optuna for Bayesian optimization (more efficient but complex)
# Alternative 2: Random search (faster but less thorough)
# Why this approach: Exhaustive search gives complete picture, results are reproducible
GRID_SEARCH_CONFIG = {
    "models": [
        "cointegrated/rubert-tiny2",
        "ai-forever/ruBert-base",
        "DeepPavlov/rubert-base-cased-conversational",
    ],
    "learning_rates": [1e-4, 2e-5, 5e-5],  # Including baseline 1e-4
    "batch_sizes": [16, 32, 64],  # Including baseline 64
    "epochs": [10],  # Keep fixed to save time, can expand later
    "max_lengths": [256],  # Can test [128, 256, 512] if needed
}

# Fixed configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
COLUMNS_TO_DROP = ["guilt", "shame"]
RESULTS_FILE = "grid_search_results.json"

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load dataset once
print("Loading dataset from HuggingFace...")
ds = load_dataset("Djacon/ru-izard-emotions")

if "train" in ds:
    df = ds["train"].to_pandas()
else:
    df = pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)

print(f"Dataset loaded: {df.shape}")
df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
print(f"Dataset shape after dropping: {df.shape}")

# Extract data
texts = df.iloc[:, 0].tolist()
label_columns = df.columns[1:].tolist()
labels = df.iloc[:, 1:].values.astype(np.float32)

print(f"Label columns: {label_columns}")
print(f"Number of labels: {len(label_columns)}")

# Split data once
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def calculate_metrics(predictions, labels):
    """
    Calculate multilabel classification metrics including AUC
    Why: Need to compare with baseline metrics (AUC, F1 micro, F1 macro)
    """
    sigmoid = nn.Sigmoid()
    probs = sigmoid(predictions).cpu().numpy()
    y_pred = (probs > 0.5).astype(int)
    y_true = labels.cpu().numpy().astype(int)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # Calculate AUC for multilabel
    # Risk: AUC calculation can fail if a label has no positive samples in validation
    try:
        auc_macro = roc_auc_score(y_true, probs, average="macro")
    except ValueError:
        auc_macro = 0.0

    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "auc_macro": auc_macro,
        "y_pred": y_pred,
        "y_true": y_true,
    }


def train_epoch(model, data_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Less frequent logging to speed up training
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(data_loader)


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            total_loss += loss.item()
            all_predictions.append(outputs.logits)
            all_labels.append(labels)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = calculate_metrics(all_predictions, all_labels)
    metrics["loss"] = total_loss / len(data_loader)

    return metrics


def train_single_configuration(model_name, learning_rate, batch_size, epochs, max_length):
    """
    Train a single model configuration and return metrics
    Why separate function: Easier to handle errors per configuration without stopping entire grid search
    """
    print(f"\n{'=' * 80}")
    print(f"Training: {model_name}")
    print(f"LR: {learning_rate}, Batch: {batch_size}, Epochs: {epochs}, MaxLen: {max_length}")
    print(f"{'=' * 80}")

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(label_columns), problem_type="multi_label_classification"
        ).to(device)

        # Create datasets
        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length)
        val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_f1_macro = 0
        best_metrics = None

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            print(f"  Training Loss: {train_loss:.4f}")

            val_metrics = evaluate_model(model, val_loader, criterion, device)
            print(f"  Validation Loss: {val_metrics['loss']:.4f}")
            print(f"  Validation F1 Macro: {val_metrics['f1_macro']:.4f}")
            print(f"  Validation F1 Micro: {val_metrics['f1_micro']:.4f}")
            print(f"  Validation AUC Macro: {val_metrics['auc_macro']:.4f}")

            if val_metrics["f1_macro"] > best_f1_macro:
                best_f1_macro = val_metrics["f1_macro"]
                best_metrics = {
                    "f1_macro": val_metrics["f1_macro"],
                    "f1_micro": val_metrics["f1_micro"],
                    "auc_macro": val_metrics["auc_macro"],
                    "val_loss": val_metrics["loss"],
                    "train_loss": train_loss,
                    "epoch": epoch + 1,
                }

        # Per-label metrics for best model
        precision, recall, f1, _ = precision_recall_fscore_support(
            best_metrics.get("y_true", val_metrics["y_true"]),
            best_metrics.get("y_pred", val_metrics["y_pred"]),
            average=None,
            zero_division=0,
        )

        per_label_f1 = {label: float(f1[i]) for i, label in enumerate(label_columns)}
        best_metrics["per_label_f1"] = per_label_f1

        # Clean up GPU memory
        del model, tokenizer, train_dataset, val_dataset, train_loader, val_loader
        torch.cuda.empty_cache()

        return best_metrics

    except Exception as e:
        print(f"ERROR training {model_name}: {str(e)}")
        # Clean up on error
        torch.cuda.empty_cache()
        return None


# Main grid search loop
print(f"\n{'#' * 80}")
print("STARTING GRID SEARCH")
print(f"{'#' * 80}")

# Calculate total combinations
total_combinations = (
    len(GRID_SEARCH_CONFIG["models"])
    * len(GRID_SEARCH_CONFIG["learning_rates"])
    * len(GRID_SEARCH_CONFIG["batch_sizes"])
    * len(GRID_SEARCH_CONFIG["epochs"])
    * len(GRID_SEARCH_CONFIG["max_lengths"])
)
print(f"Total configurations to test: {total_combinations}")

# Load existing results if available
# Why: Resume from previous runs without retraining
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} existing results")
else:
    all_results = []

current_combination = 0

for model_name in GRID_SEARCH_CONFIG["models"]:
    for lr in GRID_SEARCH_CONFIG["learning_rates"]:
        for batch_size in GRID_SEARCH_CONFIG["batch_sizes"]:
            for epochs in GRID_SEARCH_CONFIG["epochs"]:
                for max_length in GRID_SEARCH_CONFIG["max_lengths"]:
                    current_combination += 1

                    # Create configuration identifier
                    config_id = f"{model_name}__lr{lr}__bs{batch_size}__ep{epochs}__ml{max_length}"

                    # Skip if already trained
                    if any(r["config_id"] == config_id for r in all_results):
                        print(
                            f"\nSkipping {current_combination}/{total_combinations}: {config_id} (already trained)"
                        )
                        continue

                    print(f"\n\nConfiguration {current_combination}/{total_combinations}")

                    metrics = train_single_configuration(
                        model_name, lr, batch_size, epochs, max_length
                    )

                    if metrics is not None:
                        result = {
                            "config_id": config_id,
                            "model_name": model_name,
                            "learning_rate": lr,
                            "batch_size": batch_size,
                            "epochs": epochs,
                            "max_length": max_length,
                            "timestamp": datetime.now().isoformat(),
                            **metrics,
                        }

                        all_results.append(result)

                        # Save after each successful run
                        # Why: Don't lose results if script crashes
                        with open(RESULTS_FILE, "w") as f:
                            json.dump(all_results, f, indent=2)

                        print(
                            f"\n  Results saved. Best F1 Macro so far: {max(r['f1_macro'] for r in all_results):.4f}"
                        )

# Final summary
print(f"\n\n{'#' * 80}")
print("GRID SEARCH COMPLETED")
print(f"{'#' * 80}")

# Sort by F1 macro
all_results.sort(key=lambda x: x["f1_macro"], reverse=True)

print("\nTop 5 configurations:")
for i, result in enumerate(all_results[:5], 1):
    print(f"\n{i}. {result['model_name']}")
    print(f"   LR: {result['learning_rate']}, Batch: {result['batch_size']}")
    print(
        f"   F1 Macro: {result['f1_macro']:.4f}, F1 Micro: {result['f1_micro']:.4f}, AUC: {result['auc_macro']:.4f}"
    )

# Save detailed results to CSV for analysis
results_df = pd.DataFrame(
    [{k: v for k, v in r.items() if k != "per_label_f1"} for r in all_results]
)
results_df.to_csv("grid_search_results.csv", index=False)
print("\nDetailed results saved to grid_search_results.csv")

# Print best configuration details
best_result = all_results[0]
print("\n\nBEST CONFIGURATION:")
print(f"Model: {best_result['model_name']}")
print(f"Learning Rate: {best_result['learning_rate']}")
print(f"Batch Size: {best_result['batch_size']}")
print(f"Epochs: {best_result['epochs']}")
print(f"F1 Macro: {best_result['f1_macro']:.4f}")
print(f"F1 Micro: {best_result['f1_micro']:.4f}")
print(f"AUC Macro: {best_result['auc_macro']:.4f}")

print("\nPer-label F1 scores:")
for label, score in best_result["per_label_f1"].items():
    print(f"  {label}: {score:.4f}")

print("\nGrid search completed successfully!")
