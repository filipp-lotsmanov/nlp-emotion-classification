import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
import os

# Avoid TensorFlow issues
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Only import PyTorch transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# Configuration
MODEL_NAME = "DeepPavlov/rubert-base-cased"
PICKLE_PATH = "balanced_emotions_1996.pkl"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("Loading balanced dataset...")
df = pd.read_pickle(PICKLE_PATH)
print(f"Dataset loaded: {df.shape}")

# Extract data
texts = df.iloc[:, 0].tolist()
label_columns = df.columns[1:].tolist()
labels = df.iloc[:, 1:].values.astype(np.float32)

print(f"Label columns: {label_columns}")
print(f"Number of labels: {len(label_columns)}")

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_columns), problem_type="multi_label_classification"
).to(device)


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


# Create datasets and dataloaders
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Setup optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Loss function
criterion = nn.BCEWithLogitsLoss()


def calculate_metrics(predictions, labels):
    """Calculate multilabel classification metrics"""
    sigmoid = nn.Sigmoid()
    probs = sigmoid(predictions).cpu().numpy()
    y_pred = (probs > 0.5).astype(int)
    y_true = labels.cpu().numpy().astype(int)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    return f1_macro, f1_micro, y_pred, y_true


def train_epoch(model, data_loader, optimizer, scheduler, device):
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

        if (batch_idx + 1) % 50 == 0:
            print(f"Batch {batch_idx + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(data_loader)


def evaluate_model(model, data_loader, device):
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

    f1_macro, f1_micro, y_pred, y_true = calculate_metrics(all_predictions, all_labels)

    return total_loss / len(data_loader), f1_macro, f1_micro, y_pred, y_true


# Training loop
print("Starting training...")
best_f1 = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 50)

    # Training
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Training Loss: {train_loss:.4f}")

    # Validation
    val_loss, val_f1_macro, val_f1_micro, y_pred, y_true = evaluate_model(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation F1 Macro: {val_f1_macro:.4f}")
    print(f"Validation F1 Micro: {val_f1_micro:.4f}")

    # Per-label metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    print("\nPer-label F1 scores:")
    for i, label in enumerate(label_columns):
        print(f"  {label}: {f1[i]:.4f}")

    # Save best model
    if val_f1_macro > best_f1:
        best_f1 = val_f1_macro
        model_save_path = "./russian_emotion_classifier_best"
        os.makedirs(model_save_path, exist_ok=True)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"New best model saved! F1 Macro: {best_f1:.4f}")

print("\nTraining completed!")
print(f"Best F1 Macro: {best_f1:.4f}")
print("Model saved to: ./russian_emotion_classifier_best")


# Test prediction function
def predict_emotions(text, model, tokenizer, label_columns, device, max_length=256):
    model.eval()
    encoding = tokenizer(
        text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.sigmoid(outputs.logits)

    results = {}
    for i, label in enumerate(label_columns):
        results[label] = float(predictions[0][i].cpu())

    return results


# Example prediction
sample_text = texts[0]
predictions = predict_emotions(sample_text, model, tokenizer, label_columns, device)

print(f"\nSample prediction for: '{sample_text[:100]}...'")
for emotion, score in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
    print(f"  {emotion}: {score:.4f}")

print("\nScript completed successfully!")
