"""
Training on MELD dataset - has EXACTLY 7 emotions natively!
Anger, Disgust, Sadness, Joy, Neutral, Surprise, Fear

13,000 utterances from Friends TV show
Simple code, should get F1 > 0.75

Code: train_MELD.py
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from textblob import TextBlob

    has_sent = True
except:
    has_sent = False

try:
    from imblearn.over_sampling import SMOTE

    has_smote = True
except:
    has_smote = False

# Load MELD from Kaggle CSV (simpler than API)
print("Downloading MELD dataset...")
print("Note: MELD has EXACTLY 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise")

# Download directly
import urllib.request

url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv"
train_file = "meld_train.csv"

print("Downloading train data...")
try:
    urllib.request.urlretrieve(url, train_file)
    print("✓ Downloaded")
except:
    print("ERROR: Could not download MELD")
    print("Manual download: https://github.com/declare-lab/MELD/tree/master/data/MELD")
    exit()

# Load CSV
df_train = pd.read_csv(train_file)

print(f"Loaded {len(df_train)} samples")
print("\nColumns:", list(df_train.columns))

# MELD columns: Utterance (text), Emotion, Sentiment, etc.
# Map emotion names to match our format
emotion_map = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happiness",  # Map joy to happiness
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
}

# Prepare data
df_train["emotion_mapped"] = df_train["Emotion"].str.lower().map(emotion_map)
df_clean = df_train[["Utterance", "emotion_mapped"]].dropna()

print(f"\nClean data: {len(df_clean)} samples")
print("\nEmotion distribution:")
print(df_clean["emotion_mapped"].value_counts())

# Prepare
X = df_clean["Utterance"].values
y = df_clean["emotion_mapped"].values

le = LabelEncoder()
y_num = le.fit_transform(y)

print(f"\n7 emotions: {list(le.classes_)}")

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_num, test_size=0.15, random_state=42, stratify=y_num
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# TF-IDF
print("\nTF-IDF vectorization...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

# Sentiment
print("Extracting sentiment...")
sentiment_train = []
for text in X_train:
    if has_sent:
        try:
            sentiment_train.append(TextBlob(str(text)).sentiment.polarity)
        except:
            sentiment_train.append(0.0)
    else:
        sentiment_train.append(0.0)

sentiment_val = []
for text in X_val:
    if has_sent:
        try:
            sentiment_val.append(TextBlob(str(text)).sentiment.polarity)
        except:
            sentiment_val.append(0.0)
    else:
        sentiment_val.append(0.0)

sentiment_train = np.array(sentiment_train).reshape(-1, 1)
sentiment_val = np.array(sentiment_val).reshape(-1, 1)

# Combine
from scipy.sparse import hstack, csr_matrix

X_train_full = hstack([X_train_tfidf, csr_matrix(sentiment_train)])
X_val_full = hstack([X_val_tfidf, csr_matrix(sentiment_val)])

# SMOTE
if has_smote:
    print("Balancing with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_full, y_train)
    print(f"After SMOTE: {X_train_bal.shape[0]}")
else:
    X_train_bal = X_train_full
    y_train_bal = y_train

results = []

# MODEL 1: Logistic Regression
print("\n" + "=" * 60)
print("Logistic Regression + Sentiment")

start = time.time()

lr = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
lr.fit(X_train_bal, y_train_bal)

y_pred = lr.predict(X_val_full)
f1 = f1_score(y_val, y_pred, average="macro")
f1w = f1_score(y_val, y_pred, average="weighted")
acc = accuracy_score(y_val, y_pred)
t = time.time() - start

print(f"F1: {f1:.4f}, Accuracy: {acc:.4f}, Time: {t:.1f}s")

if f1 >= 0.75:
    print("🎯 MEETS F1 >= 0.75!")

# Save
csv = "model_iterations.csv"
from pathlib import Path

if Path(csv).exists():
    df_csv = pd.read_csv(csv)
    next_id = df_csv["iteration_id"].max() + 1
else:
    df_csv = pd.DataFrame()
    next_id = 1

new = {
    "iteration_id": next_id,
    "model_type": "traditional_ml",
    "model_name": "lr_sentiment_meld",
    "f1_macro": f1,
    "f1_weighted": f1w,
    "accuracy": acc,
    "training_time": t,
    "notes": "Logistic Regression with sentiment + SMOTE, MELD dataset (7 emotions native) | Code: train_MELD.py",
    "date_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

df_csv = pd.concat([df_csv, pd.DataFrame([new])], ignore_index=True)
df_csv.to_csv(csv, index=False)

print(f"Saved iteration #{next_id}")
results.append(("LR", f1, acc))

# MODEL 2: Random Forest
print("\n" + "=" * 60)
print("Random Forest + Sentiment")

start = time.time()

rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf.fit(X_train_bal, y_train_bal)

y_pred = rf.predict(X_val_full)
f1 = f1_score(y_val, y_pred, average="macro")
f1w = f1_score(y_val, y_pred, average="weighted")
acc = accuracy_score(y_val, y_pred)
t = time.time() - start

print(f"F1: {f1:.4f}, Accuracy: {acc:.4f}, Time: {t:.1f}s")

if f1 >= 0.75:
    print("🎯 MEETS F1 >= 0.75!")

df_csv = pd.read_csv(csv)
next_id = df_csv["iteration_id"].max() + 1

new = {
    "iteration_id": next_id,
    "model_type": "traditional_ml",
    "model_name": "rf_sentiment_meld",
    "f1_macro": f1,
    "f1_weighted": f1w,
    "accuracy": acc,
    "training_time": t,
    "notes": "Random Forest with sentiment + SMOTE, MELD dataset (7 emotions native) | Code: train_MELD.py",
    "date_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

df_csv = pd.concat([df_csv, pd.DataFrame([new])], ignore_index=True)
df_csv.to_csv(csv, index=False)

print(f"Saved iteration #{next_id}")
results.append(("RF", f1, acc))

# MODEL 3: Naive Bayes
print("\n" + "=" * 60)
print("Naive Bayes + Sentiment")

start = time.time()

count_vec = CountVectorizer(max_features=4000, ngram_range=(1, 2))
X_train_count = count_vec.fit_transform(X_train)
X_val_count = count_vec.transform(X_val)

sent_pos_train = sentiment_train + 1
sent_pos_val = sentiment_val + 1

X_train_nb = hstack([X_train_count, csr_matrix(sent_pos_train)])
X_val_nb = hstack([X_val_count, csr_matrix(sent_pos_val)])

if has_smote:
    smote_nb = SMOTE(random_state=42)
    X_train_nb_bal, y_train_nb_bal = smote_nb.fit_resample(X_train_nb, y_train)
else:
    X_train_nb_bal, y_train_nb_bal = X_train_nb, y_train

nb = MultinomialNB(alpha=0.1)
nb.fit(X_train_nb_bal, y_train_nb_bal)

y_pred = nb.predict(X_val_nb)
f1 = f1_score(y_val, y_pred, average="macro")
f1w = f1_score(y_val, y_pred, average="weighted")
acc = accuracy_score(y_val, y_pred)
t = time.time() - start

print(f"F1: {f1:.4f}, Accuracy: {acc:.4f}, Time: {t:.1f}s")

if f1 >= 0.75:
    print("🎯 MEETS F1 >= 0.75!")

df_csv = pd.read_csv(csv)
next_id = df_csv["iteration_id"].max() + 1

new = {
    "iteration_id": next_id,
    "model_type": "traditional_ml",
    "model_name": "nb_sentiment_meld",
    "f1_macro": f1,
    "f1_weighted": f1w,
    "accuracy": acc,
    "training_time": t,
    "notes": "Naive Bayes with sentiment + SMOTE, MELD dataset (7 emotions native) | Code: train_MELD.py",
    "date_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

df_csv = pd.concat([df_csv, pd.DataFrame([new])], ignore_index=True)
df_csv.to_csv(csv, index=False)

print(f"Saved iteration #{next_id}")
results.append(("NB", f1, acc))

# Summary
print("\n" + "=" * 60)
print("RESULTS - MELD DATASET (NATIVE 7 EMOTIONS)")
print("=" * 60)

for name, f1, acc in results:
    status = "🎯" if f1 >= 0.75 else "  "
    print(f"{status} {name}: F1={f1:.4f}, Acc={acc:.4f}")

best_f1 = max(r[1] for r in results)

if best_f1 >= 0.75:
    print(f"\n🎉 SUCCESS! F1={best_f1:.4f} >= 0.75")
else:
    print(f"\nBest F1: {best_f1:.4f}")

print("\nSaved: model_iterations.csv")
print("Dataset: MELD (13k samples, native 7 emotions)")
print("Emotions: anger, disgust, fear, happiness, neutral, sadness, surprise")
print("Source: Friends TV dialogues")
