# """
# Data loading and splitting functionality (FIXED COLUMN NAMES)
# """
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import pickle
# from pathlib import Path
#
# from src.config import DATA_FILE, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE, DATA_DIR
#
# def load_raw_data():
#     """Load raw emotion data from Excel file"""
#     print(f"Loading data from {DATA_FILE}")
#
#     # Check if file exists
#     if not DATA_FILE.exists():
#         raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
#
#     df = pd.read_excel(DATA_FILE)
#
#     print(f"Loaded {len(df)} samples")
#     print(f"Columns: {list(df.columns)}")
#
#     # Basic data validation - UPDATED COLUMN NAMES
#     required_cols = ['Original_Text', 'Translated_Text', 'Predicted_Emotion', 'Confidence']
#     missing_cols = [col for col in required_cols if col not in df.columns]
#     if missing_cols:
#         raise ValueError(f"Missing required columns: {missing_cols}")
#
#     # Display emotion distribution - UPDATED COLUMN NAME
#     emotion_counts = df['Predicted_Emotion'].value_counts()
#     print("\nEmotion distribution:")
#     for emotion, count in emotion_counts.items():
#         percentage = (count / len(df)) * 100
#         print(f"  {emotion}: {count} ({percentage:.1f}%)")
#
#     return df
#
# def create_data_splits(df=None, force_recreate=False):
#     """Create train/validation/test splits with stratification"""
#
#     # Load data if not provided
#     if df is None:
#         df = load_raw_data()
#
#     # Check if splits already exist
#     processed_dir = DATA_DIR / "processed"
#     processed_dir.mkdir(exist_ok=True)
#     splits_path = processed_dir / "data_splits.pkl"
#
#     if splits_path.exists() and not force_recreate:
#         print(f"Loading existing splits from {splits_path}")
#         with open(splits_path, 'rb') as f:
#             return pickle.load(f)
#
#     print("Creating new data splits...")
#
#     # Features and labels - UPDATED COLUMN NAMES
#     X = df[['Original_Text', 'Translated_Text', 'Confidence']].copy()
#     y = df['Predicted_Emotion'].copy()
#
#     # Remove any null values
#     mask = X['Original_Text'].notna() & X['Translated_Text'].notna() & y.notna()
#     X = X[mask]
#     y = y[mask]
#
#     print(f"After cleaning: {len(X)} samples")
#
#     # Encode labels
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)
#
#     # Create train/test split
#     X_temp, X_test, y_temp, y_test = train_test_split(
#         X, y_encoded,
#         test_size=TEST_SIZE,
#         stratify=y_encoded,
#         random_state=RANDOM_STATE
#     )
#
#     # Create train/validation split
#     validation_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_temp, y_temp,
#         test_size=validation_size_adjusted,
#         stratify=y_temp,
#         random_state=RANDOM_STATE
#     )
#
#     print(f"\nData splits:")
#     print(f"  Training: {len(X_train)} samples")
#     print(f"  Validation: {len(X_val)} samples")
#     print(f"  Test: {len(X_test)} samples")
#
#     # Print class distribution for each split
#     for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
#         print(f"\n{split_name} class distribution:")
#         unique, counts = np.unique(y_split, return_counts=True)
#         for label_idx, count in zip(unique, counts):
#             emotion = label_encoder.inverse_transform([label_idx])[0]
#             print(f"  {emotion}: {count}")
#
#     splits_data = {
#         'X_train': X_train,
#         'X_val': X_val,
#         'X_test': X_test,
#         'y_train': y_train,
#         'y_val': y_val,
#         'y_test': y_test,
#         'label_encoder': label_encoder,
#         'emotion_classes': label_encoder.classes_
#     }
#
#     # Save splits for reuse
#     with open(splits_path, 'wb') as f:
#         pickle.dump(splits_data, f)
#
#     print(f"\nSplits saved to {splits_path}")
#     return splits_data
#
# def load_data_splits():
#     """Load or create data splits"""
#     return create_data_splits(force_recreate=False)
#
# if __name__ == "__main__":
#     # Test the data loading
#     print("Testing data loading...")
#     splits = create_data_splits(force_recreate=True)
#     print("\nData loading test completed successfully!")

"""
Data loading and splitting functionality (OPTIMIZED)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

from src.config import DATA_FILE, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE, DATA_DIR


def load_raw_data():
    """Load and validate emotion data"""
    print(f"Loading data from {DATA_FILE}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE)
    print(f"Loaded {len(df)} samples with columns: {list(df.columns)}")

    # Validate required columns
    required_cols = ["Original_Text", "Translated_Text", "Predicted_Emotion", "Confidence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Show distribution
    emotion_counts = df["Predicted_Emotion"].value_counts()
    print("\nEmotion distribution:")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {emotion}: {count} ({percentage:.1f}%)")

    # Calculate data scarcity metrics
    samples_per_class = len(df) / len(emotion_counts)
    print("\nData scarcity analysis:")
    print(f"  Average samples per class: {samples_per_class:.1f}")
    print("  Minimum recommended: 100+ per class")
    print(f"  Data scarcity factor: {100 / samples_per_class:.1f}x too small")

    return df


def create_data_splits(df=None, force_recreate=False):
    """Create optimized stratified splits"""

    if df is None:
        df = load_raw_data()

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)
    splits_path = processed_dir / "data_splits.pkl"

    if splits_path.exists() and not force_recreate:
        print(f"Loading existing splits from {splits_path}")
        with open(splits_path, "rb") as f:
            return pickle.load(f)

    print("Creating optimized data splits...")

    # Prepare features and labels
    X = df[["Original_Text", "Translated_Text", "Confidence"]].copy()
    y = df["Predicted_Emotion"].copy()

    # Clean data
    mask = (
        X["Original_Text"].notna()
        & X["Translated_Text"].notna()
        & y.notna()
        & (X["Original_Text"].str.len() > 5)  # Remove very short texts
        & (X["Translated_Text"].str.len() > 5)
    )

    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    print(f"After cleaning: {len(X)} samples")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Check class distribution for splitting
    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    min_class_count = min(class_counts)

    if min_class_count < 3:
        print("Warning: Some classes have < 3 samples. Consider combining classes.")

    # Create splits with stratification
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=RANDOM_STATE
    )

    validation_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=validation_size_adjusted,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    # Report split statistics
    print("\nOptimized splits created:")
    print(
        f"  Training: {len(X_train)} samples ({len(X_train) / len(unique_classes):.1f} per class)"
    )
    print(f"  Validation: {len(X_val)} samples ({len(X_val) / len(unique_classes):.1f} per class)")
    print(f"  Test: {len(X_test)} samples ({len(X_test) / len(unique_classes):.1f} per class)")

    # Verify stratification worked
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        min_count = min(counts)
        max_count = max(counts)
        if min_count == 0:
            print(f"Warning: {split_name} split missing some emotion classes")

    splits_data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "emotion_classes": label_encoder.classes_,
    }

    # Save splits
    with open(splits_path, "wb") as f:
        pickle.dump(splits_data, f)

    print(f"Splits saved to {splits_path}")
    return splits_data


def load_data_splits():
    """Load existing data splits"""
    return create_data_splits(force_recreate=False)


def create_cross_validation_splits(df=None, n_splits=5):
    """Create cross-validation splits for small dataset evaluation"""
    if df is None:
        df = load_raw_data()

    X = df[["Original_Text", "Translated_Text", "Confidence"]].copy()
    y = df["Predicted_Emotion"].copy()

    # Clean data
    mask = X["Original_Text"].notna() & X["Translated_Text"].notna() & y.notna()
    X, y = X[mask], y[mask]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Create stratified CV splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = []

    for train_idx, val_idx in skf.split(X, y_encoded):
        cv_splits.append(
            {
                "X_train": X.iloc[train_idx],
                "X_val": X.iloc[val_idx],
                "y_train": y_encoded[train_idx],
                "y_val": y_encoded[val_idx],
            }
        )

    print(f"Created {n_splits}-fold cross-validation splits")
    return cv_splits, label_encoder


if __name__ == "__main__":
    # Test optimized data loading
    splits = create_data_splits(force_recreate=True)
    print("Optimized data loading completed!")
