# """
# Utility functions for emotion classification
# """
# import re
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import pickle
# import torch
#
# from src.config import RESULTS_DIR, DATA_DIR
#
# def load_preprocessed_data(model_type):
#     """
#     Load preprocessed data for model training - USE THIS IN YOUR MODELS
#
#     Args:
#         model_type: 'traditional_ml', 'deep_learning', 'bert', 'xlm_roberta'
#
#     Returns:
#         tuple: (processed_data, preprocessor, splits)
#
#     Example:
#         processed_data, preprocessor, splits = load_preprocessed_data('traditional_ml')
#         X_train = processed_data['X_train']
#         y_train = splits['y_train']
#     """
#     from src.data_loader import load_data_splits
#
#     processed_dir = DATA_DIR / "processed"
#     splits = load_data_splits()
#
#     print(f"Loading preprocessed data for {model_type}...")
#
#     if model_type == "traditional_ml":
#         with open(processed_dir / "traditional_ml_data.pkl", 'rb') as f:
#             data = pickle.load(f)
#         print(f"✅ Loaded traditional ML features: {data['features']['X_train'].shape}")
#         return data['features'], data['preprocessor'], splits
#
#     elif model_type == "deep_learning":
#         with open(processed_dir / "deep_learning_data.pkl", 'rb') as f:
#             data = pickle.load(f)
#         print(f"✅ Loaded deep learning sequences: {data['sequences']['X_train'].shape}")
#         return data['sequences'], data['preprocessor'], splits
#
#     elif model_type in ["bert", "xlm_roberta"]:
#         with open(processed_dir / f"{model_type}_data.pkl", 'rb') as f:
#             data = pickle.load(f)
#         print(f"✅ Loaded {model_type} encodings: {data['encodings']['X_train']['input_ids'].shape}")
#         return data['encodings'], data['preprocessor'], splits
#
#     else:
#         available = ['traditional_ml', 'deep_learning', 'bert', 'xlm_roberta']
#         raise ValueError(f"Unknown model_type: {model_type}. Available: {available}")
#
# def save_model(model, model_name, model_type="sklearn"):
#     """Save trained model"""
#     models_dir = RESULTS_DIR / "models"
#     models_dir.mkdir(parents=True, exist_ok=True)
#
#     if model_type == "sklearn":
#         model_path = models_dir / f"{model_name}.pkl"
#         with open(model_path, 'wb') as f:
#             pickle.dump(model, f)
#     elif model_type == "torch":
#         model_path = models_dir / f"{model_name}.pth"
#         torch.save(model.state_dict(), model_path)
#     elif model_type == "transformers":
#         model_path = models_dir / model_name
#         model.save_pretrained(model_path)
#
#     print(f"💾 Model saved to {model_path}")
#     return model_path
#
# def load_model(model_path, model_type="sklearn", model_class=None):
#     """Load saved model"""
#     if model_type == "sklearn":
#         with open(model_path, 'rb') as f:
#             return pickle.load(f)
#     elif model_type == "torch":
#         if model_class is None:
#             raise ValueError("model_class required for PyTorch models")
#         model = model_class()
#         model.load_state_dict(torch.load(model_path))
#         return model
#     elif model_type == "transformers":
#         from transformers import AutoModelForSequenceClassification
#         return AutoModelForSequenceClassification.from_pretrained(model_path)
#
# def plot_training_history(history, model_name="Model", save_path=None):
#     """Plot training history for deep learning models"""
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#
#     # Plot loss
#     axes[0].plot(history['train_loss'], label='Training Loss')
#     if 'val_loss' in history:
#         axes[0].plot(history['val_loss'], label='Validation Loss')
#     axes[0].set_title(f'{model_name} - Loss')
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylabel('Loss')
#     axes[0].legend()
#
#     # Plot accuracy
#     axes[1].plot(history['train_acc'], label='Training Accuracy')
#     if 'val_acc' in history:
#         axes[1].plot(history['val_acc'], label='Validation Accuracy')
#     axes[1].set_title(f'{model_name} - Accuracy')
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('Accuracy')
#     axes[1].legend()
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Training history plot saved to {save_path}")
#
#     plt.show()
#
# def analyze_class_performance(y_true, y_pred, emotion_classes, model_name="Model"):
#     """Analyze per-class performance"""
#     from sklearn.metrics import classification_report, precision_recall_fscore_support
#
#     # Get per-class metrics
#     precision, recall, f1, support = precision_recall_fscore_support(
#         y_true, y_pred, average=None
#     )
#
#     # Create detailed analysis
#     class_analysis = pd.DataFrame({
#         'Emotion': emotion_classes,
#         'Precision': precision,
#         'Recall': recall,
#         'F1-Score': f1,
#         'Support': support
#     })
#
#     # Sort by F1-score
#     class_analysis = class_analysis.sort_values('F1-Score', ascending=False)
#
#     print(f"\n{model_name} - Per-Class Performance Analysis:")
#     print("=" * 60)
#     print(class_analysis.to_string(index=False, float_format='%.3f'))
#
#     # Identify problematic classes
#     low_f1_classes = class_analysis[class_analysis['F1-Score'] < 0.5]['Emotion'].tolist()
#     if low_f1_classes:
#         print(f"\n⚠️  Classes with F1 < 0.5: {', '.join(low_f1_classes)}")
#
#     return class_analysis
#
# def plot_emotion_distribution(y, emotion_classes, title="Emotion Distribution", save_path=None):
#     """Plot emotion class distribution"""
#     # Convert label indices to emotion names if needed
#     if isinstance(y[0], int):
#         emotion_names = [emotion_classes[label] for label in y]
#     else:
#         emotion_names = y
#
#     # Count emotions
#     emotion_counts = pd.Series(emotion_names).value_counts()
#
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(emotion_counts.index, emotion_counts.values)
#
#     # Add count labels on bars
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{int(height)}', ha='center', va='bottom')
#
#     plt.title(title)
#     plt.xlabel('Emotion')
#     plt.ylabel('Count')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Distribution plot saved to {save_path}")
#
#     plt.show()
#
# def compare_models(results_dict, save_path=None):
#     """Compare multiple model results"""
#     comparison_data = []
#
#     for model_name, metrics in results_dict.items():
#         comparison_data.append({
#             'Model': model_name,
#             'F1-Macro': metrics.get('f1_macro', 0),
#             'F1-Weighted': metrics.get('f1_weighted', 0),
#             'Accuracy': metrics.get('accuracy', 0),
#             'Precision': metrics.get('precision_macro', 0),
#             'Recall': metrics.get('recall_macro', 0)
#         })
#
#     df_comparison = pd.DataFrame(comparison_data)
#
#     # Sort by F1-macro score
#     df_comparison = df_comparison.sort_values('F1-Macro', ascending=False)
#
#     print("\n📊 Model Comparison Results:")
#     print("=" * 80)
#     print(df_comparison.to_string(index=False, float_format='%.3f'))
#
#     # Plot comparison
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
#     # F1-Macro scores
#     axes[0].barh(df_comparison['Model'], df_comparison['F1-Macro'])
#     axes[0].set_title('F1-Macro Score')
#     axes[0].set_xlabel('F1-Macro')
#     axes[0].axvline(x=0.75, color='r', linestyle='--', alpha=0.7, label='Target: 0.75')
#     axes[0].legend()
#
#     # Accuracy
#     axes[1].barh(df_comparison['Model'], df_comparison['Accuracy'])
#     axes[1].set_title('Accuracy')
#     axes[1].set_xlabel('Accuracy')
#
#     # F1-Weighted
#     axes[2].barh(df_comparison['Model'], df_comparison['F1-Weighted'])
#     axes[2].set_title('F1-Weighted Score')
#     axes[2].set_xlabel('F1-Weighted')
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Comparison plot saved to {save_path}")
#
#     plt.show()
#
#     # Save comparison data
#     if save_path:
#         csv_path = Path(save_path).with_suffix('.csv')
#         df_comparison.to_csv(csv_path, index=False)
#         print(f"Comparison data saved to {csv_path}")
#
#     return df_comparison
#
# def check_data_quality(df, text_columns=['Original_Text', 'Translated_Text']):
#     """Check data quality issues"""
#     print("Data Quality Check:")
#     print("-" * 30)
#
#     total_samples = len(df)
#     print(f"Total samples: {total_samples}")
#
#     # Check for missing values
#     for col in text_columns:
#         missing = df[col].isna().sum()
#         if missing > 0:
#             print(f"⚠️  Missing values in {col}: {missing} ({missing/total_samples*100:.1f}%)")
#
#     # Check for empty strings
#     for col in text_columns:
#         empty = (df[col].astype(str).str.strip() == '').sum()
#         if empty > 0:
#             print(f"⚠️  Empty strings in {col}: {empty} ({empty/total_samples*100:.1f}%)")
#
#     # Check text length distribution
#     for col in text_columns:
#         lengths = df[col].astype(str).str.len()
#         print(f"\n{col} length stats:")
#         print(f"  Mean: {lengths.mean():.1f} chars")
#         print(f"  Median: {lengths.median():.1f} chars")
#         print(f"  Min: {lengths.min()} chars")
#         print(f"  Max: {lengths.max()} chars")
#
# def setup_results_directories():
#     """Create all necessary result directories"""
#     directories = [
#         RESULTS_DIR / "models",
#         RESULTS_DIR / "metrics",
#         RESULTS_DIR / "plots",
#         DATA_DIR / "processed"
#     ]
#
#     for directory in directories:
#         directory.mkdir(parents=True, exist_ok=True)
#
#     print("📁 Results directories created successfully!")
#
# if __name__ == "__main__":
#     # Test utilities
#     setup_results_directories()
#     print("✅ Utilities module tested successfully!")

"""
Utility functions for emotion classification (OPTIMIZED)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import torch

from src.config import RESULTS_DIR, DATA_DIR


def load_preprocessed_data(model_type):
    """Load preprocessed data for model training"""
    from src.data_loader import load_data_splits

    processed_dir = DATA_DIR / "processed"
    splits = load_data_splits()

    print(f"Loading preprocessed data for {model_type}...")

    try:
        if model_type == "traditional_ml":
            with open(processed_dir / "traditional_ml_data.pkl", "rb") as f:
                data = pickle.load(f)
            print(f"Loaded traditional ML features: {data['features']['X_train'].shape}")
            return data["features"], data["preprocessor"], splits

        elif model_type == "deep_learning":
            with open(processed_dir / "deep_learning_data.pkl", "rb") as f:
                data = pickle.load(f)
            print(f"Loaded deep learning sequences: {data['sequences']['X_train'].shape}")
            return data["sequences"], data["preprocessor"], splits

        elif model_type in ["bert", "xlm_roberta"]:
            with open(processed_dir / f"{model_type}_data.pkl", "rb") as f:
                data = pickle.load(f)
            print(
                f"Loaded {model_type} encodings: {data['encodings']['X_train']['input_ids'].shape}"
            )
            return data["encodings"], data["preprocessor"], splits

        else:
            available = ["traditional_ml", "deep_learning", "bert", "xlm_roberta"]
            raise ValueError(f"Unknown model_type: {model_type}. Available: {available}")

    except FileNotFoundError:
        print(f"Preprocessed data not found for {model_type}. Run preprocessing first.")
        raise
    except Exception as e:
        print(f"Error loading {model_type} data: {e}")
        raise


def save_model(model, model_name, model_type="sklearn"):
    """Save trained model with appropriate format"""
    models_dir = RESULTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        if model_type == "sklearn":
            model_path = models_dir / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        elif model_type == "torch":
            model_path = models_dir / f"{model_name}.pth"
            torch.save(model.state_dict(), model_path)
        elif model_type == "transformers":
            model_path = models_dir / model_name
            model.save_pretrained(model_path)

        print(f"Model saved to {model_path}")
        return model_path

    except Exception as e:
        print(f"Error saving model: {e}")
        raise


def setup_results_directories():
    """Create necessary result directories"""
    directories = [
        RESULTS_DIR / "models",
        RESULTS_DIR / "metrics",
        RESULTS_DIR / "plots",
        DATA_DIR / "processed",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def check_data_quality(df, text_columns=["Original_Text", "Translated_Text"]):
    """Comprehensive data quality check"""
    print("Data Quality Analysis:")
    print("-" * 30)

    total_samples = len(df)
    print(f"Total samples: {total_samples}")

    for col in text_columns:
        # Missing values
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"Missing values in {col}: {missing} ({missing / total_samples * 100:.1f}%)")

        # Empty strings
        empty = (df[col].astype(str).str.strip() == "").sum()
        if empty > 0:
            print(f"Empty strings in {col}: {empty} ({empty / total_samples * 100:.1f}%)")

        # Text statistics
        lengths = df[col].astype(str).str.len()
        print(f"\n{col} statistics:")
        print(f"  Length - Mean: {lengths.mean():.1f}, Median: {lengths.median():.1f}")
        print(f"  Length - Min: {lengths.min()}, Max: {lengths.max()}")

        # Very short texts (potential quality issues)
        very_short = (lengths < 10).sum()
        if very_short > 0:
            print(f"  Very short texts (<10 chars): {very_short}")


def create_ensemble_predictions(model_predictions_dict):
    """Create ensemble predictions from multiple models"""
    if not model_predictions_dict:
        raise ValueError("No model predictions provided")

    # Simple voting ensemble
    all_predictions = np.array(list(model_predictions_dict.values()))

    # Majority voting
    ensemble_predictions = []
    for sample_idx in range(all_predictions.shape[1]):
        sample_predictions = all_predictions[:, sample_idx]
        # Get most common prediction
        unique, counts = np.unique(sample_predictions, return_counts=True)
        majority_vote = unique[np.argmax(counts)]
        ensemble_predictions.append(majority_vote)

    return np.array(ensemble_predictions)


def generate_final_report(csv_file="model_iterations.csv", save_path=None):
    """Generate comprehensive final report"""
    try:
        df = pd.read_csv(csv_file)
        latest_results = df.groupby("model_name").last().reset_index()

        report = {
            "total_models": len(latest_results),
            "models_above_threshold": (latest_results["f1_macro"] >= 0.75).sum(),
            "best_model": latest_results.loc[latest_results["f1_macro"].idxmax()],
            "worst_model": latest_results.loc[latest_results["f1_macro"].idxmin()],
            "average_f1": latest_results["f1_macro"].mean(),
            "traditional_ml_best": latest_results[latest_results["model_type"] == "traditional_ml"][
                "f1_macro"
            ].max(),
            "deep_learning_best": latest_results[latest_results["model_type"] == "deep_learning"][
                "f1_macro"
            ].max(),
            "transformer_best": latest_results[latest_results["model_type"] == "transformer"][
                "f1_macro"
            ].max(),
        }

        # Generate report text
        report_text = f"""
EMOTION CLASSIFICATION PROJECT - FINAL REPORT

OVERVIEW:
- Total models implemented: {report["total_models"]}
- Models meeting F1 > 0.75 threshold: {report["models_above_threshold"]}
- Average F1-score across all models: {report["average_f1"]:.3f}

BEST PERFORMERS:
- Overall best: {report["best_model"]["model_name"]} (F1: {report["best_model"]["f1_macro"]:.3f})
- Traditional ML best: F1 = {report["traditional_ml_best"]:.3f}
- Deep Learning best: F1 = {report["deep_learning_best"]:.3f}
- Transformer best: F1 = {report["transformer_best"]:.3f}

ANALYSIS:
The primary challenge is data scarcity with only ~57 samples per emotion class.
This limits all model types' ability to learn robust emotion patterns.

RECOMMENDATIONS:
1. Data augmentation to increase sample size
2. Focus on transformer models for best performance
3. Ensemble approach combining top performers
4. Feature engineering for traditional ML models
"""

        print(report_text)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")

        return report

    except Exception as e:
        print(f"Error generating report: {e}")
        return None


if __name__ == "__main__":
    setup_results_directories()
    print("Utilities optimization completed!")
