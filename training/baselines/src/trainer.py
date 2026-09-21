# """
# Training and evaluation utilities
# """
# import time
# import pandas as pd
# import numpy as np
# from sklearn.metrics import (
#     classification_report, confusion_matrix,
#     f1_score, accuracy_score, precision_score, recall_score
# )
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import csv
#
# from src.config import MIN_F1_SCORE, EMOTION_CLASSES, RESULTS_DIR, PROJECT_ROOT
#
#
# class ModelTracker:
#     """Track model iterations for the assignment"""
#
#     def __init__(self, tracking_file=None):
#         if tracking_file is None:
#             self.tracking_file = PROJECT_ROOT / "model_iterations.csv"
#         else:
#             self.tracking_file = tracking_file
#
#         # Ensure file exists with headers
#         if not self.tracking_file.exists():
#             self.create_tracking_file()
#
#     def create_tracking_file(self):
#         """Create tracking file with headers"""
#         headers = [
#             'iteration_id', 'model_type', 'model_name',
#             'f1_macro', 'f1_weighted', 'accuracy',
#             'training_time', 'notes', 'date_run'
#         ]
#
#         with open(self.tracking_file, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(headers)
#
#     def log_iteration(self, model_type, model_name, metrics, training_time, notes=""):
#         """Log a model iteration"""
#         from datetime import datetime
#
#         # Get next iteration ID
#         try:
#             df = pd.read_csv(self.tracking_file)
#             next_id = df['iteration_id'].max() + 1 if not df.empty else 1
#         except:
#             next_id = 1
#
#         # Prepare row data
#         row_data = [
#             next_id,
#             model_type,
#             model_name,
#             metrics.get('f1_macro', 0),
#             metrics.get('f1_weighted', 0),
#             metrics.get('accuracy', 0),
#             training_time,
#             notes,
#             datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         ]
#
#         # Append to file
#         with open(self.tracking_file, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(row_data)
#
#         print(f"📊 Logged iteration {next_id} for {model_name}")
#
#         # Check if meets threshold
#         f1_macro = metrics.get('f1_macro', 0)
#         if f1_macro >= MIN_F1_SCORE:
#             print(f"🎯 Model {model_name} meets F1 threshold: {f1_macro:.3f} >= {MIN_F1_SCORE}")
#         else:
#             print(f"⚠️  Model {model_name} below F1 threshold: {f1_macro:.3f} < {MIN_F1_SCORE}")
#
#
# class ModelEvaluator:
#     """Evaluate model performance"""
#
#     def __init__(self, emotion_classes=None):
#         if emotion_classes is None:
#             self.emotion_classes = EMOTION_CLASSES
#         else:
#             self.emotion_classes = emotion_classes
#
#     def evaluate_model(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
#         """Comprehensive model evaluation"""
#
#         # Basic metrics
#         accuracy = accuracy_score(y_true, y_pred)
#         f1_macro = f1_score(y_true, y_pred, average='macro')
#         f1_weighted = f1_score(y_true, y_pred, average='weighted')
#         precision_macro = precision_score(y_true, y_pred, average='macro')
#         recall_macro = recall_score(y_true, y_pred, average='macro')
#
#         metrics = {
#             'accuracy': accuracy,
#             'f1_macro': f1_macro,
#             'f1_weighted': f1_weighted,
#             'precision_macro': precision_macro,
#             'recall_macro': recall_macro
#         }
#
#         print(f"\n📊 {model_name} Performance:")
#         print("-" * 50)
#         print(f"Accuracy: {accuracy:.3f}")
#         print(f"F1-Score (Macro): {f1_macro:.3f}")
#         print(f"F1-Score (Weighted): {f1_weighted:.3f}")
#         print(f"Precision (Macro): {precision_macro:.3f}")
#         print(f"Recall (Macro): {recall_macro:.3f}")
#
#         # Classification report
#         print(f"\nDetailed Classification Report:")
#         print(classification_report(y_true, y_pred, target_names=self.emotion_classes))
#
#         return metrics
#
#     def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", save_path=None):
#         """Plot confusion matrix"""
#         cm = confusion_matrix(y_true, y_pred)
#
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                     xticklabels=self.emotion_classes,
#                     yticklabels=self.emotion_classes)
#         plt.title(f'Confusion Matrix - {model_name}')
#         plt.ylabel('True Label')
#         plt.xlabel('Predicted Label')
#
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             print(f"Confusion matrix saved to {save_path}")
#
#         plt.show()
#         return cm
#
#     def save_predictions(self, y_true, y_pred, y_pred_proba=None,
#                          model_name="model", save_dir=None):
#         """Save model predictions"""
#         if save_dir is None:
#             save_dir = RESULTS_DIR / "metrics"
#
#         save_dir = Path(save_dir)
#         save_dir.mkdir(parents=True, exist_ok=True)
#
#         # Create predictions dataframe
#         predictions_df = pd.DataFrame({
#             'true_label': [self.emotion_classes[label] for label in y_true],
#             'predicted_label': [self.emotion_classes[label] for label in y_pred],
#             'correct': y_true == y_pred
#         })
#
#         # Add probabilities if available
#         if y_pred_proba is not None:
#             for i, emotion in enumerate(self.emotion_classes):
#                 predictions_df[f'prob_{emotion}'] = y_pred_proba[:, i]
#
#         # Save predictions
#         pred_file = save_dir / f"{model_name}_predictions.csv"
#         predictions_df.to_csv(pred_file, index=False)
#         print(f"Predictions saved to {pred_file}")
#
#         return predictions_df
#
#
# def time_model_training(func):
#     """Decorator to time model training"""
#
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         training_time = end_time - start_time
#
#         print(f"Training completed in {training_time:.2f} seconds")
#         return result, training_time
#
#     return wrapper
#
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
#     axes[0].axvline(x=MIN_F1_SCORE, color='r', linestyle='--', alpha=0.7, label=f'Target: {MIN_F1_SCORE}')
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

"""
Training and evaluation utilities (OPTIMIZED)
"""

import time
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import csv

from src.config import MIN_F1_SCORE, EMOTION_CLASSES, RESULTS_DIR, PROJECT_ROOT


class ModelTracker:
    """Optimized model iteration tracking"""

    def __init__(self, tracking_file=None):
        self.tracking_file = tracking_file or PROJECT_ROOT / "model_iterations.csv"

        if not self.tracking_file.exists():
            self._create_tracking_file()

    def _create_tracking_file(self):
        """Create tracking file with headers"""
        headers = [
            "iteration_id",
            "model_type",
            "model_name",
            "f1_macro",
            "f1_weighted",
            "accuracy",
            "training_time",
            "notes",
            "date_run",
        ]

        with open(self.tracking_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_iteration(self, model_type, model_name, metrics, training_time, notes=""):
        """Log model iteration with validation"""
        from datetime import datetime

        # Validate metrics
        required_metrics = ["f1_macro", "f1_weighted", "accuracy"]
        for metric in required_metrics:
            if metric not in metrics:
                raise ValueError(f"Missing required metric: {metric}")

        # Get next iteration ID
        try:
            df = pd.read_csv(self.tracking_file)
            next_id = df["iteration_id"].max() + 1 if not df.empty else 1
        except (FileNotFoundError, pd.errors.EmptyDataError):
            next_id = 1

        # Prepare and save data
        row_data = [
            next_id,
            model_type,
            model_name,
            round(metrics["f1_macro"], 4),
            round(metrics["f1_weighted"], 4),
            round(metrics["accuracy"], 4),
            round(training_time, 2),
            notes,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]

        with open(self.tracking_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        # Performance feedback
        f1_macro = metrics["f1_macro"]
        status = "MEETS" if f1_macro >= MIN_F1_SCORE else "BELOW"
        print(f"Logged iteration {next_id}: {model_name} F1={f1_macro:.3f} ({status} threshold)")

    def get_best_models(self, top_n=3):
        """Get top N performing models"""
        try:
            df = pd.read_csv(self.tracking_file)
            latest_results = df.groupby("model_name").last().reset_index()
            best_models = latest_results.nlargest(top_n, "f1_macro")
            return best_models[["model_name", "model_type", "f1_macro", "training_time"]]
        except Exception as e:
            print(f"Error getting best models: {e}")
            return None


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, emotion_classes=None):
        self.emotion_classes = emotion_classes or EMOTION_CLASSES

    def evaluate_model(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """Evaluate model with comprehensive metrics"""

        # Suppress warnings for ill-defined metrics
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

        # Display results
        print(f"\n{model_name} Performance:")
        print("-" * 50)
        for metric_name, value in metrics.items():
            print(f"{metric_name.replace('_', ' ').title()}: {value:.3f}")

        # Check prediction diversity
        unique_predictions = len(np.unique(y_pred))
        total_classes = len(self.emotion_classes)

        if unique_predictions < total_classes:
            missing_classes = total_classes - unique_predictions
            print(f"\nWarning: Model only predicts {unique_predictions}/{total_classes} classes")
            print(f"Missing {missing_classes} emotion classes in predictions")

        return metrics

    def create_confusion_matrix(self, y_true, y_pred, model_name="Model", save_path=None):
        """Create and optionally save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.emotion_classes,
            yticklabels=self.emotion_classes,
            cbar_kws={"label": "Count"},
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Emotion")
        plt.xlabel("Predicted Emotion")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to {save_path}")

        plt.show()
        return cm


def compare_all_models(csv_file="model_iterations.csv", save_path=None):
    """Compare all trained models from tracking file"""
    try:
        df = pd.read_csv(csv_file)
        latest_results = df.groupby("model_name").last().reset_index()

        # Sort by F1-macro score
        latest_results = latest_results.sort_values("f1_macro", ascending=False)

        print("\nFINAL MODEL COMPARISON:")
        print("=" * 80)

        # Summary table
        summary_cols = ["model_name", "model_type", "f1_macro", "accuracy", "training_time"]
        display_df = latest_results[summary_cols].copy()
        display_df["training_time"] = display_df["training_time"].round(1)
        print(display_df.to_string(index=False, float_format="%.3f"))

        # Performance analysis
        print("\nPERFORMANCE ANALYSIS:")
        print(
            f"Best model: {latest_results.iloc[0]['model_name']} (F1: {latest_results.iloc[0]['f1_macro']:.3f})"
        )
        print(
            f"Models above threshold: {(latest_results['f1_macro'] >= 0.75).sum()}/{len(latest_results)}"
        )

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # F1-Score comparison
        bars = ax1.barh(latest_results["model_name"], latest_results["f1_macro"])
        ax1.axvline(x=0.75, color="red", linestyle="--", alpha=0.7, label="Target: 0.75")
        ax1.set_xlabel("F1-Score (Macro)")
        ax1.set_title("Model Performance Comparison")
        ax1.legend()
        ax1.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax1.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="left",
                va="center",
                fontweight="bold",
            )

        # Training time vs performance
        type_colors = {
            "traditional_ml": "#1f77b4",
            "deep_learning": "#ff7f0e",
            "transformer": "#2ca02c",
        }
        for model_type in type_colors:
            type_data = latest_results[latest_results["model_type"] == model_type]
            if not type_data.empty:
                ax2.scatter(
                    type_data["training_time"],
                    type_data["f1_macro"],
                    color=type_colors[model_type],
                    label=model_type.replace("_", " "),
                    s=100,
                )

        ax2.set_xlabel("Training Time (seconds, log scale)")
        ax2.set_ylabel("F1-Score")
        ax2.set_title("Performance vs Training Time")
        ax2.set_xscale("log")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Comparison plot saved to {save_path}")

        plt.show()

        return latest_results

    except Exception as e:
        print(f"Error comparing models: {e}")
        return None


if __name__ == "__main__":
    compare_all_models()
