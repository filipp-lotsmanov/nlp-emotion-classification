# """
# Configuration settings for emotion classification project
# """
# from pathlib import Path
#
# # Project paths
# PROJECT_ROOT = Path(__file__).parent.parent
# DATA_DIR = PROJECT_ROOT / "data"
# RESULTS_DIR = PROJECT_ROOT / "results"
#
# # Data file
# DATA_FILE = DATA_DIR / "raw" / "balanced_dataset.xlsx"
#
# # Model configuration
# EMOTION_CLASSES = ['neutral', 'disgust', 'surprise', 'anger', 'sadness', 'happiness', 'fear']
# NUM_CLASSES = len(EMOTION_CLASSES)
#
# # Training parameters
# RANDOM_STATE = 42
# TEST_SIZE = 0.2
# VALIDATION_SIZE = 0.15
#
# # Text processing parameters
# MAX_SEQUENCE_LENGTH = 128
# MAX_FEATURES_TFIDF = 10000
#
# # Traditional ML parameters
# TRADITIONAL_ML_PARAMS = {
#     'logistic_regression': {
#         'class_weight': 'balanced',
#         'max_iter': 1000,
#         'random_state': RANDOM_STATE
#     },
#     'naive_bayes': {
#         'alpha': 1.0
#     },
#     'svm': {
#         'class_weight': 'balanced',
#         'kernel': 'rbf',
#         'random_state': RANDOM_STATE
#     }
# }
#
# # Deep learning parameters
# DEEP_LEARNING_PARAMS = {
#     'batch_size': 32,
#     'epochs': 50,
#     'patience': 10,
#     'learning_rate': 0.001,
#     'embedding_dim': 128,
#     'hidden_size': 64,
#     'dropout': 0.3
# }
#
# # Transformer parameters
# TRANSFORMER_PARAMS = {
#     'batch_size': 16,
#     'epochs': 5,
#     'learning_rate': 2e-5,
#     'max_length': 256,
#     'warmup_steps': 500
# }
#
# # Transformer models
# TRANSFORMER_MODELS = {
#     'bert': 'bert-base-uncased',
#     'xlm_roberta': 'xlm-roberta-base'
# }
#
# # Performance threshold
# MIN_F1_SCORE = 0.75

"""
Configuration settings for emotion classification project (OPTIMIZED)
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data file
DATA_FILE = DATA_DIR / "raw" / "balanced_dataset.xlsx"

# Model configuration
EMOTION_CLASSES = ["neutral", "disgust", "surprise", "anger", "sadness", "happiness", "fear"]
NUM_CLASSES = len(EMOTION_CLASSES)

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

# Text processing parameters (optimized for small dataset)
MAX_SEQUENCE_LENGTH = 256  # Increased for better context
MAX_FEATURES_TFIDF = 15000  # Increased for richer features

# Traditional ML parameters (optimized for small dataset)
TRADITIONAL_ML_PARAMS = {
    "logistic_regression": {
        "class_weight": "balanced",
        "max_iter": 10000,  # Much higher for convergence
        "solver": "liblinear",  # Better for small datasets
        "C": 0.01,  # Strong regularization for small data
        "penalty": "l2",
        "multi_class": "ovr",  # One-vs-rest for imbalanced data
        "random_state": RANDOM_STATE,
    },
    "naive_bayes": {
        "alpha": 0.1  # Lower smoothing for small data
    },
    "svm": {
        "class_weight": "balanced",
        "kernel": "linear",  # Linear kernel faster and often better for small data
        "C": 0.1,  # Lower C for small dataset
        "probability": True,
        "random_state": RANDOM_STATE,
    },
}

# Deep learning parameters (optimized for small dataset)
DEEP_LEARNING_PARAMS = {
    "batch_size": 16,  # Smaller batches for small dataset
    "epochs": 100,  # More epochs but with early stopping
    "patience": 15,  # Early stopping patience
    "learning_rate": 0.01,  # Higher learning rate for small data
    "embedding_dim": 256,  # Larger embeddings for richer representations
    "hidden_size": 128,  # Larger hidden size
    "dropout": 0.5,  # Higher dropout to prevent overfitting
    "weight_decay": 1e-4,  # L2 regularization
}

# Transformer parameters (optimized for small dataset)
TRANSFORMER_PARAMS = {
    "batch_size": 8,  # Smaller batches for stability
    "epochs": 10,  # More epochs for better learning
    "learning_rate": 1e-5,  # Lower learning rate for stability
    "max_length": 512,  # Longer sequences for better context
    "warmup_steps": 100,  # Fewer warmup steps
    "weight_decay": 0.01,  # Regularization
    "gradient_accumulation_steps": 2,  # Simulate larger batch size
}

# Transformer models
TRANSFORMER_MODELS = {"bert": "bert-base-uncased", "xlm_roberta": "xlm-roberta-base"}

# Performance threshold
MIN_F1_SCORE = 0.75

# Data scarcity adjustments
SMALL_DATA_ADJUSTMENTS = {
    "use_cross_validation": True,  # Use CV instead of single split
    "cv_folds": 5,
    "feature_selection": True,  # Use feature selection for traditional ML
    "data_augmentation": True,  # Enable text augmentation
    "ensemble_top_n": 3,  # Ensemble top 3 models
}
