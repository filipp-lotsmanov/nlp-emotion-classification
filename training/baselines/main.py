# print('=' * 60)
# print('DEBUGGING PREPROCESSING STEPS')
# print('=' * 60)
#
# import pickle
# from pathlib import Path
# from src.data_loader import load_raw_data, create_data_splits
# from src.config import DATA_DIR, TRANSFORMER_MODELS
# from src.utils import setup_results_directories
#
# # Load data
# df = load_raw_data()
# splits = create_data_splits(df)
# setup_results_directories()
# processed_dir = DATA_DIR / 'processed'
#
# # Ensure processed directory exists
# processed_dir.mkdir(exist_ok=True)
#
# print('\n1. Testing Traditional ML preprocessing...')
# try:
#     from src.preprocessors import TraditionalMLPreprocessor
#
#     trad_preprocessor = TraditionalMLPreprocessor(use_tfidf=True, ngram_range=(1, 3))
#     trad_preprocessor.fit(splits['X_train'])
#
#     # Save the preprocessor
#     with open(processed_dir / 'traditional_ml_preprocessor.pkl', 'wb') as f:
#         pickle.dump(trad_preprocessor, f)
#
#     print('✅ Traditional ML works and saved')
# except Exception as e:
#     print(f'❌ Traditional ML failed: {e}')
#
# print('\n2. Testing Deep Learning preprocessing...')
# try:
#     from src.preprocessors import DeepLearningPreprocessor
#
#     dl_preprocessor = DeepLearningPreprocessor(max_length=128, vocab_size=10000)
#     dl_preprocessor.fit(splits['X_train'], use_language='both')
#
#     # Save the preprocessor
#     with open(processed_dir / 'deep_learning_preprocessor.pkl', 'wb') as f:
#         pickle.dump(dl_preprocessor, f)
#
#     print('✅ Deep Learning works and saved')
# except Exception as e:
#     print(f'❌ Deep Learning failed: {e}')
#
# print('\n3. Testing BERT preprocessing...')
# try:
#     from src.preprocessors import TransformerPreprocessor
#
#     bert_preprocessor = TransformerPreprocessor(model_name='bert-base-uncased', max_length=256)
#     bert_preprocessor.fit(splits['X_train'])
#
#     # Save the preprocessor
#     with open(processed_dir / 'bert_preprocessor.pkl', 'wb') as f:
#         pickle.dump(bert_preprocessor, f)
#
#     print('✅ BERT works and saved')
# except Exception as e:
#     print(f'❌ BERT failed: {e}')
#
# print('\n4. Testing XLM-RoBERTa preprocessing...')
# try:
#     xlm_preprocessor = TransformerPreprocessor(model_name='xlm-roberta-base', max_length=256)
#     xlm_preprocessor.fit(splits['X_train'])
#
#     # Save the preprocessor
#     with open(processed_dir / 'xlm_roberta_preprocessor.pkl', 'wb') as f:
#         pickle.dump(xlm_preprocessor, f)
#
#     print('✅ XLM-RoBERTa works and saved')
# except Exception as e:
#     print(f'❌ XLM-RoBERTa failed: {e}')
#
# print(f'\n📁 Saved files in {processed_dir}:')
# for pkl_file in processed_dir.glob('*.pkl'):
#     print(f'  - {pkl_file.name}')

"""
Main entry point for emotion classification project
"""

import time
import pickle
from pathlib import Path

from src.data_loader import load_raw_data, create_data_splits, load_data_splits
from src.preprocessors import (
    TraditionalMLPreprocessor,
    DeepLearningPreprocessor,
    TransformerPreprocessor,
)
from src.utils import setup_results_directories, check_data_quality
from src.config import DATA_DIR, TRANSFORMER_MODELS


def preprocess_all_data():
    """Comprehensive data preprocessing for all model types"""
    print("=" * 60)
    print("PREPROCESSING DATA FOR ALL MODELS")
    print("=" * 60)

    # Setup
    setup_results_directories()
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)

    # Load and validate data
    print("\n1. Loading and validating data...")
    df = load_raw_data()
    check_data_quality(df)

    # Create stratified splits
    print("\n2. Creating stratified data splits...")
    splits = create_data_splits(df, force_recreate=True)  # Force recreate for consistency
    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]

    print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    preprocessing_results = {}

    # Traditional ML preprocessing
    print("\n3. Traditional ML preprocessing...")
    try:
        trad_preprocessor = TraditionalMLPreprocessor(
            use_tfidf=True,
            ngram_range=(1, 3),  # Include trigrams for better context
            max_features=15000,  # Increase features for better performance
        )
        trad_preprocessor.fit(X_train)

        trad_features = {
            "X_train": trad_preprocessor.transform(X_train),
            "X_val": trad_preprocessor.transform(X_val),
            "X_test": trad_preprocessor.transform(X_test),
        }

        # Save data
        with open(processed_dir / "traditional_ml_data.pkl", "wb") as f:
            pickle.dump({"features": trad_features, "preprocessor": trad_preprocessor}, f)

        preprocessing_results["traditional_ml"] = trad_features["X_train"].shape
        print(f"Success: {trad_features['X_train'].shape} features")

    except Exception as e:
        print(f"Failed: {e}")
        preprocessing_results["traditional_ml"] = "Failed"

    # Deep learning preprocessing
    print("\n4. Deep learning preprocessing...")
    try:
        dl_preprocessor = DeepLearningPreprocessor(
            max_length=256,  # Increase from 128 for better context
            vocab_size=15000,  # Increase vocabulary size
        )
        dl_preprocessor.fit(X_train, use_language="both")

        dl_sequences = {
            "X_train": dl_preprocessor.transform(X_train, use_language="both"),
            "X_val": dl_preprocessor.transform(X_val, use_language="both"),
            "X_test": dl_preprocessor.transform(X_test, use_language="both"),
        }

        # Save data
        with open(processed_dir / "deep_learning_data.pkl", "wb") as f:
            pickle.dump({"sequences": dl_sequences, "preprocessor": dl_preprocessor}, f)

        preprocessing_results["deep_learning"] = dl_sequences["X_train"].shape
        print(f"Success: {dl_sequences['X_train'].shape} sequences")

    except Exception as e:
        print(f"Failed: {e}")
        preprocessing_results["deep_learning"] = "Failed"

    # Transformer preprocessing
    for model_key, model_name in TRANSFORMER_MODELS.items():
        print(f"\n5. {model_key.upper()} preprocessing...")
        try:
            trans_preprocessor = TransformerPreprocessor(
                model_name=model_name,
                max_length=512,  # Increase max length for better context
            )
            trans_preprocessor.fit(X_train)

            use_language = "both" if "xlm" in model_name.lower() else "translated"
            trans_encodings = {
                "X_train": trans_preprocessor.transform(X_train, use_language=use_language),
                "X_val": trans_preprocessor.transform(X_val, use_language=use_language),
                "X_test": trans_preprocessor.transform(X_test, use_language=use_language),
                "use_language": use_language,
            }

            # Save data
            with open(processed_dir / f"{model_key}_data.pkl", "wb") as f:
                pickle.dump({"encodings": trans_encodings, "preprocessor": trans_preprocessor}, f)

            preprocessing_results[model_key] = trans_encodings["X_train"]["input_ids"].shape
            print(f"Success: {trans_encodings['X_train']['input_ids'].shape} tokens")

        except Exception as e:
            print(f"Failed: {e}")
            preprocessing_results[model_key] = "Failed"

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    for data_type, result in preprocessing_results.items():
        status = "SUCCESS" if result != "Failed" else "FAILED"
        print(f"{data_type}: {result} - {status}")

    successful_types = [k for k, v in preprocessing_results.items() if v != "Failed"]
    print(f"\nReady for model types: {', '.join(successful_types)}")
    print(f"Preprocessed data cached in: {processed_dir}")

    return len(successful_types) > 0


def main():
    """Main function - setup and preprocessing"""
    print("=" * 60)
    print("EMOTION CLASSIFICATION PROJECT")
    print("=" * 60)

    # Run preprocessing
    success = preprocess_all_data()

    if success:
        print("\nNext steps:")
        print("1. Train models: python models/logistic_regression.py")
        print(
            "2. View results: python -c \"import pandas as pd; print(pd.read_csv('model_iterations.csv'))\""
        )
        print("3. Compare models: from src.trainer import compare_models")
    else:
        print("\nPreprocessing failed. Check error messages above.")


if __name__ == "__main__":
    main()
