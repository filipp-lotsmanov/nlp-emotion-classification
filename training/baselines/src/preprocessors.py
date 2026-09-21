# """
# Model-specific preprocessing classes (FIXED COLUMN NAMES)
# """
# import pandas as pd
# import numpy as np
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.preprocessing import StandardScaler
# import pickle
# from pathlib import Path
#
# try:
#     from transformers import AutoTokenizer
#     TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     TRANSFORMERS_AVAILABLE = False
#     print("Warning: transformers library not available. Transformer preprocessing will not work.")
#
# try:
#     import torch
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False
#     print("Warning: torch library not available. Deep learning preprocessing may have issues.")
#
# from src.config import MAX_FEATURES_TFIDF, MAX_SEQUENCE_LENGTH, DATA_DIR
#
# class TraditionalMLPreprocessor:
#     """Preprocessor for traditional ML models (LogReg, NB, SVM)"""
#
#     def __init__(self, use_tfidf=True, ngram_range=(1, 2), max_features=MAX_FEATURES_TFIDF):
#         self.use_tfidf = use_tfidf
#         self.ngram_range = ngram_range
#         self.max_features = max_features
#         self.original_vectorizer = None
#         self.translated_vectorizer = None
#         self.scaler = StandardScaler()
#         self.fitted = False
#
#     def clean_text(self, text):
#         """Basic text cleaning"""
#         if pd.isna(text):
#             return ""
#         # Convert to lowercase and remove extra whitespace
#         text = str(text).lower().strip()
#         # Remove extra spaces
#         text = re.sub(r'\s+', ' ', text)
#         return text
#
#     def fit(self, X_train):
#         """Fit the preprocessor on training data"""
#         print("Fitting traditional ML preprocessor...")
#
#         # Clean texts - UPDATED COLUMN NAMES
#         original_texts = X_train['Original_Text'].apply(self.clean_text)
#         translated_texts = X_train['Translated_Text'].apply(self.clean_text)
#
#         # Initialize vectorizers
#         if self.use_tfidf:
#             self.original_vectorizer = TfidfVectorizer(
#                 ngram_range=self.ngram_range,
#                 max_features=self.max_features,
#                 stop_words=None  # Keep all words for emotion detection
#             )
#             self.translated_vectorizer = TfidfVectorizer(
#                 ngram_range=self.ngram_range,
#                 max_features=self.max_features,
#                 stop_words='english'
#             )
#         else:
#             # For Naive Bayes - use count vectors
#             self.original_vectorizer = CountVectorizer(
#                 ngram_range=self.ngram_range,
#                 max_features=self.max_features,
#                 binary=True  # Binary features work well with NB
#             )
#             self.translated_vectorizer = CountVectorizer(
#                 ngram_range=self.ngram_range,
#                 max_features=self.max_features,
#                 stop_words='english',
#                 binary=True
#             )
#
#         # Fit vectorizers
#         self.original_vectorizer.fit(original_texts)
#         self.translated_vectorizer.fit(translated_texts)
#
#         # Fit scaler on confidence scores
#         confidence_scores = X_train['Confidence'].values.reshape(-1, 1)
#         self.scaler.fit(confidence_scores)
#
#         self.fitted = True
#         print("Traditional ML preprocessor fitted successfully!")
#
#     def transform(self, X):
#         """Transform data to feature vectors"""
#         if not self.fitted:
#             raise ValueError("Preprocessor must be fitted before transform")
#
#         # Clean texts - UPDATED COLUMN NAMES
#         original_texts = X['Original_Text'].apply(self.clean_text)
#         translated_texts = X['Translated_Text'].apply(self.clean_text)
#
#         # Vectorize texts
#         original_features = self.original_vectorizer.transform(original_texts)
#         translated_features = self.translated_vectorizer.transform(translated_texts)
#
#         # Scale confidence scores
#         confidence_features = self.scaler.transform(X['Confidence'].values.reshape(-1, 1))
#
#         # Additional features - UPDATED COLUMN NAMES
#         original_lengths = X['Original_Text'].apply(len).values.reshape(-1, 1)
#         translated_lengths = X['Translated_Text'].apply(len).values.reshape(-1, 1)
#
#         # Combine all features
#         from scipy.sparse import hstack, csr_matrix
#
#         features = hstack([
#             original_features,
#             translated_features,
#             csr_matrix(confidence_features),
#             csr_matrix(original_lengths),
#             csr_matrix(translated_lengths)
#         ])
#
#         print(f"Transformed to feature matrix of shape: {features.shape}")
#         return features
#
# class DeepLearningPreprocessor:
#     """Preprocessor for deep learning models (RNN, LSTM)"""
#
#     def __init__(self, max_length=MAX_SEQUENCE_LENGTH, vocab_size=10000):
#         self.max_length = max_length
#         self.vocab_size = vocab_size
#         self.word_to_idx = {}
#         self.idx_to_word = {}
#         self.fitted = False
#
#     def tokenize_text(self, text):
#         """Simple tokenization"""
#         if pd.isna(text):
#             return []
#         text = str(text).lower()
#         # Simple tokenization - extract words
#         tokens = re.findall(r'\b\w+\b', text)
#         return tokens
#
#     def build_vocabulary(self, texts):
#         """Build vocabulary from training texts"""
#         word_counts = {}
#
#         for text in texts:
#             tokens = self.tokenize_text(text)
#             for token in tokens:
#                 word_counts[token] = word_counts.get(token, 0) + 1
#
#         # Sort by frequency and take top vocab_size
#         sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
#
#         # Add special tokens
#         self.word_to_idx = {
#             '<PAD>': 0,
#             '<UNK>': 1,
#             '<START>': 2,
#             '<END>': 3
#         }
#
#         # Add most frequent words
#         for word, count in sorted_words[:self.vocab_size - 4]:
#             if word not in self.word_to_idx:
#                 self.word_to_idx[word] = len(self.word_to_idx)
#
#         # Create reverse mapping
#         self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
#
#         print(f"Vocabulary size: {len(self.word_to_idx)}")
#
#     def text_to_sequence(self, text):
#         """Convert text to sequence of token indices"""
#         tokens = self.tokenize_text(text)
#         sequence = [self.word_to_idx.get(token, 1) for token in tokens]  # 1 is <UNK>
#
#         # Truncate or pad
#         if len(sequence) > self.max_length:
#             sequence = sequence[:self.max_length]
#         else:
#             sequence = sequence + [0] * (self.max_length - len(sequence))  # 0 is <PAD>
#
#         return sequence
#
#     def fit(self, X_train, use_language='both'):
#         """Fit the preprocessor on training data"""
#         print("Fitting deep learning preprocessor...")
#
#         texts = []
#         if use_language in ['original', 'both']:
#             texts.extend(X_train['Original_Text'].dropna().tolist())
#         if use_language in ['translated', 'both']:
#             texts.extend(X_train['Translated_Text'].dropna().tolist())
#
#         self.build_vocabulary(texts)
#         self.fitted = True
#         print("Deep learning preprocessor fitted successfully!")
#
#     def transform(self, X, use_language='both'):
#         """Transform texts to sequences"""
#         if not self.fitted:
#             raise ValueError("Preprocessor must be fitted before transform")
#
#         if use_language == 'both':
#             # Concatenate Original and Translated texts - UPDATED COLUMN NAMES
#             combined_texts = X['Original_Text'].astype(str) + " " + X['Translated_Text'].astype(str)
#             sequences = [self.text_to_sequence(text) for text in combined_texts]
#         elif use_language == 'original':
#             sequences = [self.text_to_sequence(text) for text in X['Original_Text']]
#         elif use_language == 'translated':
#             sequences = [self.text_to_sequence(text) for text in X['Translated_Text']]
#         else:
#             raise ValueError("use_language must be 'original', 'translated', or 'both'")
#
#         sequences_array = np.array(sequences)
#         print(f"Transformed to sequence array of shape: {sequences_array.shape}")
#         return sequences_array
#
# class TransformerPreprocessor:
#     """Preprocessor for transformer models (BERT, XLM-RoBERTa)"""
#
#     def __init__(self, model_name='bert-base-uncased', max_length=256):
#         if not TRANSFORMERS_AVAILABLE:
#             raise ImportError("transformers library is required for TransformerPreprocessor")
#
#         self.model_name = model_name
#         self.max_length = max_length
#         self.tokenizer = None
#         self.fitted = False
#
#     def fit(self, X_train):
#         """Initialize tokenizer"""
#         print(f"Loading tokenizer for {self.model_name}...")
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             self.fitted = True
#             print(f"Tokenizer loaded successfully for {self.model_name}")
#         except Exception as e:
#             print(f"Error loading tokenizer: {e}")
#             print("This might happen if you don't have internet connection or the model name is incorrect")
#             raise
#
#     def transform(self, X, use_language='both'):
#         """Transform texts using tokenizer"""
#         if not self.fitted:
#             raise ValueError("Preprocessor must be fitted before transform")
#
#         # Determine which text to use - UPDATED COLUMN NAMES
#         if use_language == 'both':
#             if 'xlm' in self.model_name.lower():
#                 # XLM-RoBERTa can handle both languages together
#                 texts = X['Original_Text'].astype(str) + " " + X['Translated_Text'].astype(str)
#             else:
#                 # For BERT, use Translated (English) only
#                 texts = X['Translated_Text'].astype(str)
#         elif use_language == 'original':
#             texts = X['Original_Text'].astype(str)
#         elif use_language == 'translated':
#             texts = X['Translated_Text'].astype(str)
#         else:
#             raise ValueError("use_language must be 'original', 'translated', or 'both'")
#
#         print(f"Tokenizing {len(texts)} texts...")
#
#         # Tokenize texts
#         encodings = self.tokenizer(
#             texts.tolist(),
#             truncation=True,
#             padding=True,
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
#
#         print(f"Tokenization complete. Input shape: {encodings['input_ids'].shape}")
#         return encodings
#
# def get_preprocessor(model_type, model_name=None, **kwargs):
#     """Factory function to get appropriate preprocessor"""
#
#     if model_type == 'traditional_ml':
#         return TraditionalMLPreprocessor(**kwargs)
#     elif model_type == 'deep_learning':
#         return DeepLearningPreprocessor(**kwargs)
#     elif model_type == 'transformer':
#         if model_name is None:
#             raise ValueError("model_name required for transformer preprocessor")
#         return TransformerPreprocessor(model_name=model_name, **kwargs)
#     else:
#         raise ValueError(f"Unknown model_type: {model_type}")
#
# def test_preprocessors():
#     """Test all preprocessors with sample data"""
#     print("Testing preprocessors with sample data...")
#
#     # Create sample data with CORRECT COLUMN NAMES
#     sample_data = pd.DataFrame({
#         'Original_Text': ['Я очень счастлив сегодня', 'Мне грустно и одиноко'],
#         'Translated_Text': ['I am very happy today', 'I feel sad and lonely'],
#         'Confidence': [0.85, 0.72]
#     })
#
#     print("Sample data created:")
#     print(sample_data)
#
#     # Test Traditional ML preprocessor
#     print("\n" + "="*50)
#     print("1. Testing Traditional ML Preprocessor:")
#     try:
#         trad_preprocessor = TraditionalMLPreprocessor()
#         trad_preprocessor.fit(sample_data)
#         trad_features = trad_preprocessor.transform(sample_data)
#         print(f"✅ Traditional ML features shape: {trad_features.shape}")
#     except Exception as e:
#         print(f"❌ Traditional ML test failed: {e}")
#
#     # Test Deep Learning preprocessor
#     print("\n" + "="*50)
#     print("2. Testing Deep Learning Preprocessor:")
#     try:
#         dl_preprocessor = DeepLearningPreprocessor(max_length=50, vocab_size=1000)
#         dl_preprocessor.fit(sample_data)
#         dl_sequences = dl_preprocessor.transform(sample_data)
#         print(f"✅ Deep learning sequences shape: {dl_sequences.shape}")
#     except Exception as e:
#         print(f"❌ Deep learning test failed: {e}")
#
#     # Test Transformer preprocessor
#     print("\n" + "="*50)
#     print("3. Testing Transformer Preprocessor:")
#     if not TRANSFORMERS_AVAILABLE:
#         print("❌ Transformers library not available, skipping test")
#     else:
#         try:
#             transformer_preprocessor = TransformerPreprocessor('bert-base-uncased')
#             transformer_preprocessor.fit(sample_data)
#             transformer_encodings = transformer_preprocessor.transform(sample_data)
#             print(f"✅ Transformer encodings keys: {list(transformer_encodings.keys())}")
#             print(f"✅ Input IDs shape: {transformer_encodings['input_ids'].shape}")
#         except Exception as e:
#             print(f"❌ Transformer test failed: {e}")
#             print("This is normal if you don't have internet connection or transformers not installed")
#
#     print("\n" + "="*50)
#     print("Preprocessor tests completed!")
#
# if __name__ == "__main__":
#     # Run tests
#     test_preprocessors()

"""
Model-specific preprocessing classes (OPTIMIZED)
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.config import MAX_FEATURES_TFIDF, MAX_SEQUENCE_LENGTH


class TraditionalMLPreprocessor:
    """Optimized preprocessor for traditional ML models"""

    def __init__(self, use_tfidf=True, ngram_range=(1, 3), max_features=15000):
        self.use_tfidf = use_tfidf
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.original_vectorizer = None
        self.translated_vectorizer = None
        self.scaler = StandardScaler()
        self.fitted = False

    def clean_text(self, text):
        """Enhanced text cleaning"""
        if pd.isna(text):
            return ""

        text = str(text).lower().strip()
        # Remove extra spaces and normalize
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep emotion indicators
        text = re.sub(r"[^\w\s!?.]", " ", text)
        # Preserve multiple exclamations/questions (emotion indicators)
        text = re.sub(r"!{2,}", "!!", text)
        text = re.sub(r"\?{2,}", "??", text)

        return text.strip()

    def fit(self, X_train):
        """Fit preprocessor with enhanced feature extraction"""
        print("Fitting traditional ML preprocessor...")

        # Clean texts
        original_texts = X_train["Original_Text"].apply(self.clean_text)
        translated_texts = X_train["Translated_Text"].apply(self.clean_text)

        # Enhanced vectorizer settings for small dataset
        vectorizer_params = {
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "min_df": 1,  # Keep rare words (small dataset)
            "max_df": 0.95,  # Remove very common words
            "sublinear_tf": True,  # Apply sublinear TF scaling
        }

        if self.use_tfidf:
            self.original_vectorizer = TfidfVectorizer(
                stop_words=None,  # Keep all words for emotion detection
                **vectorizer_params,
            )
            self.translated_vectorizer = TfidfVectorizer(stop_words="english", **vectorizer_params)
        else:
            # For Naive Bayes
            vectorizer_params.pop("sublinear_tf")  # Not applicable to CountVectorizer
            self.original_vectorizer = CountVectorizer(binary=True, **vectorizer_params)
            self.translated_vectorizer = CountVectorizer(
                stop_words="english", binary=True, **vectorizer_params
            )

        # Fit vectorizers
        self.original_vectorizer.fit(original_texts)
        self.translated_vectorizer.fit(translated_texts)

        # Fit scaler
        confidence_scores = X_train["Confidence"].values.reshape(-1, 1)
        self.scaler.fit(confidence_scores)

        self.fitted = True
        print("Traditional ML preprocessor fitted successfully!")

    def transform(self, X):
        """Transform with enhanced features"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Clean texts
        original_texts = X["Original_Text"].apply(self.clean_text)
        translated_texts = X["Translated_Text"].apply(self.clean_text)

        # Vectorize texts
        original_features = self.original_vectorizer.transform(original_texts)
        translated_features = self.translated_vectorizer.transform(translated_texts)

        # Enhanced features
        confidence_features = self.scaler.transform(X["Confidence"].values.reshape(-1, 1))
        original_lengths = X["Original_Text"].str.len().values.reshape(-1, 1)
        translated_lengths = X["Translated_Text"].str.len().values.reshape(-1, 1)

        # Additional features for emotion detection
        exclamation_counts = X["Original_Text"].str.count("!").values.reshape(-1, 1)
        question_counts = X["Original_Text"].str.count(r"\?").values.reshape(-1, 1)
        caps_ratio = (
            X["Original_Text"].str.count("[A-Z]") / X["Original_Text"].str.len().fillna(1)
        ).values.reshape(-1, 1)

        # Combine all features
        from scipy.sparse import hstack, csr_matrix

        features = hstack(
            [
                original_features,
                translated_features,
                csr_matrix(confidence_features),
                csr_matrix(original_lengths),
                csr_matrix(translated_lengths),
                csr_matrix(exclamation_counts),
                csr_matrix(question_counts),
                csr_matrix(caps_ratio),
            ]
        )

        return features


class DeepLearningPreprocessor:
    """Optimized preprocessor for deep learning models"""

    def __init__(self, max_length=256, vocab_size=15000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.fitted = False

    def enhanced_tokenize(self, text):
        """Enhanced tokenization preserving emotion indicators"""
        if pd.isna(text):
            return []

        text = str(text).lower()
        # Keep emotion indicators
        text = re.sub(r"!{2,}", " EXCLAMATION_MULTIPLE ", text)
        text = re.sub(r"\?{2,}", " QUESTION_MULTIPLE ", text)
        text = re.sub(r"!", " EXCLAMATION ", text)
        text = re.sub(r"\?", " QUESTION ", text)

        # Extract words and special tokens
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def build_vocabulary(self, texts):
        """Build vocabulary with emotion-specific tokens"""
        word_counts = {}

        for text in texts:
            tokens = self.enhanced_tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Enhanced special tokens for emotions
        self.word_to_idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3,
            "EXCLAMATION": 4,
            "QUESTION": 5,
            "EXCLAMATION_MULTIPLE": 6,
            "QUESTION_MULTIPLE": 7,
        }

        # Add most frequent words
        start_idx = len(self.word_to_idx)
        for word, count in sorted_words[: self.vocab_size - start_idx]:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)

        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        print(f"Enhanced vocabulary size: {len(self.word_to_idx)}")

    def text_to_sequence(self, text):
        """Convert text to sequence with emotion preservation"""
        tokens = self.enhanced_tokenize(text)
        sequence = [self.word_to_idx.get(token, 1) for token in tokens]

        # Truncate or pad
        if len(sequence) > self.max_length:
            sequence = sequence[: self.max_length]
        else:
            sequence = sequence + [0] * (self.max_length - len(sequence))

        return sequence

    def fit(self, X_train, use_language="both"):
        """Fit with enhanced text processing"""
        print("Fitting deep learning preprocessor...")

        texts = []
        if use_language in ["original", "both"]:
            texts.extend(X_train["Original_Text"].dropna().tolist())
        if use_language in ["translated", "both"]:
            texts.extend(X_train["Translated_Text"].dropna().tolist())

        self.build_vocabulary(texts)
        self.fitted = True
        print("Deep learning preprocessor fitted successfully!")

    def transform(self, X, use_language="both"):
        """Transform with enhanced sequence processing"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        if use_language == "both":
            combined_texts = X["Original_Text"].astype(str) + " " + X["Translated_Text"].astype(str)
            sequences = [self.text_to_sequence(text) for text in combined_texts]
        elif use_language == "original":
            sequences = [self.text_to_sequence(text) for text in X["Original_Text"]]
        elif use_language == "translated":
            sequences = [self.text_to_sequence(text) for text in X["Translated_Text"]]
        else:
            raise ValueError("use_language must be 'original', 'translated', or 'both'")

        return np.array(sequences)


class TransformerPreprocessor:
    """Optimized preprocessor for transformer models"""

    def __init__(self, model_name="bert-base-uncased", max_length=512):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.fitted = False

    def fit(self, X_train):
        """Initialize tokenizer with optimized settings"""
        print(f"Loading tokenizer for {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.fitted = True
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

    def transform(self, X, use_language="both"):
        """Transform with optimized tokenization"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Determine text strategy
        if use_language == "both":
            if "xlm" in self.model_name.lower():
                # XLM-RoBERTa: concatenate with separator
                texts = (
                    X["Original_Text"].astype(str) + " [SEP] " + X["Translated_Text"].astype(str)
                )
            else:
                # BERT: use translated text only
                texts = X["Translated_Text"].astype(str)
        elif use_language == "original":
            texts = X["Original_Text"].astype(str)
        elif use_language == "translated":
            texts = X["Translated_Text"].astype(str)
        else:
            raise ValueError("Invalid use_language parameter")

        # Optimized tokenization
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        return encodings


def get_preprocessor(model_type, model_name=None, **kwargs):
    """Factory function for preprocessors"""
    if model_type == "traditional_ml":
        return TraditionalMLPreprocessor(**kwargs)
    elif model_type == "deep_learning":
        return DeepLearningPreprocessor(**kwargs)
    elif model_type == "transformer":
        if model_name is None:
            raise ValueError("model_name required for transformer preprocessor")
        return TransformerPreprocessor(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    # Test preprocessors
    print("Testing optimized preprocessors...")

    sample_data = pd.DataFrame(
        {
            "Original_Text": ["Text sample 1", "Text sample 2"],
            "Translated_Text": ["Sample translation 1", "Sample translation 2"],
            "Confidence": [0.85, 0.72],
        }
    )

    try:
        trad_preprocessor = TraditionalMLPreprocessor()
        trad_preprocessor.fit(sample_data)
        features = trad_preprocessor.transform(sample_data)
        print(f"Traditional ML test successful: {features.shape}")
    except Exception as e:
        print(f"Traditional ML test failed: {e}")

    print("Preprocessor optimization completed!")
