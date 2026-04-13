"""Feature engineering pipeline for NLP disaster tweets.

All transformers are sklearn Pipeline-compatible (ADR-020).
Fit on train only, transform both train and test (ADR-019).

This module provides two feature extraction approaches:
1. TF-IDF (sparse, bag-of-words) — fast, interpretable, baseline approach
2. BERT embeddings (dense, semantic) — captures meaning, pre-trained on billions of words

Both can be used independently or combined.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoModel, AutoTokenizer


class TextTfidfTransformer(BaseEstimator, TransformerMixin):
    """TF-IDF vectorizer that reads from a DataFrame column.

    Pipeline-compatible: fit() learns vocabulary from train,
    transform() applies to any DataFrame with the same column.
    """

    def __init__(self, text_col: str = "text_clean", max_features: int = 5000):
        self.text_col = text_col
        self.max_features = max_features

    def fit(self, X: pd.DataFrame, y=None):
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.vectorizer_.fit(X[self.text_col])
        return self

    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        return self.vectorizer_.transform(X[self.text_col])


class CharTfidfTransformer(BaseEstimator, TransformerMixin):
    """Character-level TF-IDF vectorizer.

    Captures spelling/style patterns (ALL CAPS, abbreviations) that word-level misses.
    Pipeline-compatible: fit() learns char n-grams from train only.
    """

    def __init__(self, text_col: str = "text_clean", max_features: int = 5000):
        self.text_col = text_col
        self.max_features = max_features

    def fit(self, X: pd.DataFrame, y=None):
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=self.max_features,
        )
        self.vectorizer_.fit(X[self.text_col])
        return self

    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        return self.vectorizer_.transform(X[self.text_col])


class BertEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Extract dense embeddings from a pre-trained BERT model.

    How BERT works (simplified):
    ─────────────────────────────
    BERT is a neural network pre-trained on massive text corpora (Wikipedia + books).
    It learned to understand language by predicting masked words in sentences.
    As a side effect, it can convert any text into a dense vector (embedding)
    that captures the *meaning* of the text, not just which words appear.

    Why this matters for our task:
    ─────────────────────────────
    TF-IDF treats "car crash" and "vehicle accident" as completely different
    (different words → different dimensions). BERT understands they mean the same
    thing and produces similar embeddings for both.

    What this class does:
    ─────────────────────
    1. Tokenizes each tweet into sub-word tokens (BERT's vocabulary)
    2. Passes tokens through the frozen (non-trainable) BERT model
    3. Extracts the [CLS] token embedding — a 768-dimensional vector that
       summarizes the entire tweet's meaning
    4. Returns a dense numpy array of shape (n_samples, 768)

    We use DistilBERT (a distilled version of BERT):
    - 2x faster, 40% smaller than BERT
    - Retains 97% of BERT's language understanding
    - Perfect for feature extraction on a small dataset

    Pipeline-compatible: fit() loads the model, transform() extracts embeddings.
    No training happens here — the model is frozen (pre-trained weights only).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        text_col: str = "text_clean",
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.text_col = text_col
        self.batch_size = batch_size
        # Auto-detect GPU: use CUDA if available, otherwise CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: pd.DataFrame, y=None):
        """Load the pre-trained model and tokenizer.

        No "learning" happens here — BERT is already trained.
        We just load it into memory so transform() can use it.
        """
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModel.from_pretrained(self.model_name)
        self.model_.to(self.device)
        # Freeze the model — we don't want to modify its weights
        self.model_.eval()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Convert texts to dense embeddings using BERT.

        Process:
        1. Split texts into batches (to fit in GPU memory)
        2. Tokenize each batch (convert words → token IDs that BERT understands)
        3. Pass through BERT model (forward pass, no gradient computation)
        4. Extract the [CLS] token output — the first token's hidden state,
           which BERT is trained to use as a sentence-level summary
        5. Return all embeddings stacked into a (n_samples, 768) numpy array
        """
        texts = X[self.text_col].tolist()
        all_embeddings = []

        # Process in batches to avoid running out of GPU memory
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            # Tokenize: convert text → token IDs + attention mask
            # - padding=True: pad shorter texts to match the longest in the batch
            # - truncation=True: cut texts longer than 512 tokens (BERT's max)
            # - return_tensors="pt": return PyTorch tensors (not lists)
            encoded = self.tokenizer_(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,  # tweets are short, 128 tokens is plenty
                return_tensors="pt",
            )
            # Move tensors to the same device as the model (GPU or CPU)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Forward pass through BERT — no gradient computation needed
            # torch.no_grad() saves memory and speeds up inference
            with torch.no_grad():
                outputs = self.model_(**encoded)

            # Extract [CLS] token embedding (first token of each sequence)
            # outputs.last_hidden_state shape: (batch_size, seq_len, 768)
            # We take [:, 0, :] = first token of each sequence = [CLS] = (batch_size, 768)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        # Stack all batches into a single array: (n_samples, 768)
        return np.vstack(all_embeddings)


class KeywordEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode the keyword_clean column.

    Pipeline-compatible: fit() learns vocabulary from train only.
    """

    def __init__(self, keyword_col: str = "keyword_clean"):
        self.keyword_col = keyword_col

    def fit(self, X: pd.DataFrame, y=None):
        # WHY tokenizer + token_pattern=None: treat each keyword as a single token,
        # not split on whitespace. Prevents "oil spill" from becoming two separate features.
        self.vectorizer_ = CountVectorizer(binary=True, tokenizer=lambda x: [x], token_pattern=None)
        self.vectorizer_.fit(X[self.keyword_col])
        return self

    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        return self.vectorizer_.transform(X[self.keyword_col])


class NumericFeatures(BaseEstimator, TransformerMixin):
    """Extract numeric columns (text_len, word_count, mention_count, hashtag_count) as sparse matrix."""

    def __init__(self, cols: list[str] | None = None):
        self.cols = cols or ["text_len", "word_count", "mention_count", "hashtag_count"]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        return sp.csr_matrix(X[self.cols].values.astype(np.float64))


def build_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 5000,
    char_max_features: int = 5000,
) -> tuple[
    sp.csr_matrix,
    sp.csr_matrix,
    TextTfidfTransformer,
    CharTfidfTransformer,
    KeywordEncoder,
    NumericFeatures,
]:
    """Build sparse feature matrices. Fit on train only (ADR-019).

    Returns (X_train, X_test, tfidf, char_tfidf, keyword_encoder, numeric_features).
    """
    tfidf = TextTfidfTransformer(max_features=max_features)
    char_tfidf = CharTfidfTransformer(max_features=char_max_features)
    keyword = KeywordEncoder()
    numeric = NumericFeatures()

    # Fit on train only
    tfidf.fit(train_df)
    char_tfidf.fit(train_df)
    keyword.fit(train_df)
    numeric.fit(train_df)

    # Transform both
    transformers = [tfidf, char_tfidf, keyword, numeric]
    x_train = sp.hstack([t.transform(train_df) for t in transformers], format="csr")
    x_test = sp.hstack([t.transform(test_df) for t in transformers], format="csr")

    return x_train, x_test, tfidf, char_tfidf, keyword, numeric


def build_bert_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, BertEmbeddingTransformer, KeywordEncoder, NumericFeatures]:
    """Build dense feature matrices using BERT embeddings. Fit on train only (ADR-019).

    How this differs from build_feature_matrix():
    ──────────────────────────────────────────────
    - TF-IDF (sparse, 8226 dims) is replaced by BERT embeddings (dense, 768 dims)
    - BERT captures semantic meaning; TF-IDF captures word frequency
    - The output is a dense numpy array, not a sparse scipy matrix
    - Keyword and numeric features are still included (they add structured signal)

    Returns (X_train, X_test, bert_transformer, keyword_encoder, numeric_features).
    """
    bert = BertEmbeddingTransformer(model_name=model_name, batch_size=batch_size)
    keyword = KeywordEncoder()
    numeric = NumericFeatures()

    # Fit on train only (ADR-019)
    bert.fit(train_df)
    keyword.fit(train_df)
    numeric.fit(train_df)

    # Transform both sets
    # BERT → dense (N, 768), keyword → sparse, numeric → sparse
    # Convert sparse to dense for horizontal stacking
    x_train = np.hstack(
        [
            bert.transform(train_df),
            keyword.transform(train_df).toarray(),
            numeric.transform(train_df).toarray(),
        ]
    )
    x_test = np.hstack(
        [
            bert.transform(test_df),
            keyword.transform(test_df).toarray(),
            numeric.transform(test_df).toarray(),
        ]
    )

    return x_train, x_test, bert, keyword, numeric
