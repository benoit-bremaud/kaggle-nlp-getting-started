"""Feature engineering pipeline for NLP disaster tweets.

All transformers are sklearn Pipeline-compatible (ADR-020).
Fit on train only, transform both train and test (ADR-019).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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


class KeywordEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode the keyword_clean column.

    Pipeline-compatible: fit() learns vocabulary from train only.
    """

    def __init__(self, keyword_col: str = "keyword_clean"):
        self.keyword_col = keyword_col

    def fit(self, X: pd.DataFrame, y=None):
        self.vectorizer_ = CountVectorizer(binary=True)
        self.vectorizer_.fit(X[self.keyword_col])
        return self

    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        return self.vectorizer_.transform(X[self.keyword_col])


class NumericFeatures(BaseEstimator, TransformerMixin):
    """Extract numeric columns (text_len, word_count) as sparse matrix."""

    def __init__(self, cols: list[str] | None = None):
        self.cols = cols or ["text_len", "word_count"]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        return sp.csr_matrix(X[self.cols].values.astype(np.float64))


def build_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 5000,
) -> tuple[sp.csr_matrix, sp.csr_matrix, TextTfidfTransformer, KeywordEncoder, NumericFeatures]:
    """Build sparse feature matrices. Fit on train only (ADR-019).

    Returns (X_train, X_test, tfidf_transformer, keyword_encoder, numeric_features).
    """
    tfidf = TextTfidfTransformer(max_features=max_features)
    keyword = KeywordEncoder()
    numeric = NumericFeatures()

    # Fit on train only
    tfidf.fit(train_df)
    keyword.fit(train_df)
    numeric.fit(train_df)

    # Transform both
    x_train = sp.hstack(
        [tfidf.transform(train_df), keyword.transform(train_df), numeric.transform(train_df)], format="csr"
    )
    x_test = sp.hstack([tfidf.transform(test_df), keyword.transform(test_df), numeric.transform(test_df)], format="csr")

    return x_train, x_test, tfidf, keyword, numeric
