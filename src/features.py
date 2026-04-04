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
