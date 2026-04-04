"""Tests for src/features.py — feature engineering pipeline."""

import pandas as pd
import scipy.sparse as sp

from src.features import (
    CharTfidfTransformer,
    KeywordEncoder,
    NumericFeatures,
    TextTfidfTransformer,
    build_feature_matrix,
)


def _sample_df() -> pd.DataFrame:
    """Create a small sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "text_clean": ["hello world", "disaster flood", "nice sunny day"],
            "keyword_clean": ["fire", "flood", "weather"],
            "text_len": [11, 15, 14],
            "word_count": [2, 2, 3],
            "mention_count": [0, 1, 0],
            "hashtag_count": [1, 0, 2],
        }
    )


def test_tfidf_transformer_fit_transform():
    df = _sample_df()
    tfidf = TextTfidfTransformer(max_features=100)
    tfidf.fit(df)
    result = tfidf.transform(df)
    assert sp.issparse(result)
    assert result.shape[0] == 3


def test_tfidf_transformer_unseen_data():
    """Transform on new data should not fail, even with unseen words."""
    train = _sample_df()
    test = pd.DataFrame({"text_clean": ["unknown words here"]})
    tfidf = TextTfidfTransformer(max_features=100)
    tfidf.fit(train)
    result = tfidf.transform(test)
    assert result.shape[0] == 1


def test_keyword_encoder_fit_transform():
    df = _sample_df()
    encoder = KeywordEncoder()
    encoder.fit(df)
    result = encoder.transform(df)
    assert sp.issparse(result)
    assert result.shape[0] == 3


def test_keyword_encoder_unseen_keyword():
    """Unseen keywords should produce zero vectors, not errors."""
    train = _sample_df()
    test = pd.DataFrame({"keyword_clean": ["earthquake"]})
    encoder = KeywordEncoder()
    encoder.fit(train)
    result = encoder.transform(test)
    assert result.shape[0] == 1
    assert result.nnz == 0  # All zeros — unseen keyword


def test_keyword_encoder_multiword_keyword():
    """Multi-word keywords (e.g. 'oil spill') should be one feature, not split."""
    df = pd.DataFrame({"keyword_clean": ["oil spill", "forest fire", "oil spill"]})
    encoder = KeywordEncoder()
    encoder.fit(df)
    vocab = encoder.vectorizer_.vocabulary_
    assert "oil spill" in vocab
    assert "oil" not in vocab  # Should NOT be split


def test_numeric_features():
    df = _sample_df()
    numeric = NumericFeatures()
    result = numeric.transform(df)
    assert sp.issparse(result)
    assert result.shape == (3, 4)


def test_char_tfidf_transformer_fit_transform():
    df = _sample_df()
    char_tfidf = CharTfidfTransformer(max_features=100)
    char_tfidf.fit(df)
    result = char_tfidf.transform(df)
    assert sp.issparse(result)
    assert result.shape[0] == 3


def test_char_tfidf_transformer_unseen_data():
    train = _sample_df()
    test = pd.DataFrame({"text_clean": ["zzzzz unknown"]})
    char_tfidf = CharTfidfTransformer(max_features=100)
    char_tfidf.fit(train)
    result = char_tfidf.transform(test)
    assert result.shape[0] == 1


def test_build_feature_matrix():
    train = _sample_df()
    test = pd.DataFrame(
        {
            "text_clean": ["new tweet"],
            "keyword_clean": ["fire"],
            "text_len": [9],
            "word_count": [2],
            "mention_count": [0],
            "hashtag_count": [1],
        }
    )
    x_train, x_test, *_ = build_feature_matrix(train, test, max_features=100, char_max_features=100)
    assert sp.issparse(x_train)
    assert sp.issparse(x_test)
    assert x_train.shape[0] == 3
    assert x_test.shape[0] == 1
    assert x_train.shape[1] == x_test.shape[1]
