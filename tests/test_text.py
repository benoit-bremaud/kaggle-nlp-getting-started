"""Tests for src/text.py — text preprocessing utilities."""

import pandas as pd

from src.text import add_text_features, clean_text


def test_clean_text_removes_urls():
    assert clean_text("check https://t.co/xyz out") == "check out"
    assert clean_text("visit www.example.com now") == "visit now"


def test_clean_text_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_clean_text_removes_special_chars():
    assert clean_text("fire!!! #disaster") == "fire disaster"


def test_add_text_features_columns():
    df = pd.DataFrame({"text": ["hello world", "test tweet"], "keyword": ["fire", "flood"]})
    result = add_text_features(df)
    expected_cols = {"text_clean", "text_len", "word_count", "keyword_clean"}
    assert expected_cols.issubset(set(result.columns))
    assert result["text_len"].iloc[0] == len("hello world")
    assert result["word_count"].iloc[0] == 2


def test_add_text_features_handles_missing_keyword():
    df = pd.DataFrame({"text": ["hello", "world"], "keyword": [None, "storm"]})
    result = add_text_features(df)
    assert result["keyword_clean"].iloc[0] == ""
    assert result["keyword_clean"].iloc[1] == "storm"


def test_add_text_features_decodes_url_encoded_keywords():
    df = pd.DataFrame({"text": ["test"], "keyword": ["oil%20spill"]})
    result = add_text_features(df)
    assert result["keyword_clean"].iloc[0] == "oil spill"
