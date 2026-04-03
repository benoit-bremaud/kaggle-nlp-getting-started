"""Text preprocessing utilities for NLP disaster tweets."""

import re
from urllib.parse import unquote

import pandas as pd


def clean_text(text: str) -> str:
    """Clean a single text string for NLP processing.

    Lowercase, remove URLs, HTML entities, special characters,
    and collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def add_text_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Add text-derived columns to a dataframe.

    Adds: text_clean, text_len, word_count, keyword_clean.
    Handles missing keywords by filling with empty string and URL-decoding.
    """
    df = df.copy()
    # Extract mention/hashtag counts from raw text (before cleaning strips them)
    raw_text = df[text_col].fillna("")
    df["mention_count"] = raw_text.str.count(r"@\w+")
    df["hashtag_count"] = raw_text.str.count(r"#\w+")
    df["text_clean"] = raw_text.apply(clean_text)
    df["text_len"] = df["text_clean"].str.len()
    df["word_count"] = df["text_clean"].str.split().str.len().fillna(0).astype(int)
    df["keyword_clean"] = df["keyword"].fillna("").apply(lambda x: unquote(str(x)))
    return df
