# ============================================================
#  TikTok Virality Prediction — CS 506
#  Unit tests for core pipeline functions
#  Matches exact logic used in baseline.ipynb and transformer.ipynb
#  Run: pytest tests/ -v
# ============================================================

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder


# ── Helper ────────────────────────────────────────────────────

def make_sample_df(n=200):
    """Synthetic DataFrame matching exact columns used in both notebooks."""
    np.random.seed(42)
    return pd.DataFrame({
        "video_view_count":           np.random.randint(100, 5_000_000, n),
        "video_like_count":           np.random.randint(0, 500_000, n),
        "video_share_count":          np.random.randint(0, 100_000, n),
        "video_comment_count":        np.random.randint(0, 50_000, n),
        "video_download_count":       np.random.randint(0, 20_000, n),
        "video_duration_sec":         np.random.randint(5, 60, n),
        "video_transcription_text":   [f"sample tiktok caption {i}" for i in range(n)],
        "claim_status":               np.random.choice(["claim", "opinion"], n),
        "verified_status":            np.random.choice(["verified", "not verified"], n),
        "author_ban_status":          np.random.choice(["active", "under review", "banned"], n),
    })


# ── Test 1: Virality label (both notebooks use same logic) ────

def test_viral_label_is_binary():
    """Label must only contain 0 and 1."""
    df = make_sample_df()
    threshold = df["video_view_count"].median()
    df["is_viral"] = (df["video_view_count"] > threshold).astype(int)
    assert set(df["is_viral"].unique()).issubset({0, 1})


def test_viral_label_balanced():
    """Median split produces ~50/50 class distribution."""
    df = make_sample_df()
    threshold = df["video_view_count"].median()
    df["is_viral"] = (df["video_view_count"] > threshold).astype(int)
    viral_rate = df["is_viral"].mean()
    assert 0.3 <= viral_rate <= 0.7, f"Imbalanced: {viral_rate:.2f}"


def test_above_median_is_viral():
    """Every video above median view count must have label = 1."""
    df = make_sample_df()
    threshold = df["video_view_count"].median()
    df["is_viral"] = (df["video_view_count"] > threshold).astype(int)
    assert (df.loc[df["video_view_count"] > threshold, "is_viral"] == 1).all()


# ── Test 2: Data cleaning (baseline.ipynb Cell 3) ─────────────

def test_numeric_coercion():
    """
    Matches baseline.ipynb Cell 3:
    pd.to_numeric(errors='coerce').fillna(0)
    """
    df = make_sample_df()
    num_cols = ["video_view_count", "video_like_count", "video_share_count",
                "video_download_count", "video_comment_count", "video_duration_sec"]
    # Simulate string values that need coercion
    df["video_like_count"] = df["video_like_count"].astype(str)
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in num_cols:
        assert df[col].isnull().sum() == 0
        assert df[col].dtype in [np.float64, np.int64]


def test_label_encoder_no_nulls():
    """
    Matches baseline.ipynb Cell 3 and transformer.ipynb Cell 10:
    LabelEncoder on verified_status and author_ban_status
    """
    df = make_sample_df()
    le = LabelEncoder()
    for col in ["verified_status", "author_ban_status"]:
        df[col] = le.fit_transform(df[col].astype(str))
    assert df["verified_status"].isnull().sum() == 0
    assert df["author_ban_status"].isnull().sum() == 0
    assert df["verified_status"].nunique() <= 2
    assert df["author_ban_status"].nunique() <= 3


# ── Test 3: Transformer-specific cleaning ────────────────────

def test_missing_transcription_text_filled():
    """
    Matches transformer.ipynb Cell 8:
    df['video_transcription_text'].fillna('').astype(str)
    """
    df = make_sample_df()
    df.loc[[0, 5, 10], "video_transcription_text"] = None
    df["video_transcription_text"] = (
        df["video_transcription_text"].fillna("").astype(str)
    )
    assert df["video_transcription_text"].isnull().sum() == 0
    assert df["video_transcription_text"].iloc[0] == ""


def test_tokenizer_input_is_string():
    """
    DistilBERT tokenizer requires all inputs to be strings.
    Matches transformer.ipynb Dataset __getitem__ logic.
    """
    df = make_sample_df()
    df["video_transcription_text"] = df["video_transcription_text"].astype(object)
    df.loc[0, "video_transcription_text"] = None
    df.loc[1, "video_transcription_text"] = 999
    df["video_transcription_text"] = (
        df["video_transcription_text"].fillna("").astype(str)
    )
    assert df["video_transcription_text"].apply(type).eq(str).all()


def test_numeric_features_finite_after_standardization():
    """
    Matches transformer.ipynb Cell 9:
    (df[col] - mean) / (std + 1e-8)
    No NaN or inf values after standardization.
    """
    df = make_sample_df()
    df["video_duration_sec"] = pd.to_numeric(
        df["video_duration_sec"], errors="coerce"
    ).fillna(0)
    mean = df["video_duration_sec"].mean()
    std  = df["video_duration_sec"].std()
    df["video_duration_sec"] = (df["video_duration_sec"] - mean) / (std + 1e-8)
    assert np.isfinite(df["video_duration_sec"].values).all()
    assert df["video_duration_sec"].isnull().sum() == 0
