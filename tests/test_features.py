import numpy as np
import pandas as pd

from deepfmp_dg.features import (
    FEATURE_COLUMNS,
    build_features,
    extract_text_features,
    minmax_rank,
    select_features,
)


def test_extract_text_features_basic():
    f = extract_text_features("Great quality! Not worth it.")
    assert f["review_pos_word_count"] == 1
    assert f["review_neg_word_count"] == 1
    assert f["review_sentiment_diff"] == 0
    assert f["review_exclamation_count"] == 1


def test_extract_text_features_non_string():
    f = extract_text_features(None)
    assert f["review_word_count"] == 0
    assert f["review_sentiment_diff"] == 0


def test_minmax_rank_range():
    s = pd.Series([10.0, 20.0, 30.0])
    r = minmax_rank(s)
    assert r.min() > 0 and r.max() <= 1.0
    assert len(r) == 3


def test_build_features_creates_all_columns():
    df = pd.DataFrame(
        {
            "review_text": ["Great dress, love it!", "Terrible quality, broke in a day"],
            "title": ["Summer Dress", "Winter Coat"],
            "delta_cosine": [0.3, 0.7],
            "delta_euclidean": [5.0, 9.0],
            "price": [25.0, 80.0],
        }
    )
    out = build_features(df)
    for col in FEATURE_COLUMNS:
        assert col in out.columns
    assert out["delta_x_price_rank"].iloc[0] == 0.3 * out["P_rank"].iloc[0]


def test_build_features_is_idempotent():
    df = pd.DataFrame(
        {
            "review_text": ["Great dress, love it!", "Terrible quality, broke in a day"],
            "title": ["Summer Dress", "Winter Coat"],
            "delta_cosine": [0.3, 0.7],
            "delta_euclidean": [5.0, 9.0],
            "price": [25.0, 80.0],
        }
    )
    once = build_features(df)
    twice = build_features(once)
    assert len(twice.columns) == len(once.columns)
    np.testing.assert_allclose(select_features(once), select_features(twice), atol=1e-6)


def test_select_features_fills_missing():
    df = pd.DataFrame(
        {
            "review_text": ["ok"],
            "title": ["T"],
            "delta_cosine": [0.2],
            "delta_euclidean": [1.0],
            "price": [10.0],
        }
    )
    out = build_features(df)
    X = select_features(out)
    assert X.shape == (1, len(FEATURE_COLUMNS))
    assert np.all(np.isfinite(X))


def test_extract_text_features_chinese():
    f = extract_text_features("这个东西太差了，垃圾，给差评")
    assert f["review_neg_word_count"] >= 2
    assert f["review_sentiment_diff"] < 0
