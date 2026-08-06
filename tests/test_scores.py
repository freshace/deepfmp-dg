import numpy as np
import pytest

from deepfmp_dg.scores import (
    build_merchant_text,
    clean_review_text,
    compute_alignment_features,
    l2_normalize,
    pair_scores,
)


def test_l2_normalize_has_unit_norm():
    x = np.array([[3.0, 4.0], [1.0, 0.0]])
    y = l2_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(y, axis=1), [1.0, 1.0], atol=1e-6)


def test_zero_vector_does_not_nan():
    y = l2_normalize(np.zeros((1, 4)))
    assert np.all(np.isfinite(y))


def test_compute_alignment_features_shape_and_range():
    rng = np.random.default_rng(0)
    n = 5
    seller = rng.normal(size=(n, 8))
    buyer = rng.normal(size=(n, 8))
    merchant = rng.normal(size=(n, 8))
    review = rng.normal(size=(n, 8))
    feats = compute_alignment_features(seller, buyer, merchant, review)
    assert feats.shape == (n, 3)
    assert feats.dtype == np.float32
    assert np.all(feats >= -1.0 - 1e-5) and np.all(feats <= 1.0 + 1e-5)


def test_pair_scores_matches_compute_alignment_features():
    rng = np.random.default_rng(1)
    s = rng.normal(size=8)
    b = rng.normal(size=8)
    m = rng.normal(size=8)
    r = rng.normal(size=8)
    scores = pair_scores(s, b, m, r)
    feats = compute_alignment_features(
        s.reshape(1, -1), b.reshape(1, -1), m.reshape(1, -1), r.reshape(1, -1)
    )[0]
    assert scores["score1"] == pytest.approx(feats[0])
    assert scores["score2"] == pytest.approx(feats[1])
    assert scores["score3"] == pytest.approx(feats[2])


def test_build_merchant_text_handles_empty_description():
    assert build_merchant_text("Great Dress", "[]") == "Great Dress"
    assert build_merchant_text("Great Dress", "Nice fabric") == "Great Dress. Nice fabric"
    assert build_merchant_text(None, "Nice fabric") == "Nice fabric"
    assert build_merchant_text(None, None) is None


def test_clean_review_text():
    assert clean_review_text("Hi") is None
    assert clean_review_text("&#34;hi&#34; there") == '"hi" there'

