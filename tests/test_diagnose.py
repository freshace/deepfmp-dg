
from deepfmp_dg.diagnose import (
    build_recommendations,
    diagnose_item,
    estimate_return_rate,
    identify_root_causes,
    risk_level,
)


def test_risk_level_boundaries():
    assert risk_level(0.2) == "极低"
    assert risk_level(0.2001) == "低"
    assert risk_level(0.4) == "低"
    assert risk_level(0.41) == "中"
    assert risk_level(0.6) == "中"
    assert risk_level(0.61) == "高"
    assert risk_level(0.8) == "高"
    assert risk_level(0.81) == "极高"


def test_estimate_return_rate_formula():
    value = estimate_return_rate(0.5, 2.0, 0.4, -0.5, 0.9)
    expected = 0.35 * 0.5 + 0.25 * (1 - 2.0 / 5) + 0.20 * 0.4 + 0.15 * (1 - 0.5 / 2) + 0.05 * 0.9
    assert abs(value - expected) < 1e-6


def test_identify_root_causes_visual_gap():
    scores = {"score1": 0.3, "score2": 0.2, "score3": 0.6}
    causes = identify_root_causes(scores, [("review_sentiment_diff", 0.1)])
    assert any("视觉" in c for c in causes)


def test_diagnose_item_output_shape():
    scores = {"score1": 0.3, "score2": 0.1, "score3": 0.5}
    top = [("review_sentiment_diff", 0.12), ("P_rank", 0.08)]
    out = diagnose_item(scores, 0.87, top)
    assert out["risk_level"] == "极高"
    assert isinstance(out["diagnosis"], list) and out["diagnosis"]
    assert isinstance(out["recommendations"], list) and out["recommendations"]


def test_build_recommendations_high_risk():
    recs = build_recommendations("高", ["货不对板型（视觉差异显著）"])
    assert any(r.startswith("P0") for r in recs)
