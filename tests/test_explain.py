import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from deepfmp_dg.explain import FEATURE_NAME_MAP, gate_weights, top_shap_features, tree_shap_values
from deepfmp_dg.models import create_model


def test_tree_shap_values_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4))
    y = np.array([0, 1] * 15)
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    rf.fit(x, y)
    v = tree_shap_values(rf, x)
    assert v.shape == (30, 4)


def test_top_shap_features_returns_sorted():
    row = np.array([0.1, -0.8, 0.3])
    names = ["a", "b", "c"]
    top = top_shap_features(row, names, k=2)
    assert top == [("b", -0.8), ("c", 0.3)]


def test_gate_weights_keys_and_sum():
    torch.manual_seed(0)
    model = create_model("softmax_gate", emb_dim=8, n_tabular=4, n_align=3)
    align = torch.randn(5, 3)
    w = gate_weights(model, align)
    assert set(w.keys()) == {"score1", "score2", "score3"}
    assert abs(sum(w.values()) - 1.0) < 1e-5


def test_feature_name_map_covers_core_features():
    for f in ["review_sentiment_diff", "delta_cosine", "P_rank"]:
        assert f in FEATURE_NAME_MAP
