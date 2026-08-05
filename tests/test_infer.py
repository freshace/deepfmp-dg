import json
import numpy as np
import joblib
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from deepfmp_dg.features import FEATURE_COLUMNS
from deepfmp_dg.infer import InferenceEngine
from deepfmp_dg.models import create_model
from deepfmp_dg.train import save_checkpoint


class FakeEncoder:
    def __init__(self, dim=8):
        self.rng = np.random.default_rng(0)
        self.dim = dim

    def encode_image(self, path):
        return self.rng.normal(size=self.dim).astype(np.float32)

    def encode_text(self, text):
        return self.rng.normal(size=self.dim).astype(np.float32)


def _make_model_dir(tmp_path, dim=8):
    torch.manual_seed(0)
    model = create_model("direct", emb_dim=dim, n_tabular=16, n_align=3)
    save_checkpoint(model, "direct", tmp_path / "deepfmp_direct.pt")

    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 16))
    y = np.array([0, 1] * 25)
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    rf.fit(x, y)
    scaler = StandardScaler().fit(x)
    joblib.dump(rf, tmp_path / "rf.joblib")
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    (tmp_path / "feature_names.json").write_text(json.dumps(FEATURE_COLUMNS), encoding="utf-8")
    return tmp_path


def test_predict_returns_structured_json(tmp_path):
    engine = InferenceEngine(_make_model_dir(tmp_path), encoder=FakeEncoder(), emb_dim=8)
    result = engine.predict(
        "seller.jpg", "buyer.jpg",
        "Summer Dress", "Lightweight cotton dress",
        "Great quality, fits perfectly!", 25.0,
    )
    assert set(result["scores"].keys()) == {"score1", "score2", "score3"}
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["risk_level"] in ["极低", "低", "中", "高", "极高"]
    assert isinstance(result["diagnosis"], list)
    assert isinstance(result["recommendations"], list)
    assert result["top_shap"], "top_shap should not be empty"
    assert "display_name" in result["top_shap"][0]


def test_predict_requires_valid_review(tmp_path):
    engine = InferenceEngine(_make_model_dir(tmp_path), encoder=FakeEncoder(), emb_dim=8)
    try:
        engine.predict("s.jpg", "b.jpg", "T", "D", "bad", 10.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
