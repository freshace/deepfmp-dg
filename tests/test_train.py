import numpy as np
import torch

from deepfmp_dg.train import run_experiment, set_seed, train_model


def _make_data(n=64, dim=8):
    rng = np.random.default_rng(42)
    seller = rng.normal(size=(n, dim)).astype(np.float32)
    buyer = rng.normal(size=(n, dim)).astype(np.float32)
    align = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    tab = rng.normal(size=(n, 4)).astype(np.float32)
    y = np.array([0, 1] * (n // 2))
    return seller, buyer, align, tab, y


def test_set_seed_reproducible():
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    torch.testing.assert_close(a, b)


def test_train_model_runs_and_returns_prob():
    seller, buyer, align, tab, y = _make_data()
    train_idx = np.arange(48)
    val_idx = np.arange(48, 64)
    prob, best_auc, _info, state, _model = train_model(
        "direct", seller, buyer, align, tab, y, train_idx, val_idx,
        epochs=3, patience=2,
    )
    assert prob.shape == (16,)
    assert np.all((prob >= 0) & (prob <= 1))
    assert best_auc >= 0.0
    assert state is not None


def test_run_experiment_returns_metrics():
    seller, buyer, align, tab, y = _make_data()
    res = run_experiment(
        "DeepFMP (S1+S2+S3, direct)", "direct",
        seller, buyer, align, tab, y, n_splits=2, epochs=2, patience=1,
    )
    for key in ["model", "n", "positive", "auc_cv", "ap_cv", "f1_cv",
                "precision_cv", "recall_cv", "accuracy_cv", "y", "prob"]:
        assert key in res
    assert res["n"] == 64

