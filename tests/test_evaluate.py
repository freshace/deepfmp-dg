import numpy as np

from deepfmp_dg.evaluate import compute_metrics, safe_ap, safe_auc


def test_safe_auc_known_value():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert safe_auc(y, p) == 0.75


def test_safe_auc_single_class_returns_nan():
    assert np.isnan(safe_auc(np.zeros(4), np.zeros(4)))


def test_safe_ap_single_class_returns_nan():
    assert np.isnan(safe_ap(np.zeros(4), np.zeros(4)))


def test_compute_metrics_keys():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = compute_metrics(y, p)
    for key in ["auc", "ap", "f1", "precision", "recall", "accuracy"]:
        assert key in m
    assert m["auc"] == 0.75
