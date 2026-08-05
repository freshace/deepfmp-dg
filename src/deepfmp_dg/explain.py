"""Model interpretability: TreeSHAP attribution and gate weights."""
from __future__ import annotations

import numpy as np
import torch

import shap

FEATURE_NAME_MAP = {
    "review_sentiment_diff": "评论情感差异",
    "review_exclamation_count": "感叹号数量",
    "review_pos_word_count": "正面词汇数量",
    "delta_x_negation": "预期差×否定词",
    "review_neg_word_count": "负面词汇数量",
    "review_word_count": "评论字数",
    "review_upper_ratio": "大写比例",
    "title_word_count": "标题字数",
    "delta_x_review_length": "预期差×评论长度",
    "log_review_length": "评论长度(log)",
    "delta_cosine": "视觉预期差",
    "delta_euclidean": "视觉距离差",
    "P_rank": "价格分位",
    "delta_x_price_rank": "预期差×价格分位",
    "img_kb_diff": "图片大小差",
    "img_kb_ratio": "图片大小比",
}


def tree_shap_values(model, X_scaled: np.ndarray) -> np.ndarray:
    """Positive-class SHAP values with shape (N, F)."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_scaled)
    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]
    return np.asarray(values)


def top_shap_features(
    shap_row: np.ndarray, feature_names: list[str], k: int = 5
) -> list[tuple[str, float]]:
    """Top-k features by |SHAP|, preserving the signed contribution."""
    order = np.argsort(np.abs(shap_row))[::-1][:k]
    return [(feature_names[i], float(shap_row[i])) for i in order]


def gate_weights(model, align_feats: torch.Tensor) -> dict[str, float]:
    """Mean Softmax gate weights for the three alignment scores."""
    if not hasattr(model, "get_gate_weights"):
        return {}
    w = model.get_gate_weights(align_feats)
    return {
        "score1": float(w[:, 0].mean()),
        "score2": float(w[:, 1].mean()),
        "score3": float(w[:, 2].mean()),
    }
