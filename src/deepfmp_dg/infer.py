"""End-to-end single-item inference pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

from deepfmp_dg.diagnose import diagnose_item
from deepfmp_dg.explain import FEATURE_NAME_MAP, top_shap_features, tree_shap_values
from deepfmp_dg.features import build_features, select_features
from deepfmp_dg.models import create_model
from deepfmp_dg.scores import (
    SigLIPEncoder,
    build_merchant_text,
    clean_review_text,
    pair_scores,
)
from deepfmp_dg.train import load_checkpoint

PathLike = str | Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "sample" / "models"


class InferenceEngine:
    """Loads the packaged models and diagnoses one item end-to-end."""

    def __init__(
        self,
        model_dir: PathLike = DEFAULT_MODEL_DIR,
        encoder: Optional[SigLIPEncoder] = None,
        emb_dim: int = 768,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.encoder = encoder or SigLIPEncoder()
        self.emb_dim = emb_dim

        checkpoint = torch.load(self.model_dir / "deepfmp_direct.pt", map_location="cpu")
        self.model = create_model(
            checkpoint["model_type"], emb_dim=emb_dim, n_tabular=16, n_align=3
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.rf = joblib.load(self.model_dir / "rf.joblib")
        self.scaler = joblib.load(self.model_dir / "scaler.joblib")
        self.feature_names = list(
            json.loads((self.model_dir / "feature_names.json").read_text(encoding="utf-8"))
        )

    def predict(
        self,
        seller_image: PathLike,
        buyer_image: PathLike,
        title: str,
        description: str,
        review_text: str,
        price: Optional[float] = None,
    ) -> dict:
        merchant_text = build_merchant_text(title, description)
        if merchant_text is None:
            raise ValueError("Title/description missing: cannot build merchant text")
        review = clean_review_text(review_text)
        if review is None:
            raise ValueError("Review text too short or missing")

        seller_emb = self.encoder.encode_image(seller_image)
        buyer_emb = self.encoder.encode_image(buyer_image)
        merchant_emb = self.encoder.encode_text(merchant_text)
        review_emb = self.encoder.encode_text(review)
        scores = pair_scores(seller_emb, buyer_emb, merchant_emb, review_emb)

        delta_cosine = 1.0 - scores["score1"]
        delta_euclidean = float(np.linalg.norm(seller_emb - buyer_emb))
        row = pd.DataFrame(
            [{
                "review_text": review,
                "title": title or "",
                "delta_cosine": delta_cosine,
                "delta_euclidean": delta_euclidean,
                "price": price if price is not None else np.nan,
            }]
        )
        x = select_features(build_features(row)).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        align = np.array([[scores["score1"], scores["score2"], scores["score3"]]], dtype=np.float32)
        with torch.no_grad():
            logits = self.model(
                torch.tensor(seller_emb, dtype=torch.float32).unsqueeze(0),
                torch.tensor(buyer_emb, dtype=torch.float32).unsqueeze(0),
                torch.tensor(align),
                torch.tensor(x_scaled, dtype=torch.float32),
            )
            prob = float(torch.softmax(logits, dim=1)[0, 1].item())

        shap_row = tree_shap_values(self.rf, x_scaled)[0]
        top = top_shap_features(shap_row, self.feature_names, k=5)
        diagnosis = diagnose_item(scores, prob, top)
        top_shap = [
            {
                "feature": name,
                "display_name": FEATURE_NAME_MAP.get(name, name),
                "value": value,
            }
            for name, value in top
        ]
        return {
            "scores": scores,
            **diagnosis,
            "top_shap": top_shap,
            "delta_cosine": delta_cosine,
            "delta_euclidean": delta_euclidean,
        }
