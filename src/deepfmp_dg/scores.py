"""SigLIP encoders and the three expectation-gap scores.

Score1 (image-image): seller display image vs buyer real photo.
Score2 (text-image): merchant description vs buyer real photo.
Score3 (text-text): merchant description vs buyer review text.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
EMBEDDING_DIM = 768

PathLike = str | Path


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with a numerical floor."""
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _extract_embedding(outputs) -> np.ndarray:
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        vec = outputs.pooler_output
    elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        vec = outputs.image_embeds
    elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        vec = outputs.text_embeds
    elif hasattr(outputs, "last_hidden_state"):
        vec = outputs.last_hidden_state.mean(dim=1)
    else:
        raise ValueError("Unsupported model output type")
    return np.asarray(vec.detach().cpu().numpy()).reshape(-1)


class SigLIPEncoder:
    """Lazy-loads the SigLIP model and encodes images and short texts."""

    def __init__(self, model_id: str = SIGLIP_MODEL_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._image_processor = None
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._image_processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self._model.eval()

    def encode_image(self, path: PathLike) -> np.ndarray:
        self._ensure_loaded()
        image = Image.open(path).convert("RGB")
        inputs = self._image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = (
                self._model.get_image_features(**inputs)
                if hasattr(self._model, "get_image_features")
                else self._model(**inputs)
            )
        return _extract_embedding(outputs)

    def encode_text(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        inputs = self._tokenizer(
            text=text, return_tensors="pt", padding=True, truncation=True, max_length=64
        ).to(self.device)
        with torch.no_grad():
            outputs = (
                self._model.get_text_features(**inputs)
                if hasattr(self._model, "get_text_features")
                else self._model(**inputs)
            )
        return _extract_embedding(outputs)


def build_merchant_text(title: str | None, description: str | None) -> str | None:
    title_txt = str(title or "").strip()
    desc_txt = str(description or "").strip()
    if desc_txt in ("[]", "", "nan", "None"):
        return title_txt or None
    return f"{title_txt}. {desc_txt}" if title_txt else desc_txt


def clean_review_text(text: str | None) -> str | None:
    if not isinstance(text, str) or len(text.strip()) < 5:
        return None
    return text.replace("&#34;", '"').replace("&amp;", "&").strip()[:512]


def compute_alignment_features(
    seller_emb: np.ndarray,
    buyer_emb: np.ndarray,
    merchant_emb: np.ndarray,
    review_emb: np.ndarray,
) -> np.ndarray:
    """Return an (N, 3) array of [score1, score2, score3] cosine similarities."""
    s = l2_normalize(seller_emb)
    b = l2_normalize(buyer_emb)
    m = l2_normalize(merchant_emb)
    r = l2_normalize(review_emb)
    score1 = np.sum(s * b, axis=1, keepdims=True)
    score2 = np.sum(m * b, axis=1, keepdims=True)
    score3 = np.sum(m * r, axis=1, keepdims=True)
    return np.hstack([score1, score2, score3]).astype(np.float32)


def pair_scores(
    seller_emb: np.ndarray,
    buyer_emb: np.ndarray,
    merchant_emb: np.ndarray,
    review_emb: np.ndarray,
) -> dict:
    """Single-vector convenience wrapper returning named scores."""
    feats = compute_alignment_features(
        seller_emb.reshape(1, -1),
        buyer_emb.reshape(1, -1),
        merchant_emb.reshape(1, -1),
        review_emb.reshape(1, -1),
    )[0]
    return {"score1": float(feats[0]), "score2": float(feats[1]), "score3": float(feats[2])}

