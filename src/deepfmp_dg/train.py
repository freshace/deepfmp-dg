"""Fixed-seed training with early stopping and cross-validation."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from deepfmp_dg.evaluate import safe_ap, safe_auc
from deepfmp_dg.models import create_model

PathLike = str | Path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model_type: str,
    seller_emb: np.ndarray,
    buyer_emb: np.ndarray,
    align_feats: np.ndarray,
    X_tabular: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int = 200,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,
    dropout: float = 0.4,
    patience: int = 30,
    n_align: int = 3,
):
    """Train one model on a fixed split; returns OOF-style probabilities on val_idx."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_dim = seller_emb.shape[1]

    s_train = torch.tensor(seller_emb[train_idx], dtype=torch.float32)
    b_train = torch.tensor(buyer_emb[train_idx], dtype=torch.float32)
    a_train = torch.tensor(align_feats[train_idx], dtype=torch.float32)
    tab_train = torch.tensor(X_tabular[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)

    s_val = torch.tensor(seller_emb[val_idx], dtype=torch.float32)
    b_val = torch.tensor(buyer_emb[val_idx], dtype=torch.float32)
    a_val = torch.tensor(align_feats[val_idx], dtype=torch.float32)
    tab_val = torch.tensor(X_tabular[val_idx], dtype=torch.float32)

    pos_w = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], device=device)
    criterion = nn.CrossEntropyLoss(weight=torch.cat([torch.ones(1), pos_w]).to(device))

    n_tab = X_tabular.shape[1]
    model = create_model(
        model_type, emb_dim=emb_dim, n_tabular=n_tab, n_align=n_align, dropout=dropout
    ).to(device)

    def build_dataset() -> TensorDataset:
        if model_type == "visual":
            return TensorDataset(s_train, b_train, y_train)
        if model_type == "visual_tab":
            return TensorDataset(s_train, b_train, tab_train, y_train)
        return TensorDataset(s_train, b_train, a_train, tab_train, y_train)

    def train_fwd(batch):
        if model_type == "visual":
            return model(batch[0].to(device), batch[1].to(device))
        if model_type == "visual_tab":
            return model(batch[0].to(device), batch[1].to(device), batch[2].to(device))
        return model(batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device))

    def val_fwd():
        if model_type == "visual":
            return model(s_val.to(device), b_val.to(device))
        if model_type == "visual_tab":
            return model(s_val.to(device), b_val.to(device), tab_val.to(device))
        return model(s_val.to(device), b_val.to(device), a_val.to(device), tab_val.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bs = min(64, len(train_idx))
    loader = DataLoader(build_dataset(), batch_size=bs, shuffle=True, drop_last=False)

    best_auc = 0.0
    best_state = None
    best_gate_weights = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            target = batch[-1].to(device)
            logits = train_fwd(batch)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = val_fwd()
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            val_auc = safe_auc(y[val_idx], prob)
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if hasattr(model, "get_gate_weights"):
                    best_gate_weights = model.get_gate_weights(a_val.to(device)).cpu().numpy()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = val_fwd()
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    info: dict = {}
    if best_gate_weights is not None:
        info["gate_weights"] = {
            "score1": float(np.mean(best_gate_weights[:, 0])),
            "score2": float(np.mean(best_gate_weights[:, 1])),
            "score3": float(np.mean(best_gate_weights[:, 2])),
        }
    return prob, best_auc, info, best_state, model


def run_experiment(
    model_name: str,
    model_type: str,
    seller_emb: np.ndarray,
    buyer_emb: np.ndarray,
    align_feats: np.ndarray,
    X_tabular: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_align: int = 3,
    epochs: int = 200,
    patience: int = 30,
) -> Optional[dict]:
    """Run stratified K-fold CV and return aggregated metrics."""
    min_class = min(np.sum(y == 0), np.sum(y == 1))
    n_splits = min(n_splits, min_class)
    if n_splits < 2 or len(y) < 30:
        return None

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_prob = np.zeros(len(y))
    all_infos: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(seller_emb, y)):
        prob, _, info, _, _ = train_model(
            model_type, seller_emb, buyer_emb, align_feats, X_tabular, y,
            train_idx, val_idx, n_align=n_align, epochs=epochs, patience=patience,
        )
        all_prob[val_idx] = prob
        all_infos.append(info)

    pred = (all_prob >= 0.5).astype(int)
    avg_info: dict = {}
    if all_infos and "gate_weights" in all_infos[0]:
        avg_info["gate_weights"] = {}
        for key in all_infos[0]["gate_weights"]:
            avg_info["gate_weights"][key] = float(np.mean(
                [info["gate_weights"][key] for info in all_infos if "gate_weights" in info]
            ))

    return {
        "model": model_name,
        "n": int(len(y)),
        "positive": int(y.sum()),
        "auc_cv": safe_auc(y, all_prob),
        "ap_cv": safe_ap(y, all_prob),
        "f1_cv": float(f1_score(y, pred, zero_division=0)),
        "precision_cv": float(precision_score(y, pred, zero_division=0)),
        "recall_cv": float(recall_score(y, pred, zero_division=0)),
        "accuracy_cv": float(accuracy_score(y, pred)),
        "y": y,
        "prob": all_prob,
        **avg_info,
    }


def save_checkpoint(model: nn.Module, model_type: str, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": model_type, "state_dict": model.state_dict()}, path)


def load_checkpoint(model: nn.Module, path: PathLike) -> str:
    """Load weights into ``model``; returns the stored model_type."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint["model_type"]
