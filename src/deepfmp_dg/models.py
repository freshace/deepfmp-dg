"""DeepFMP-DG network definitions.

The architecture fuses a dual visual encoder, three expectation-gap
alignment scores, and tabular features through attention and gating.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        b, d = x_a.shape
        q = self.q_proj(x_a).view(b, self.num_heads, self.head_dim)
        k = self.k_proj(x_b).view(b, self.num_heads, self.head_dim)
        v = self.v_proj(x_b).view(b, self.num_heads, self.head_dim)
        attn = (q * k).sum(dim=-1) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * v).reshape(b, d)
        out = self.out_proj(out)
        return self.norm1(x_a + out)


class DualEncoderWithAttention(nn.Module):
    def __init__(self, emb_dim: int = 768, proj_dim: int = 128, dropout: float = 0.4) -> None:
        super().__init__()
        self.encoder_a = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(),
        )
        self.encoder_b = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(),
        )
        self.cross_attn_ab = CrossModalAttention(proj_dim, num_heads=4, dropout=dropout)
        self.cross_attn_ba = CrossModalAttention(proj_dim, num_heads=4, dropout=dropout)
        self.diff_proj = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
        )
        self.hadamard_proj = nn.Sequential(
            nn.Linear(proj_dim, 64), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.out_dim = proj_dim * 2 + 128 + 64

    def encode(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        enc_a = self.encoder_a(emb_a)
        enc_b = self.encoder_b(emb_b)
        enc_a_attended = self.cross_attn_ab(enc_a, enc_b)
        enc_b_attended = self.cross_attn_ba(enc_b, enc_a)
        diff = self.diff_proj(emb_a - emb_b)
        hadamard = self.hadamard_proj(enc_a_attended * enc_b_attended)
        return torch.cat([enc_a_attended, enc_b_attended, diff, hadamard], dim=-1)


class VisualOnlyNetwork(nn.Module):
    def __init__(self, emb_dim: int = 768, proj_dim: int = 128, dropout: float = 0.4) -> None:
        super().__init__()
        self.visual_net = DualEncoderWithAttention(emb_dim, proj_dim, dropout)
        fused_dim = self.visual_net.out_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(proj_dim // 2, 2),
        )

    def forward(self, seller_img: torch.Tensor, buyer_img: torch.Tensor) -> torch.Tensor:
        return self.head(self.visual_net.encode(seller_img, buyer_img))


class VisualTabNetwork(nn.Module):
    def __init__(
        self, emb_dim: int = 768, n_tabular: int = 16, proj_dim: int = 128, dropout: float = 0.4
    ) -> None:
        super().__init__()
        self.visual_net = DualEncoderWithAttention(emb_dim, proj_dim, dropout)
        self.tabular_proj = nn.Sequential(
            nn.Linear(n_tabular, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Dropout(dropout),
        )
        fused_dim = self.visual_net.out_dim + proj_dim
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, proj_dim), nn.Tanh(), nn.Linear(proj_dim, fused_dim), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(proj_dim // 2, 2),
        )

    def forward(self, seller_img: torch.Tensor, buyer_img: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        visual_feat = self.visual_net.encode(seller_img, buyer_img)
        tab_feat = self.tabular_proj(tabular)
        combined = torch.cat([visual_feat, tab_feat], dim=-1)
        gated = combined * self.gate(combined)
        return self.head(gated)


class AlignDirectNetwork(nn.Module):
    def __init__(
        self, emb_dim: int = 768, n_tabular: int = 16, n_align: int = 3,
        proj_dim: int = 128, dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.visual_net = DualEncoderWithAttention(emb_dim, proj_dim, dropout)
        self.align_proj = nn.Sequential(
            nn.Linear(n_align, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
        )
        self.tabular_proj = nn.Sequential(
            nn.Linear(n_tabular, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Dropout(dropout),
        )
        fused_dim = self.visual_net.out_dim + 32 + proj_dim
        self.fusion_attn = nn.Sequential(
            nn.Linear(fused_dim, proj_dim), nn.Tanh(), nn.Linear(proj_dim, fused_dim), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(proj_dim // 2, 2),
        )

    def forward(
        self, seller_img: torch.Tensor, buyer_img: torch.Tensor,
        align_feats: torch.Tensor, tabular: torch.Tensor,
    ) -> torch.Tensor:
        visual_feat = self.visual_net.encode(seller_img, buyer_img)
        align_feat = self.align_proj(align_feats)
        tab_feat = self.tabular_proj(tabular)
        combined = torch.cat([visual_feat, align_feat, tab_feat], dim=-1)
        fused = combined * self.fusion_attn(combined)
        return self.head(fused)


class SoftmaxGateNetwork(nn.Module):
    def __init__(
        self, emb_dim: int = 768, n_tabular: int = 16, n_align: int = 3,
        proj_dim: int = 128, dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.visual_net = DualEncoderWithAttention(emb_dim, proj_dim, dropout)
        self.gate_mlp = nn.Sequential(
            nn.Linear(n_align, 16), nn.GELU(),
            nn.Linear(16, n_align),
        )
        self.align_proj = nn.Sequential(
            nn.Linear(n_align, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
        )
        self.tabular_proj = nn.Sequential(
            nn.Linear(n_tabular, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Dropout(dropout),
        )
        fused_dim = self.visual_net.out_dim + 32 + proj_dim
        self.fusion_attn = nn.Sequential(
            nn.Linear(fused_dim, proj_dim), nn.Tanh(), nn.Linear(proj_dim, fused_dim), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(proj_dim // 2, 2),
        )

    def forward(
        self, seller_img: torch.Tensor, buyer_img: torch.Tensor,
        align_feats: torch.Tensor, tabular: torch.Tensor,
    ) -> torch.Tensor:
        visual_feat = self.visual_net.encode(seller_img, buyer_img)
        gate_logits = self.gate_mlp(align_feats)
        gate_weights = F.softmax(gate_logits, dim=-1)
        weighted_feats = align_feats * gate_weights
        align_feat = self.align_proj(weighted_feats)
        tab_feat = self.tabular_proj(tabular)
        combined = torch.cat([visual_feat, align_feat, tab_feat], dim=-1)
        fused = combined * self.fusion_attn(combined)
        return self.head(fused)

    def get_gate_weights(self, align_feats: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            gate_logits = self.gate_mlp(align_feats)
            return F.softmax(gate_logits, dim=-1)


MODEL_TYPES = {
    "visual": VisualOnlyNetwork,
    "visual_tab": VisualTabNetwork,
    "direct": AlignDirectNetwork,
    "softmax_gate": SoftmaxGateNetwork,
}


def create_model(
    model_type: str, emb_dim: int = 768, n_tabular: int = 16,
    n_align: int = 3, dropout: float = 0.4,
) -> nn.Module:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type: {model_type}")
    if model_type == "visual":
        return VisualOnlyNetwork(emb_dim=emb_dim, dropout=dropout)
    if model_type == "visual_tab":
        return VisualTabNetwork(emb_dim=emb_dim, n_tabular=n_tabular, dropout=dropout)
    return MODEL_TYPES[model_type](
        emb_dim=emb_dim, n_tabular=n_tabular, n_align=n_align, dropout=dropout
    )

