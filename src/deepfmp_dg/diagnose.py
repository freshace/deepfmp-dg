"""Item-level risk grading, root-cause identification, and prescriptions."""
from __future__ import annotations

import numpy as np

RISK_LEVELS = ["极低", "低", "中", "高", "极高"]
RISK_THRESHOLDS = [0.2, 0.4, 0.6, 0.8]

RISK_CAUSE_PREFIXES = ("货不对板", "夸大宣传", "信息不对称", "价格错配", "体验不佳")

RETURN_RATE_WEIGHTS = {
    "negative": 0.35,
    "rating": 0.25,
    "visual": 0.20,
    "sentiment": 0.15,
    "model": 0.05,
}


def risk_level(prob: float) -> str:
    """Map a risk probability to one of five levels."""
    for level, threshold in zip(RISK_LEVELS, RISK_THRESHOLDS + [1.01]):
        if prob <= threshold:
            return level
    return RISK_LEVELS[-1]


def estimate_return_rate(
    negative_review_ratio: float,
    avg_rating: float,
    avg_delta_cosine: float,
    avg_sentiment: float,
    avg_model_prob: float,
) -> float:
    """Industry-style heuristic return-rate estimate in [0, 1]."""
    value = (
        RETURN_RATE_WEIGHTS["negative"] * negative_review_ratio
        + RETURN_RATE_WEIGHTS["rating"] * (1 - avg_rating / 5)
        + RETURN_RATE_WEIGHTS["visual"] * float(np.clip(avg_delta_cosine, 0, 1))
        + RETURN_RATE_WEIGHTS["sentiment"] * (1 - (avg_sentiment + 1) / 2)
        + RETURN_RATE_WEIGHTS["model"] * avg_model_prob
    )
    return float(np.clip(value, 0, 1))


def identify_root_causes(
    scores: dict, top_shap: list[tuple[str, float]]
) -> list[str]:
    """Rule-based root-cause mapping; thresholds are documented heuristics."""
    causes: list[str] = []
    score1 = scores.get("score1", 0.5)
    score2 = scores.get("score2", 0.0)
    score3 = scores.get("score3", 0.5)

    if 1 - score1 > 0.45:
        causes.append("货不对板型（视觉差异显著）")
    if score2 < 0.0:
        causes.append("夸大宣传型（描述与实拍不一致）")
    if score3 < 0.3:
        causes.append("信息不对称型（描述与评论不一致）")

    for name, value in top_shap:
        if name == "P_rank" and value > 0:
            causes.append("价格错配型（价格与体验不匹配）")
            break
    for name, value in top_shap:
        if name == "review_sentiment_diff" and value > 0:
            causes.append("体验不佳型（评论情感显著负面）")
            break

    return causes or ["信息不足型（需人工复核）"]


EXPLICIT_NEGATIVE_MARKERS = (
    "差评", "一星", "1星", "给一星", "退货", "退款", "再也不买",
    "垃圾", "太差", "差劲", "后悔", "难用",
    "waste", "refund", "return", "one star", "1 star",
    "terrible", "awful", "horrible", "disappointed",
)


def detect_explicit_negative_markers(text: object) -> list[str]:
    """Return explicit negative-review markers found in the text."""
    if not isinstance(text, str):
        return []
    low = text.lower()
    return [m for m in EXPLICIT_NEGATIVE_MARKERS if m in low]


def build_recommendations(level: str, causes: list[str]) -> list[str]:
    """Prioritized prescription templates (P0-P3)."""
    cause_text = "、".join(causes) if causes else "待复核"
    if level in ("高", "极高"):
        return [
            "P0: 平台介入审核，必要时暂停该商品展示，阻断退货与口碑扩散",
            "P1: 要求卖家补充/更换无修图实拍图，展示真实材质与细节",
            "P2: 修正商品描述中的绝对化承诺，补充瑕疵与尺码提示",
            f"P3: 针对根因（{cause_text}）启动专项整改与售后跟进",
        ]
    if level == "中":
        return [
            "P1: 将该商品纳入重点监控，跟踪评论与退货动态",
            f"P2: 针对根因（{cause_text}）提示卖家优化信息展示",
        ]
    substantive = [c for c in causes if c.startswith(RISK_CAUSE_PREFIXES)]
    if substantive:
        return [
            f"P1: 风险概率较低，但检测到{substantive[0]}信号，建议核实商品信息与实拍一致性",
            "P2: 持续监控评论与退货动态，若信号增强则升级处置",
        ]
    return [
        "P1: 维持当前信息展示水平，作为同品类标杆参考",
        "P2: 定期复查评论与预期差指标，防止质量波动",
    ]


def diagnose_item(
    scores: dict,
    risk_prob: float,
    top_shap: list[tuple[str, float]],
) -> dict:
    """Assemble the structured diagnosis for one item."""
    level = risk_level(risk_prob)
    causes = identify_root_causes(scores, top_shap)
    return {
        "risk_probability": float(risk_prob),
        "risk_level": level,
        "diagnosis": causes,
        "recommendations": build_recommendations(level, causes),
    }
