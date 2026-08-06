"""Tabular feature engineering for the expectation-gap model."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

NEGATIVE_WORDS = {
    "not", "no", "never", "worst", "terrible", "horrible", "awful",
    "bad", "poor", "ugly", "disappointing", "disappointed", "cheap",
    "flimsy", "broken", "defective", "waste", "return", "refund",
    "trash", "garbage", "scam", "fake", "overpriced",
}

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "perfect", "love", "beautiful",
    "exceeded", "recommend", "comfortable", "sturdy", "quality",
    "happy", "pleased", "wonderful", "fantastic", "awesome", "nice",
    "best", "outstanding", "superb", "impressed",
}

CHINESE_NEGATIVE_WORDS = {
    "差评", "垃圾", "退货", "退款", "太差", "很差", "差劲", "糟糕",
    "失望", "后悔", "再也不买", "烂", "难看", "不值", "假货", "欺骗",
    "一星", "1星", "讨厌", "难用", "质量差", "质量不好", "质量不行",
    "不行", "破", "坏了", "脏", "色差", "尺码不对", "尺码偏", "发货慢",
    "不推荐", "别买", "态度差", "浪费钱", "假的", "骗人",
}

CHINESE_POSITIVE_WORDS = {
    "好评", "喜欢", "满意", "完美", "不错", "好用", "超值", "推荐",
    "很棒", "很喜欢", "质量好", "舒服", "好看",
}

DELTA_FEATURES = ["delta_cosine", "delta_euclidean"]
PRICE_FEATURES = ["P_rank", "delta_x_price_rank"]
TEXT_FEATURES = [
    "review_sentiment_diff", "review_neg_word_count", "review_pos_word_count",
    "review_exclamation_count", "review_word_count", "review_upper_ratio",
    "title_word_count", "delta_x_review_length", "delta_x_negation",
    "log_review_length",
]
IMG_META_FEATURES = ["img_kb_diff", "img_kb_ratio"]
FEATURE_COLUMNS = DELTA_FEATURES + PRICE_FEATURES + TEXT_FEATURES + IMG_META_FEATURES


def extract_text_features(text: object) -> dict:
    """Lexical review features used by the tabular branch."""
    if not isinstance(text, str):
        return {
            "review_length": 0,
            "review_word_count": 0,
            "review_exclamation_count": 0,
            "review_question_count": 0,
            "review_upper_ratio": 0.0,
            "review_neg_word_count": 0,
            "review_pos_word_count": 0,
            "review_sentiment_diff": 0,
        }
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = clean.replace("&#34;", '"').replace("&amp;", "&")
    words = clean.lower().split()
    word_set = set(words)
    neg_cn = sum(1 for w in CHINESE_NEGATIVE_WORDS if w in clean)
    pos_cn = sum(1 for w in CHINESE_POSITIVE_WORDS if w in clean)
    neg_count = len(word_set & NEGATIVE_WORDS) + neg_cn
    pos_count = len(word_set & POSITIVE_WORDS) + pos_cn
    return {
        "review_length": len(clean),
        "review_word_count": len(words),
        "review_exclamation_count": clean.count("!"),
        "review_question_count": clean.count("?"),
        "review_upper_ratio": sum(1 for c in clean if c.isupper()) / max(len(clean), 1),
        "review_neg_word_count": neg_count,
        "review_pos_word_count": pos_count,
        "review_sentiment_diff": pos_count - neg_count,
    }


def extract_title_features(title: object) -> dict:
    if not isinstance(title, str):
        return {"title_length": 0, "title_word_count": 0}
    return {"title_length": len(title), "title_word_count": len(title.split())}


def minmax_rank(series: pd.Series) -> pd.Series:
    """Percentile-style rank in [0, 1], robust for small samples."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(np.nan, index=series.index)
    return numeric.rank(method="average", pct=True)


def img_size_kb(path_value: object, image_root: Path | None) -> float:
    if not isinstance(path_value, str) or image_root is None:
        return np.nan
    p = image_root / path_value
    if p.exists():
        return p.stat().st_size / 1024.0
    return np.nan


def build_features(df: pd.DataFrame, image_root: Path | None = None) -> pd.DataFrame:
    """Add all 16 modeling features to a copy of ``df``.

    Required input columns: review_text, title, delta_cosine, delta_euclidean, price.
    Optional: seller_image_path, buyer_image_path (for image-size features).
    """
    out = df.copy()

    if "review_sentiment_diff" not in out.columns:
        text_feats = out["review_text"].apply(extract_text_features).apply(pd.Series)
        title_feats = out["title"].apply(extract_title_features).apply(pd.Series)
        out = pd.concat([out, text_feats, title_feats], axis=1)

    if "P_rank" not in out.columns:
        out["P_rank"] = minmax_rank(out["price"])

    if "img_kb_diff" not in out.columns:
        if "seller_image_path" in out.columns:
            out["seller_img_kb"] = out["seller_image_path"].apply(
                lambda v: img_size_kb(v, image_root)
            )
            out["buyer_img_kb"] = out["buyer_image_path"].apply(
                lambda v: img_size_kb(v, image_root)
            )
            out["img_kb_diff"] = out["seller_img_kb"] - out["buyer_img_kb"]
            out["img_kb_ratio"] = out["seller_img_kb"] / out["buyer_img_kb"].replace(0, np.nan)
        else:
            out["img_kb_diff"] = np.nan
            out["img_kb_ratio"] = np.nan

    if "delta_x_review_length" not in out.columns:
        out["delta_x_review_length"] = out["delta_cosine"] * out["review_word_count"]
        out["delta_x_negation"] = out["delta_cosine"] * out["review_neg_word_count"]
        out["delta_x_price_rank"] = out["delta_cosine"] * out["P_rank"]
        out["log_review_length"] = np.log1p(out["review_word_count"])

    return out


def select_features(df: pd.DataFrame, fill_value: float = 0.0) -> np.ndarray:
    """Return the (N, 16) feature matrix, replacing missing values."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_COLUMNS].fillna(fill_value).values.astype(np.float32)

