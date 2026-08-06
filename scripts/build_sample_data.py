"""Build the small committed sample dataset from the full research data.

Usage:
    python scripts/build_sample_data.py --source "<research-data-root>" --out data/sample
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from deepfmp_dg.features import FEATURE_COLUMNS, build_features, select_features
from deepfmp_dg.scores import compute_alignment_features
from deepfmp_dg.train import save_checkpoint, set_seed, train_model

DEFAULT_SOURCE = Path(r"E:\Case Competition\Case Competition for PG\DeepFMP-DG\数据及其他")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "sample"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-rows", type=int, default=100)
    parser.add_argument("--n-image-pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = args.source / "数据集" / "data" / "processed"
    images_root = args.source / "数据集"
    models_dir = args.source / "训练模型" / "models"

    df = pd.read_csv(data_dir / "modeling_table_enhanced_4934.csv")
    visual_emb = np.load(data_dir / "siglip_embeddings_4934.npz")
    visual_meta = pd.read_csv(data_dir / "siglip_meta_4934.csv")
    score2_emb = np.load(data_dir / "score2_embeddings_4934.npz")
    score2_csv = pd.read_csv(data_dir / "score2_text_image_4934.csv")
    score3_emb = np.load(data_dir / "score3_embeddings_4934.npz")
    score3_csv = pd.read_csv(data_dir / "score3_text_text_4934.csv")

    # Align visual / score2 / score3 embeddings exactly as the research pipeline does.
    visual_indices = visual_meta.index.values
    score2_indices = score2_csv["row_index"].values
    score3_indices = score3_csv["row_index"].values
    common = np.intersect1d(np.intersect1d(visual_indices, score2_indices), score3_indices)

    v_map = {idx: i for i, idx in enumerate(visual_indices)}
    s2_map = {idx: i for i, idx in enumerate(score2_indices)}
    s3_map = {idx: i for i, idx in enumerate(score3_indices)}

    v_idx = np.array([v_map[idx] for idx in common])
    s2_idx = np.array([s2_map[idx] for idx in common])
    s3_idx = np.array([s3_map[idx] for idx in common])

    seller = np.asarray(visual_emb["seller_embs"][v_idx], dtype=np.float32)
    buyer = np.asarray(visual_emb["buyer_embs"][v_idx], dtype=np.float32)
    merchant = np.asarray(score3_emb["merchant_embs"][s3_idx], dtype=np.float32)
    review = np.asarray(score3_emb["review_embs"][s3_idx], dtype=np.float32)
    meta = visual_meta.iloc[v_idx].reset_index(drop=True)

    merged = meta[["parent_asin", "rating", "seller_image_path", "buyer_image_path"]].reset_index(drop=True)
    merged = merged.merge(
        df[["parent_asin", "rating", "Y_low_rating", "title", "review_text", "price"] + FEATURE_COLUMNS]
        .drop_duplicates(subset=["parent_asin", "rating"]),
        on=["parent_asin", "rating"],
        how="inner",
    )
    merged = merged.dropna(subset=["Y_low_rating"]).reset_index(drop=True)
    print(f"Aligned rows after merge: {len(merged)}")

    # Stratified sample: keep the class distribution of the full data.
    frac = min(1.0, args.n_rows / max(len(merged), 1))
    sampled_idx = (
        merged.groupby("Y_low_rating", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=args.seed))
        .index
        .to_numpy()
    )
    if len(sampled_idx) > args.n_rows:
        sampled_idx = np.sort(np.random.choice(sampled_idx, args.n_rows, replace=False))
    sampled_idx = np.sort(sampled_idx)

    sample_df = merged.iloc[sampled_idx].reset_index(drop=True)
    sample_seller = seller[sampled_idx]
    sample_buyer = buyer[sampled_idx]
    sample_merchant = merchant[sampled_idx]
    sample_review = review[sampled_idx]

    out = args.out
    (out / "images" / "seller").mkdir(parents=True, exist_ok=True)
    (out / "images" / "buyer").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)

    keep_cols = ["parent_asin", "rating", "Y_low_rating", "title", "review_text", "price"] + FEATURE_COLUMNS
    sample_df[keep_cols].to_csv(out / "modeling_sample.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        out / "embeddings_sample.npz",
        seller_embs=sample_seller,
        buyer_embs=sample_buyer,
        merchant_embs=sample_merchant,
        review_embs=sample_review,
    )

    # Copy a handful of image pairs for the demo (demo-only assets).
    copied = 0
    for _, row in sample_df.head(args.n_image_pairs).iterrows():
        seller_src = images_root / str(row["seller_image_path"])
        buyer_src = images_root / str(row["buyer_image_path"])
        if seller_src.exists():
            shutil.copy2(seller_src, out / "images" / "seller" / seller_src.name)
            copied += 1
        if buyer_src.exists():
            shutil.copy2(buyer_src, out / "images" / "buyer" / buyer_src.name)
            copied += 1
    print(f"Copied {copied} image files")

    # Fit small demo models on the sample.
    align = compute_alignment_features(sample_seller, sample_buyer, sample_merchant, sample_review)
    x = select_features(build_features(sample_df))
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    y = sample_df["Y_low_rating"].astype(int).to_numpy()

    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    split = int(n * 0.8)
    _, _, _, state, model = train_model(
        "direct", sample_seller, sample_buyer, align, x_scaled, y,
        idx[:split], idx[split:], epochs=30, patience=10,
    )
    if state is None:
        raise RuntimeError("Training failed on sample data")
    save_checkpoint(model, "direct", out / "models" / "deepfmp_direct.pt")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=args.seed, n_jobs=1,
    )
    rf.fit(x_scaled, y)
    joblib.dump(rf, out / "models" / "rf.joblib")
    joblib.dump(scaler, out / "models" / "scaler.joblib")
    (out / "models" / "feature_names.json").write_text(json.dumps(FEATURE_COLUMNS), encoding="utf-8")

    print(f"Sample saved to {out}: {len(sample_df)} rows, {args.n_image_pairs} image pairs")


if __name__ == "__main__":
    main()


