"""Command-line interface for deepfmp-dg."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepfmp_dg import __version__
from deepfmp_dg.infer import DEFAULT_MODEL_DIR, InferenceEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepfmp-dg",
        description="Multimodal expectation-gap diagnosis for e-commerce",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="Diagnose one item from images and text")
    p.add_argument("--seller-img", required=True, help="Path to the seller display image")
    p.add_argument("--buyer-img", required=True, help="Path to the buyer photo")
    p.add_argument("--title", default="", help="Product title")
    p.add_argument("--description", default="", help="Product description")
    p.add_argument("--review", required=True, help="Buyer review text")
    p.add_argument("--price", type=float, default=None, help="Product price")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("demo", help="Launch the interactive Gradio demo")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--share", action="store_true", help="Create a public share link")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("train", help="Train the direct-fusion model on a sample dataset")
    p.add_argument("--data", type=Path, required=True, help="Modeling table CSV")
    p.add_argument("--embeddings", type=Path, required=True, help="NPZ with seller/buyer/merchant/review embeddings")
    p.add_argument("--out", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Evaluate a checkpoint with 5-fold CV")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_evaluate)

    return parser


def cmd_predict(args) -> int:
    engine = InferenceEngine(args.model_dir)
    result = engine.predict(
        seller_image=args.seller_img,
        buyer_image=args.buyer_img,
        title=args.title,
        description=args.description,
        review_text=args.review,
        price=args.price,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_demo(args) -> int:
    from deepfmp_dg.demo_app import launch

    launch(model_dir=args.model_dir, share=args.share)
    return 0


def cmd_train(args) -> int:
    import json as _json

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    from deepfmp_dg.features import FEATURE_COLUMNS, build_features, select_features
    from deepfmp_dg.scores import compute_alignment_features
    from deepfmp_dg.train import run_experiment, save_checkpoint, set_seed, train_model

    set_seed(args.seed)
    df = pd.read_csv(args.data)
    emb = np.load(args.embeddings)
    seller = emb["seller_embs"]
    buyer = emb["buyer_embs"]
    merchant = emb["merchant_embs"]
    review = emb["review_embs"]
    align = compute_alignment_features(seller, buyer, merchant, review)
    x = select_features(build_features(df))
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    y = df["Y_low_rating"].astype(int).to_numpy()

    res = run_experiment(
        "DeepFMP (S1+S2+S3, direct)", "direct", seller, buyer, align, x_scaled, y,
        n_splits=5, epochs=args.epochs,
    )
    if res is not None:
        summary = {k: v for k, v in res.items() if k not in ("y", "prob")}
        print(_json.dumps(summary, ensure_ascii=False, indent=2))

    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    split = int(n * 0.8)
    _, _, _, state, model = train_model(
        "direct", seller, buyer, align, x_scaled, y,
        idx[:split], idx[split:], epochs=args.epochs,
    )
    if state is not None:
        save_checkpoint(model, "direct", args.out / "deepfmp_direct.pt")
    joblib.dump(scaler, args.out / "scaler.joblib")
    (args.out / "feature_names.json").write_text(_json.dumps(FEATURE_COLUMNS), encoding="utf-8")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=args.seed, n_jobs=1,
    )
    rf.fit(x_scaled, y)
    joblib.dump(rf, args.out / "rf.joblib")
    print(f"Checkpoint + scaler + RF saved to {args.out}")
    return 0


def cmd_evaluate(args) -> int:
    import json as _json

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    from deepfmp_dg.features import build_features, select_features
    from deepfmp_dg.scores import compute_alignment_features
    from deepfmp_dg.train import run_experiment, set_seed

    set_seed(args.seed)
    df = pd.read_csv(args.data)
    emb = np.load(args.embeddings)
    align = compute_alignment_features(
        emb["seller_embs"], emb["buyer_embs"], emb["merchant_embs"], emb["review_embs"]
    )
    x = select_features(build_features(df))
    scaler = StandardScaler().fit(x)
    y = df["Y_low_rating"].astype(int).to_numpy()
    for name, model_type in [
        ("DeepFMP (visual)", "visual"),
        ("DeepFMP (visual+tab)", "visual_tab"),
        ("DeepFMP (S1+S2+S3, direct)", "direct"),
        ("DeepFMP (softmax-gate)", "softmax_gate"),
    ]:
        res = run_experiment(
            name, model_type, emb["seller_embs"], emb["buyer_embs"], align,
            scaler.transform(x), y, n_splits=5, epochs=args.epochs,
        )
        if res is not None:
            summary = {k: v for k, v in res.items() if k not in ("y", "prob")}
            print(_json.dumps(summary, ensure_ascii=False))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

