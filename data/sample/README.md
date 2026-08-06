# Sample Data

Small subset of the research dataset, committed so the project runs out of the box.

- `modeling_sample.csv` — 100 paired samples (seller image + buyer photo + review) with the 16 modeling features and the low-rating label.
- `embeddings_sample.npz` — SigLIP embeddings for the same rows (keys: `seller_embs`, `buyer_embs`, `merchant_embs`, `review_embs`).
- `images/` — a few seller/buyer image pairs used by the demo.
- `models/` — demo models trained on this sample (torch checkpoint, RF for SHAP, scaler, feature names).

**Demo-only notice:** the sample images are public product photos used for demonstration only.
Data source: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab).
