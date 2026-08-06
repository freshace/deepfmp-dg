# Model Card

## Model
- Name: DeepFMP-DG (direct-fusion variant)
- Task: binary classification — low-rating risk (rating <= 2)
- Params: ~270K
- Backbone: google/siglip-base-patch16-224 (frozen at feature-extraction time)
- Head: dual encoder + cross-modal attention + difference/Hadamard projections + align/tabular fusion

## Training data
- Research data: Amazon Reviews 2023, Fashion category; 3,412 paired samples with all three scores; 4,934 total pairs.
- Label: `Y_low_rating = (rating <= 2)`; positive rate ≈ 15.2%.
- Committed sample: 100 rows for demo/tests only.

## Features
- Alignment scores: Score1/2/3 (cosine similarity, higher = more aligned).
- Tabular (16): sentiment diff, pos/neg word counts, exclamation count, word count, upper-case ratio, title word count, price rank, cosine/euclidean deltas, image-size diff/ratio, and three interactions.

## Metrics (5-fold CV, seed=42, full research data)
- DeepFMP (visual): AUC 0.9069
- DeepFMP (visual+tab): AUC 0.9456
- DeepFMP (S1+S2+S3, direct): AUC 0.9477
- DeepFMP (softmax gate): AUC 0.9365 (gate weights S1=0.346, S2=0.322, S3=0.332)

## Intended use
- E-commerce platform risk control (early warning, listing review, dispute support)
- Seller-side content optimization (main image, description, price communication)
- Consumer expectation management

## Limitations
- Small sample, single category (Fashion), English Amazon data.
- The label is a proxy for expectation mismatch (low rating), not a direct measurement.
- Diagnosis rules are heuristic thresholds (documented in `diagnose.py`); use for triage, not automated enforcement.
- Sample images are public product photos for demonstration only.
