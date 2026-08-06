# DeepFMP-DG

**Multimodal expectation-gap diagnosis for e-commerce.**

DeepFMP-DG quantifies how far a product's *seller-side presentation* (display image, description, price) diverges from the *buyer-side experience* (real photo, review text), predicts low-rating risk, and explains *why* with an interpretable diagnosis.

[![CI](https://github.com/your-name/deepfmp-dg/actions/workflows/ci.yml/badge.svg)](https://github.com/your-name/deepfmp-dg/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Highlights

- **Three interpretable expectation-gap scores** — image-image (seller display vs buyer photo), text-image (description vs buyer photo), and text-text (description vs review), all in a shared SigLIP embedding space.
- **Lightweight end-to-end network** — dual visual encoders with cross-modal attention, difference projection, Hadamard interaction, and a gated fusion head (~270K parameters, CPU-friendly).
- **Explainable loop** — risk probability + TreeSHAP attribution + gate weights + five-level item diagnosis with prioritized prescriptions.
- **Reproducible** — fixed seed (`42`), stratified 5-fold CV, packaged CLI, unit tests, and CI.

## Results

On the full research dataset (3,412 paired samples, 5-fold CV, seed=42):

| Model | AUC |
|---|---|
| DeepFMP (visual) | 0.9069 |
| DeepFMP (visual + tabular) | 0.9456 |
| **DeepFMP (S1+S2+S3, direct)** | **0.9477** |
| DeepFMP (Softmax gate) | 0.9365 |

Run `python -m deepfmp_dg.cli evaluate --data <modeling-table.csv> --embeddings <embeddings.npz>` to reproduce.

## Quickstart

```bash
pip install -e ".[dev]"
python -m deepfmp_dg.cli predict \
  --seller-img data/sample/images/seller/<file>.jpg \
  --buyer-img data/sample/images/buyer/<file>.jpg \
  --title "Summer Dress" \
  --description "Lightweight cotton dress" \
  --review "Great quality, fits perfectly!" \
  --price 25.0
```

## Interactive demo

```bash
python -m deepfmp_dg.cli demo
```

Open `http://127.0.0.1:7860`, upload a seller display image and a buyer photo, and inspect the three scores, risk level, SHAP attribution, and prescriptions.

## Repository layout

```
src/deepfmp_dg/   # installable package (scores, features, models, train, explain, diagnose, infer, cli)
demo/             # thin wrapper for the Gradio app
examples/         # notebook walkthrough
data/sample/      # small committed sample + demo models (images are demo-only)
scripts/          # sample-data builder and full-data reproduction
tests/            # unit tests
docs/             # architecture + model card
```

## Relation to DeepFMP

This project is inspired by and extends **DeepFMP** (Zhang, Ji & Cai, *Clothing Recommendation with Multimodal Feature Fusion: Price Sensitivity and Personalization Optimization*, Applied Sciences 15(8):4591, 2025). DeepFMP targets outfit recommendation with visual/text/price fusion; DeepFMP-DG adapts the multimodal-fusion idea to *single-item expectation-gap diagnosis*:

| | DeepFMP | DeepFMP-DG |
|---|---|---|
| Task | Outfit recommendation | Single-item expectation-gap diagnosis |
| Input | Visual + text + price | Seller-side vs buyer-side paired information |
| Mechanism | Enhanced DeepFM + attention | Three gap scores + dual cross-modal attention + gated fusion |
| Output | Ranking | Low-rating risk + interpretable diagnosis |

This is an extension implementation, **not** the official DeepFMP code.

## Acknowledgements

- **DeepFMP** — Zhang, Ji & Cai, Applied Sciences 15(8):4591, 2025.
- **SigLIP** — Zhai et al., *Sigmoid Loss for Language Image Pre-Training*.
- **Amazon Reviews 2023** — McAuley Lab dataset used for research and sample data.
- Sample images are public product photos included for demonstration only.

## Roadmap

- FastAPI serving + Docker
- Hugging Face-hosted weights and datasets
- Chinese e-commerce (buyer-show images) transfer experiments
- VLM-assisted fine-grained review (OCR/VQA) for high-risk items

## License

MIT
