"""Assemble a deployable Hugging Face Space from the repository.

The Space needs the package source, sample data, and a small launcher.
This script builds ``dist_space/`` (gitignored) with everything required;
push that folder to https://huggingface.co/spaces/<user>/deepfmp-dg-demo.

Usage:
    python scripts/build_hf_space.py [--out dist_space]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPACE_README_TEMPLATE = """---
title: DeepFMP-DG
emoji: 🔍
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: mit
---

# DeepFMP-DG

Interactive demo of the multimodal expectation-gap diagnosis model.
Upload a seller display image, a buyer photo, and the review text to get
the three gap scores, low-rating risk, SHAP attribution, and prescriptions.

First run downloads the SigLIP backbone (~1 GB) and caches it.
"""

APP_PY = '''"""Hugging Face Space launcher."""
from pathlib import Path

from deepfmp_dg.demo_app import launch

if __name__ == "__main__":
    launch(model_dir=Path(__file__).resolve().parent / "data" / "sample" / "models")
'''

REQUIREMENTS = """-e .
numpy>=1.24
pandas>=2.0
torch>=2.0
transformers>=4.40
accelerate>=0.27
Pillow>=10.0
scikit-learn>=1.3
shap>=0.44
gradio>=4.0
matplotlib>=3.7
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "dist_space")
    args = parser.parse_args()

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "src").mkdir(parents=True)

    shutil.copytree(ROOT / "src", out / "src", dirs_exist_ok=True)
    shutil.copytree(ROOT / "data" / "sample", out / "data" / "sample", dirs_exist_ok=True)
    shutil.copy2(ROOT / "pyproject.toml", out / "pyproject.toml")

    (out / "app.py").write_text(APP_PY, encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    import gradio

    readme = SPACE_README_TEMPLATE.format(gradio_version=gradio.__version__)
    (out / "README.md").write_text(readme, encoding="utf-8")

    print(f"Space bundle ready at {out}")
    print("Next: cd dist_space && git init && git add -A && git commit -m 'init'")
    print("Then push to https://huggingface.co/spaces/<user>/deepfmp-dg-demo (branch main)")


if __name__ == "__main__":
    main()
