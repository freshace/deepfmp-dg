"""Gradio demo launcher (UI implemented in Task 11)."""
from __future__ import annotations

from pathlib import Path

from deepfmp_dg.infer import DEFAULT_MODEL_DIR

PathLike = str | Path


def launch(model_dir: PathLike = DEFAULT_MODEL_DIR, share: bool = False) -> None:
    raise NotImplementedError("Gradio UI is implemented in Task 11")
