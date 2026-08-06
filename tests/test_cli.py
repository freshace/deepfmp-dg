import subprocess
import sys

import pytest

from deepfmp_dg.cli import build_parser


def test_version_flag():
    out = subprocess.run(
        [sys.executable, "-m", "deepfmp_dg.cli", "--version"],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0
    assert "0.1.0" in out.stdout


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_predict_args():
    args = build_parser().parse_args([
        "predict", "--seller-img", "s.jpg", "--buyer-img", "b.jpg", "--review", "ok",
    ])
    assert args.command == "predict"
    assert args.seller_img == "s.jpg"

