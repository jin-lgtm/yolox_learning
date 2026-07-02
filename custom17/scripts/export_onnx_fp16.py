#!/usr/bin/env python3
"""Export an existing custom17 checkpoint to FP16 ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = REPO_ROOT / "upstream_yolox"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from yolox.exp import get_exp

from custom17.runtime_patches import export_ckpt_to_fp16_onnx, patch_torch_load_for_checkpoints


def make_parser():
    parser = argparse.ArgumentParser("Export custom17 checkpoint to FP16 ONNX")
    parser.add_argument("-f", "--exp_file", required=True, type=str, help="Experiment file.")
    parser.add_argument("-c", "--ckpt", required=True, type=str, help="Path to .pth checkpoint.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="Output ONNX path. Default: <ckpt_dir>/<ckpt_stem>_fp16.onnx",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    return parser


def main():
    args = make_parser().parse_args()
    patch_torch_load_for_checkpoints()

    exp = get_exp(args.exp_file, None)
    ckpt_path = Path(args.ckpt).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else ckpt_path.with_suffix("").with_name(f"{ckpt_path.stem}_fp16.onnx")
    )
    export_ckpt_to_fp16_onnx(
        exp=exp,
        ckpt_path=ckpt_path,
        output_path=output_path,
        device="cuda:0" if args.device == "gpu" else "cpu",
    )


if __name__ == "__main__":
    main()
