#!/usr/bin/env python3
"""Benchmark a single ONNX model for deployment diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from custom17.scripts.compare_onnx_models import (
    benchmark_model_with_nms,
    benchmark_model_with_nms_on_images,
    load_benchmark_images,
)


def make_parser():
    parser = argparse.ArgumentParser("Benchmark a single ONNX model")
    parser.add_argument("-m", "--model", required=True, type=Path, help="ONNX model path.")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--score-thr", type=float, default=0.2)
    parser.add_argument("--nms-thr", type=float, default=0.45)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main():
    args = make_parser().parse_args()
    frames = load_benchmark_images(args.image, args.image_dir)
    if frames:
        frames = frames[: max(args.max_images, 1)]
        result = benchmark_model_with_nms_on_images(
            args.model.resolve(),
            args.provider,
            args.warmup,
            args.iterations,
            args.score_thr,
            args.nms_thr,
            frames,
        )
    else:
        result = benchmark_model_with_nms(
            args.model.resolve(),
            args.provider,
            args.warmup,
            args.iterations,
            args.score_thr,
            args.nms_thr,
        )

    payload = {
        "model": str(args.model),
        "provider": args.provider,
        "image_dir": args.image_dir,
        "image": args.image,
        "max_images": args.max_images,
        **result,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
