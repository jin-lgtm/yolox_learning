#!/usr/bin/env python3
"""Append one ablation result row to a CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDNAMES = [
    "variant",
    "status",
    "exp_name",
    "output_dir",
    "onnx_path",
    "metrics_json",
    "benchmark_json",
    "mAP50",
    "mAP50_95",
    "AP_small",
    "AP_medium",
    "AP_large",
    "avg_forward_ms",
    "avg_nms_ms",
    "avg_total_ms",
    "bench_input_shape",
    "bench_candidates",
    "bench_preprocess_ms",
    "bench_infer_ms",
    "bench_decode_ms",
    "bench_nms_ms",
    "bench_total_ms",
    "bench_scored_candidates",
    "bench_detections",
]


def read_json(path: str | None) -> dict:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main():
    parser = argparse.ArgumentParser("Append ablation summary row")
    parser.add_argument("--csv-path", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--benchmark-json", default="")
    args = parser.parse_args()

    metrics = read_json(args.metrics_json)
    bench = read_json(args.benchmark_json)

    row = {
        "variant": args.variant,
        "status": args.status,
        "exp_name": args.exp_name,
        "output_dir": args.output_dir,
        "onnx_path": args.onnx_path,
        "metrics_json": args.metrics_json,
        "benchmark_json": args.benchmark_json,
        "mAP50": metrics.get("mAP50", ""),
        "mAP50_95": metrics.get("mAP50_95", ""),
        "AP_small": metrics.get("AP_small", ""),
        "AP_medium": metrics.get("AP_medium", ""),
        "AP_large": metrics.get("AP_large", ""),
        "avg_forward_ms": metrics.get("avg_forward_ms", ""),
        "avg_nms_ms": metrics.get("avg_nms_ms", ""),
        "avg_total_ms": metrics.get("avg_total_ms", ""),
        "bench_input_shape": json.dumps(bench.get("input_shape", []), ensure_ascii=False),
        "bench_candidates": bench.get("candidates", ""),
        "bench_preprocess_ms": bench.get("mean_preprocess_ms", ""),
        "bench_infer_ms": bench.get("mean_infer_ms", ""),
        "bench_decode_ms": bench.get("mean_decode_ms", ""),
        "bench_nms_ms": bench.get("mean_nms_ms", ""),
        "bench_total_ms": bench.get("mean_total_ms", ""),
        "bench_scored_candidates": bench.get("mean_scored_candidates", ""),
        "bench_detections": bench.get("mean_detections", ""),
    }

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = args.csv_path.exists()
    with args.csv_path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
