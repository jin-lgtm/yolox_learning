#!/usr/bin/env python3
"""Compare two ONNX models for deployment diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = REPO_ROOT / "upstream_yolox"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from yolox.utils import multiclass_nms


def tensor_shape(value_info) -> list[str]:
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(str(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def tensor_dtype(value_info) -> str:
    elem_type = value_info.type.tensor_type.elem_type
    return TensorProto.DataType.Name(elem_type)


def summarize_model(path: Path) -> dict:
    model = onnx.load(str(path))
    graph = model.graph

    inputs = [
        {
            "name": value.name,
            "dtype": tensor_dtype(value),
            "shape": tensor_shape(value),
        }
        for value in graph.input
    ]
    outputs = [
        {
            "name": value.name,
            "dtype": tensor_dtype(value),
            "shape": tensor_shape(value),
        }
        for value in graph.output
    ]
    op_counts = Counter(node.op_type for node in graph.node)
    initializers_bytes = sum(init.raw_data and len(init.raw_data) or 0 for init in graph.initializer)

    return {
        "path": str(path),
        "ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "opset_imports": {op.domain or "ai.onnx": op.version for op in model.opset_import},
        "inputs": inputs,
        "outputs": outputs,
        "num_nodes": len(graph.node),
        "num_initializers": len(graph.initializer),
        "initializer_bytes": initializers_bytes,
        "op_counts": dict(sorted(op_counts.items())),
        "node_names": [node.name or f"{node.op_type}_{idx}" for idx, node in enumerate(graph.node)],
    }


def numpy_dtype_from_onnx(dtype_name: str):
    mapping = {
        "FLOAT": np.float32,
        "FLOAT16": np.float16,
        "DOUBLE": np.float64,
        "INT64": np.int64,
        "INT32": np.int32,
        "INT16": np.int16,
        "INT8": np.int8,
        "UINT8": np.uint8,
        "BOOL": np.bool_,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported ONNX input dtype for benchmarking: {dtype_name}")
    return mapping[dtype_name]


def concrete_input_shape(shape: list[str]) -> list[int]:
    dims = []
    for idx, dim in enumerate(shape):
        if dim.isdigit():
            dims.append(int(dim))
        elif idx == 0:
            dims.append(1)
        else:
            raise ValueError(f"Dynamic non-batch dimension is not supported for benchmarking: {shape}")
    return dims


def benchmark_model(path: Path, provider: str, warmup: int, iterations: int) -> dict:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = onnxruntime.InferenceSession(str(path), providers=providers)
    input_meta = session.get_inputs()[0]
    input_shape = concrete_input_shape(
        [str(dim) for dim in input_meta.shape]
    )
    dtype = numpy_dtype_from_onnx(input_meta.type.replace("tensor(", "").replace(")", "").upper())
    if np.issubdtype(dtype, np.floating):
        sample = np.random.rand(*input_shape).astype(dtype)
    elif np.issubdtype(dtype, np.bool_):
        sample = np.random.randint(0, 2, size=input_shape).astype(dtype)
    else:
        sample = np.random.randint(0, 8, size=input_shape).astype(dtype)

    ort_inputs = {input_meta.name: sample}
    for _ in range(warmup):
        session.run(None, ort_inputs)

    durations_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, ort_inputs)
        end = time.perf_counter()
        durations_ms.append((end - start) * 1000.0)

    return {
        "path": str(path),
        "provider": session.get_providers()[0] if session.get_providers() else "unknown",
        "input_name": input_meta.name,
        "input_type": input_meta.type,
        "input_shape": input_shape,
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": float(np.mean(durations_ms)),
        "median_ms": float(np.median(durations_ms)),
        "min_ms": float(np.min(durations_ms)),
        "max_ms": float(np.max(durations_ms)),
    }


def infer_strides_from_output_count(img_size: list[int], output_count: int) -> list[int]:
    candidate_stride_sets = (
        [8, 16, 32],
        [16, 32],
        [8, 16, 32, 64],
    )
    for strides in candidate_stride_sets:
        expected = sum((img_size[2] // stride) * (img_size[3] // stride) for stride in strides)
        if expected == output_count:
            return strides
    raise ValueError(
        f"Unable to infer strides for output_count={output_count} and input_shape={img_size}. "
        f"Known candidates: {candidate_stride_sets}"
    )


def decode_outputs_with_strides(outputs: np.ndarray, input_shape: list[int], strides: list[int]) -> np.ndarray:
    img_h, img_w = input_shape[2], input_shape[3]
    grids = []
    expanded_strides = []
    hsizes = [img_h // stride for stride in strides]
    wsizes = [img_w // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, axis=1)
    expanded_strides = np.concatenate(expanded_strides, axis=1)
    outputs = outputs.copy()
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def benchmark_model_with_nms(
    path: Path,
    provider: str,
    warmup: int,
    iterations: int,
    score_thr: float,
    nms_thr: float,
) -> dict:
    base = benchmark_model(path, provider, warmup=0, iterations=1)
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = onnxruntime.InferenceSession(str(path), providers=providers)
    input_meta = session.get_inputs()[0]
    input_shape = concrete_input_shape([str(dim) for dim in input_meta.shape])
    dtype = numpy_dtype_from_onnx(input_meta.type.replace("tensor(", "").replace(")", "").upper())
    sample = np.random.rand(*input_shape).astype(dtype) if np.issubdtype(dtype, np.floating) else np.random.randint(0, 8, size=input_shape).astype(dtype)
    ort_inputs = {input_meta.name: sample}

    sample_output = session.run(None, ort_inputs)[0]
    strides = infer_strides_from_output_count(input_shape, int(sample_output.shape[1]))

    for _ in range(warmup):
        output = session.run(None, ort_inputs)[0]
        decoded = decode_outputs_with_strides(output, input_shape, strides)[0]
        boxes = decoded[:, :4]
        scores = decoded[:, 4:5] * decoded[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)

    decode_ms = []
    nms_ms = []
    total_ms = []
    det_counts = []
    candidate_count = int(sample_output.shape[1])
    for _ in range(iterations):
        start = time.perf_counter()
        output = session.run(None, ort_inputs)[0]
        infer_end = time.perf_counter()

        decoded = decode_outputs_with_strides(output, input_shape, strides)[0]
        decode_end = time.perf_counter()

        boxes = decoded[:, :4]
        scores = decoded[:, 4:5] * decoded[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)
        nms_end = time.perf_counter()

        decode_ms.append((decode_end - infer_end) * 1000.0)
        nms_ms.append((nms_end - decode_end) * 1000.0)
        total_ms.append((nms_end - start) * 1000.0)
        det_counts.append(0 if dets is None else int(dets.shape[0]))

    return {
        "path": str(path),
        "provider": session.get_providers()[0] if session.get_providers() else "unknown",
        "input_shape": input_shape,
        "strides": strides,
        "score_thr": score_thr,
        "nms_thr": nms_thr,
        "candidates": candidate_count,
        "warmup": warmup,
        "iterations": iterations,
        "mean_decode_ms": float(np.mean(decode_ms)),
        "mean_nms_ms": float(np.mean(nms_ms)),
        "mean_total_ms": float(np.mean(total_ms)),
        "median_total_ms": float(np.median(total_ms)),
        "mean_detections": float(np.mean(det_counts)),
    }


def print_json(title: str, payload) -> None:
    print(title)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def compare_lists(name: str, a, b) -> None:
    if a == b:
        print(f"{name}: same")
        return
    print(f"{name}: different")
    print_json(f"{name} A", a)
    print_json(f"{name} B", b)


def compare_dict_counts(name: str, a: dict, b: dict) -> None:
    keys = sorted(set(a) | set(b))
    rows = []
    for key in keys:
        av = a.get(key, 0)
        bv = b.get(key, 0)
        if av != bv:
            rows.append((key, av, bv, bv - av))
    if not rows:
        print(f"{name}: same")
        return
    print(f"{name}: different")
    for key, av, bv, diff in rows:
        print(f"  {key}: A={av}, B={bv}, delta={diff}")


def main() -> None:
    parser = argparse.ArgumentParser("Compare two ONNX models")
    parser.add_argument("--a", required=True, type=Path, help="Reference ONNX model path.")
    parser.add_argument("--b", required=True, type=Path, help="Candidate ONNX model path.")
    parser.add_argument("--dump-json", action="store_true", help="Print full summaries as JSON.")
    parser.add_argument("--benchmark", action="store_true", help="Measure ORT inference time for both models.")
    parser.add_argument(
        "--benchmark-nms",
        action="store_true",
        help="Measure ORT inference + decode + NMS for both models.",
    )
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu", help="Execution provider for benchmarking.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for benchmarking.")
    parser.add_argument("--iterations", type=int, default=30, help="Timed iterations for benchmarking.")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold for NMS benchmark.")
    parser.add_argument("--nms-thr", type=float, default=0.45, help="NMS threshold for NMS benchmark.")
    args = parser.parse_args()

    summary_a = summarize_model(args.a.resolve())
    summary_b = summarize_model(args.b.resolve())

    print(f"A: {summary_a['path']}")
    print(f"B: {summary_b['path']}")
    print(f"IR version: A={summary_a['ir_version']} B={summary_b['ir_version']}")
    print(f"Producer: A={summary_a['producer_name']} {summary_a['producer_version']}")
    print(f"Producer: B={summary_b['producer_name']} {summary_b['producer_version']}")
    compare_lists("Opset imports", summary_a["opset_imports"], summary_b["opset_imports"])
    compare_lists("Inputs", summary_a["inputs"], summary_b["inputs"])
    compare_lists("Outputs", summary_a["outputs"], summary_b["outputs"])

    if summary_a["num_nodes"] == summary_b["num_nodes"]:
        print(f"Node count: same ({summary_a['num_nodes']})")
    else:
        print(f"Node count: A={summary_a['num_nodes']} B={summary_b['num_nodes']}")

    if summary_a["num_initializers"] == summary_b["num_initializers"]:
        print(f"Initializer count: same ({summary_a['num_initializers']})")
    else:
        print(
            f"Initializer count: A={summary_a['num_initializers']} B={summary_b['num_initializers']}"
        )

    if summary_a["initializer_bytes"] == summary_b["initializer_bytes"]:
        print(f"Initializer bytes: same ({summary_a['initializer_bytes']})")
    else:
        print(
            f"Initializer bytes: A={summary_a['initializer_bytes']} B={summary_b['initializer_bytes']}"
        )

    compare_dict_counts("Op histogram", summary_a["op_counts"], summary_b["op_counts"])

    node_set_a = set(summary_a["node_names"])
    node_set_b = set(summary_b["node_names"])
    only_a = sorted(node_set_a - node_set_b)
    only_b = sorted(node_set_b - node_set_a)
    print(f"Node names only in A: {len(only_a)}")
    if only_a[:20]:
        print_json("Only in A (first 20)", only_a[:20])
    print(f"Node names only in B: {len(only_b)}")
    if only_b[:20]:
        print_json("Only in B (first 20)", only_b[:20])

    if args.dump_json:
        print_json("Summary A", summary_a)
        print_json("Summary B", summary_b)

    if args.benchmark:
        bench_a = benchmark_model(args.a.resolve(), args.provider, args.warmup, args.iterations)
        bench_b = benchmark_model(args.b.resolve(), args.provider, args.warmup, args.iterations)
        print("Benchmark A")
        print_json("A timing", bench_a)
        print("Benchmark B")
        print_json("B timing", bench_b)
        delta = bench_b["mean_ms"] - bench_a["mean_ms"]
        ratio = bench_b["mean_ms"] / bench_a["mean_ms"] if bench_a["mean_ms"] > 0 else float("inf")
        print(f"Benchmark mean delta (B-A): {delta:.3f} ms")
        print(f"Benchmark mean ratio (B/A): {ratio:.3f}x")

    if args.benchmark_nms:
        bench_a = benchmark_model_with_nms(
            args.a.resolve(),
            args.provider,
            args.warmup,
            args.iterations,
            args.score_thr,
            args.nms_thr,
        )
        bench_b = benchmark_model_with_nms(
            args.b.resolve(),
            args.provider,
            args.warmup,
            args.iterations,
            args.score_thr,
            args.nms_thr,
        )
        print("Benchmark+NMS A")
        print_json("A timing+postprocess", bench_a)
        print("Benchmark+NMS B")
        print_json("B timing+postprocess", bench_b)
        delta = bench_b["mean_total_ms"] - bench_a["mean_total_ms"]
        ratio = bench_b["mean_total_ms"] / bench_a["mean_total_ms"] if bench_a["mean_total_ms"] > 0 else float("inf")
        print(f"Benchmark+NMS total mean delta (B-A): {delta:.3f} ms")
        print(f"Benchmark+NMS total mean ratio (B/A): {ratio:.3f}x")


if __name__ == "__main__":
    main()
