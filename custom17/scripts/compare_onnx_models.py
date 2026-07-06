#!/usr/bin/env python3
"""Compare two ONNX models for deployment diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper


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
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu", help="Execution provider for benchmarking.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for benchmarking.")
    parser.add_argument("--iterations", type=int, default=30, help="Timed iterations for benchmarking.")
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


if __name__ == "__main__":
    main()
