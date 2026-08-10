#!/usr/bin/env python3
"""Lightweight ONNX + MLflow helpers shared by training and manual registration."""

from __future__ import annotations

from pathlib import Path

from loguru import logger


def _onnx_tensor_shape(value_info) -> list[object]:
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(str(dim.dim_param))
        else:
            shape.append("?")
    return shape


def _onnx_tensor_dtype(value_info) -> str:
    from onnx import TensorProto

    return TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)


def extract_onnx_io_summary(onnx_path: Path) -> dict[str, list[dict[str, object]]]:
    import onnx

    model = onnx.load(str(onnx_path))
    graph = model.graph
    return {
        "inputs": [
            {
                "name": value.name,
                "dtype": _onnx_tensor_dtype(value),
                "shape": _onnx_tensor_shape(value),
            }
            for value in graph.input
        ],
        "outputs": [
            {
                "name": value.name,
                "dtype": _onnx_tensor_dtype(value),
                "shape": _onnx_tensor_shape(value),
            }
            for value in graph.output
        ],
    }


def _mlflow_dtype_from_onnx(dtype_name: str):
    import numpy as np

    mapping = {
        "FLOAT": np.dtype("float32"),
        "FLOAT16": np.dtype("float16"),
        "DOUBLE": np.dtype("float64"),
        "INT64": np.dtype("int64"),
        "INT32": np.dtype("int32"),
        "INT16": np.dtype("int16"),
        "INT8": np.dtype("int8"),
        "UINT8": np.dtype("uint8"),
        "BOOL": np.dtype("bool"),
        "STRING": np.dtype("str"),
    }
    return mapping.get(dtype_name.upper())


def _mlflow_shape_from_onnx(shape: list[object]) -> tuple[int, ...]:
    dims = []
    for dim in shape:
        dim_text = str(dim)
        dims.append(int(dim_text) if dim_text.isdigit() else -1)
    return tuple(dims)


def _make_tensor_spec(TensorSpec, mapped_dtype, shape: tuple[int, ...], name: str):
    try:
        return TensorSpec(dtype=mapped_dtype, shape=shape, name=name)
    except TypeError:
        return TensorSpec(type=mapped_dtype, shape=shape, name=name)


def register_mlflow_onnx_model(
    mlflow_client,
    onnx_path: Path,
    registered_model_name: str,
    io_summary: dict[str, list[dict[str, object]]] | None,
    artifact_path: str = "model",
) -> None:
    import mlflow.onnx
    import onnx
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, TensorSpec

    if not registered_model_name.strip():
        return

    onnx_model = onnx.load(str(onnx_path))
    inputs = [
        _make_tensor_spec(
            TensorSpec,
            mapped_dtype,
            _mlflow_shape_from_onnx(list(value["shape"])),
            str(value["name"]),
        )
        for value in (io_summary or {}).get("inputs", [])
        if (mapped_dtype := _mlflow_dtype_from_onnx(str(value["dtype"]))) is not None
    ]
    outputs = [
        _make_tensor_spec(
            TensorSpec,
            mapped_dtype,
            _mlflow_shape_from_onnx(list(value["shape"])),
            str(value["name"]),
        )
        for value in (io_summary or {}).get("outputs", [])
        if (mapped_dtype := _mlflow_dtype_from_onnx(str(value["dtype"]))) is not None
    ]
    signature = None
    if inputs or outputs:
        signature = ModelSignature(
            inputs=Schema(inputs) if inputs else None,
            outputs=Schema(outputs) if outputs else None,
        )

    metadata = {"custom17_onnx_io": io_summary} if io_summary is not None else None
    mlflow.onnx.log_model(
        onnx_model=onnx_model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name.strip(),
        signature=signature,
        metadata=metadata,
        save_as_external_data=False,
    )
    logger.info("Registered MLflow ONNX model: {}", registered_model_name.strip())
