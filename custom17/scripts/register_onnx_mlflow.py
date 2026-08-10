#!/usr/bin/env python3
"""Upload an existing ONNX file to MLflow and optionally register it."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlflow
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = REPO_ROOT / "upstream_yolox"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from custom17.mlflow_onnx import extract_onnx_io_summary, register_mlflow_onnx_model


def make_parser():
    parser = argparse.ArgumentParser("Register an ONNX model to MLflow")
    parser.add_argument("-m", "--model", required=True, type=str, help="Path to source ONNX file.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional MLflow run name.")
    parser.add_argument(
        "--artifact-path",
        type=str,
        default="model",
        help="Artifact path for MLflow ONNX model logging. Default: model",
    )
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default=None,
        help="Optional MLflow Model Registry name. If omitted, only artifacts are uploaded.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional local output directory for generated model.onnx and model_io.json. Default: source ONNX directory",
    )
    parser.add_argument(
        "--upload-artifacts",
        action="store_true",
        default=False,
        help="Also upload model.onnx and model_io.json as standard MLflow artifacts under the current run.",
    )
    return parser


def main():
    args = make_parser().parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must be set to upload/register an ONNX model.")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "").strip()
    if not experiment_name:
        logger.warning("MLFLOW_EXPERIMENT_NAME is not set. MLflow will use the Default experiment.")

    source_onnx = Path(args.model).resolve()
    if not source_onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {source_onnx}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else source_onnx.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    deploy_onnx = output_dir / "model.onnx"
    if source_onnx != deploy_onnx.resolve():
        shutil.copy2(source_onnx, deploy_onnx)
    io_summary = extract_onnx_io_summary(deploy_onnx)
    io_json_path = output_dir / "model_io.json"
    with io_json_path.open("w", encoding="utf-8") as fp:
        json.dump(io_summary, fp, ensure_ascii=False, indent=2)

    mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "custom17.onnx_inputs": json.dumps(io_summary["inputs"], separators=(",", ":"), ensure_ascii=False),
                "custom17.onnx_outputs": json.dumps(io_summary["outputs"], separators=(",", ":"), ensure_ascii=False),
                "custom17.model_source_path": str(source_onnx),
            }
        )
        if args.upload_artifacts:
            mlflow.log_artifact(str(deploy_onnx), args.artifact_path)
            mlflow.log_artifact(str(io_json_path), args.artifact_path)

        if args.registered_model_name:
            register_mlflow_onnx_model(
                mlflow,
                deploy_onnx,
                registered_model_name=args.registered_model_name,
                io_summary=io_summary,
                artifact_path=args.artifact_path,
            )
        else:
            import onnx

            mlflow.onnx.log_model(
                onnx_model=onnx.load(str(deploy_onnx)),
                artifact_path=args.artifact_path,
                metadata={"custom17_onnx_io": io_summary},
            )
            logger.info("Logged MLflow ONNX model without registry name: {}", deploy_onnx)

    logger.info("Prepared MLflow ONNX artifacts at {}", output_dir)


if __name__ == "__main__":
    main()
