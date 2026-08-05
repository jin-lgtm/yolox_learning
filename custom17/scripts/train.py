#!/usr/bin/env python3
"""Run YOLOX training with the upstream submodule on sys.path."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = REPO_ROOT / "upstream_yolox"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from custom17.runtime_patches import (
    patch_trainer_for_balanced_resample_length,
    patch_coco_evaluator_output,
    patch_occupy_mem_safeguard,
    patch_mlflow_logger_for_custom17,
    patch_trainer_for_mlflow_eval_metrics,
    patch_trainer_for_onnx_export,
    patch_torch_load_for_checkpoints,
)


def apply_custom17_train_args() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--balanced-resample", action="store_true")
    parser.add_argument("--balanced-resample-seed", type=int, default=None)
    parser.add_argument("--balanced-target-count", type=int, default=None)
    parser.add_argument("--mlflow-log-onnx-model", action="store_true")
    parser.add_argument("--mlflow-register-onnx-model", type=str, default=None)
    args, remaining = parser.parse_known_args(sys.argv[1:])

    if args.balanced_resample:
        os.environ["CUSTOM17_BALANCED_RESAMPLE"] = "1"
    if args.balanced_resample_seed is not None:
        os.environ["CUSTOM17_BALANCED_RESAMPLE_SEED"] = str(args.balanced_resample_seed)
    if args.balanced_target_count is not None:
        os.environ["CUSTOM17_BALANCED_TARGET_COUNT"] = str(args.balanced_target_count)
    if args.mlflow_log_onnx_model:
        os.environ["CUSTOM17_MLFLOW_LOG_ONNX_MODEL"] = "1"
    if args.mlflow_register_onnx_model:
        os.environ["CUSTOM17_MLFLOW_LOG_ONNX_MODEL"] = "1"
        os.environ["CUSTOM17_MLFLOW_REGISTER_ONNX_MODEL_NAME"] = args.mlflow_register_onnx_model

    sys.argv = [sys.argv[0], *remaining]


apply_custom17_train_args()
patch_torch_load_for_checkpoints()
patch_coco_evaluator_output()
patch_occupy_mem_safeguard()
patch_mlflow_logger_for_custom17()
patch_trainer_for_mlflow_eval_metrics()
patch_trainer_for_balanced_resample_length()
patch_trainer_for_onnx_export()
runpy.run_path(str(YOLOX_ROOT / "tools" / "train.py"), run_name="__main__")
