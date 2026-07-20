#!/usr/bin/env python3
"""Runtime patches for YOLOX integration."""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from loguru import logger
import torch
from tabulate import tabulate

from custom17.common import CUSTOM17_CLASSES


def _build_metric_table(metric_by_class, headers, columns=6):
    num_cols = min(columns, len(metric_by_class) * len(headers))
    result_pair = [x for pair in metric_by_class.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    return tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left"
    )


def _per_class_ap50_table(coco_eval, class_names: Sequence[str], headers=None, columns=6):
    headers = headers or ["class", "AP50"]
    precisions = coco_eval.eval["precision"]
    assert len(class_names) == precisions.shape[2]

    iou_thresholds = np.array(coco_eval.params.iouThrs)
    iou_index = int(np.argmin(np.abs(iou_thresholds - 0.5)))
    if not np.isclose(iou_thresholds[iou_index], 0.5):
        raise ValueError(f"Unable to find IoU=0.50 in COCOeval thresholds: {iou_thresholds}")

    per_class_ap50 = {}
    for idx, name in enumerate(class_names):
        precision = precisions[iou_index, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap50 = np.mean(precision) if precision.size else float("nan")
        per_class_ap50[name] = float(ap50 * 100)

    return _build_metric_table(per_class_ap50, headers=headers, columns=columns)


def patch_coco_evaluator_output() -> None:
    from yolox.evaluators import coco_evaluator as coco_eval_module
    from yolox.utils import is_main_process

    if getattr(coco_eval_module, "_custom17_ap50_patch_applied", False):
        return

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")
        ann_type = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )
        info = time_info + "\n"

        if len(data_dict) == 0:
            return 0, 0, info

        coco_gt = self.dataloader.dataset.coco
        if self.testdev:
            json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
            coco_dt = coco_gt.loadRes("./yolox_testdev_2017.json")
        else:
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            coco_dt = coco_gt.loadRes(tmp)

        try:
            from yolox.layers import COCOeval_opt as COCOeval
        except ImportError:
            from pycocotools.cocoeval import COCOeval

            logger.warning("Use standard COCOeval.")

        coco_eval = COCOeval(coco_gt, coco_dt, ann_type[1])
        coco_eval.evaluate()
        coco_eval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        info += redirect_string.getvalue()

        cat_ids = list(coco_gt.cats.keys())
        cat_names = [coco_gt.cats[cat_id]["name"] for cat_id in sorted(cat_ids)]
        if self.per_class_AP:
            ap_table = coco_eval_module.per_class_AP_table(coco_eval, class_names=cat_names)
            ap50_table = _per_class_ap50_table(coco_eval, class_names=cat_names)
            info += "per class AP:\n" + ap_table + "\n"
            info += "per class AP50:\n" + ap50_table + "\n"
        if self.per_class_AR:
            ar_table = coco_eval_module.per_class_AR_table(coco_eval, class_names=cat_names)
            info += "per class AR:\n" + ar_table + "\n"
        return coco_eval.stats[0], coco_eval.stats[1], info

    coco_eval_module.COCOEvaluator.evaluate_prediction = evaluate_prediction
    coco_eval_module._custom17_ap50_patch_applied = True


def patch_torch_load_for_checkpoints() -> None:
    """Restore pre-2.6 torch.load behavior for trusted YOLOX checkpoints."""
    if getattr(torch, "_custom17_torch_load_patch_applied", False):
        return

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    torch._custom17_torch_load_patch_applied = True


def _log_mlflow_artifact(mlflow_logger, artifact_path: Path, artifact_dir: str) -> None:
    if not artifact_path.exists():
        return
    try:
        mlflow_logger._ml_flow.log_artifact(str(artifact_path), artifact_dir)
        logger.info("Logged MLflow artifact: {}", artifact_path)
    except Exception:
        logger.exception("Failed to log MLflow artifact: {}", artifact_path)


def patch_mlflow_logger_for_custom17() -> None:
    from yolox.utils import is_main_process
    import yolox.utils.mlflow_logger as mlflow_logger_module

    if getattr(mlflow_logger_module, "_custom17_mlflow_patch_applied", False):
        return

    original_setup = mlflow_logger_module.MlflowLogger.setup
    original_on_train_end = mlflow_logger_module.MlflowLogger.on_train_end

    def patched_setup(self, args, exp):
        if os.getenv("YOLOX_MLFLOW_RUN_NAME") is None:
            os.environ["YOLOX_MLFLOW_RUN_NAME"] = ""
        original_setup(self, args, exp)
        if not is_main_process() or not getattr(self, "_initialized", False):
            return

        extra_params = {
            "custom17.num_classes": len(CUSTOM17_CLASSES),
            "custom17.class_names": json.dumps(list(CUSTOM17_CLASSES), separators=(",", ":")),
            "custom17.input_override": os.getenv("CUSTOM17_INPUT_SIZE", ""),
            "custom17.data_source": os.getenv("CUSTOM17_DATA_SOURCE", ""),
        }
        self.log_params_mlflow(extra_params)

    def patched_on_train_end(self, args, file_name, metadata):
        if is_main_process() and getattr(self, "_initialized", False):
            artifact_dir = args.experiment_name
            file_dir = Path(file_name)
            _log_mlflow_artifact(self, file_dir / "best_ckpt.pth", artifact_dir)
            _log_mlflow_artifact(self, file_dir / "best_ckpt.onnx", artifact_dir)
            exp_file = getattr(args, "exp_file", None)
            if exp_file:
                _log_mlflow_artifact(self, Path(exp_file).resolve(), artifact_dir)
        return original_on_train_end(self, args, file_name, metadata)

    mlflow_logger_module.MlflowLogger.setup = patched_setup
    mlflow_logger_module.MlflowLogger.on_train_end = patched_on_train_end
    mlflow_logger_module._custom17_mlflow_patch_applied = True


def export_best_ckpt_to_onnx(trainer) -> Path | None:
    return export_ckpt_to_onnx(
        exp=trainer.exp,
        ckpt_path=Path(trainer.file_name) / "best_ckpt.pth",
        output_path=Path(trainer.file_name) / getattr(trainer.exp, "onnx_export_name", "best_ckpt.onnx"),
        device=trainer.device if torch.cuda.is_available() else "cpu",
    )


def export_best_ckpt_to_fp16_onnx(trainer) -> Path | None:
    return export_best_ckpt_to_onnx(trainer)


def export_ckpt_to_onnx(exp, ckpt_path: Path, output_path: Path, device: str = "cpu") -> Path | None:
    from torch import nn
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module

    ckpt_path = Path(ckpt_path)
    output_path = Path(output_path)
    if not ckpt_path.exists():
        logger.warning("Skip ONNX export because checkpoint was not found: {}", ckpt_path)
        return None

    opset_version = int(getattr(exp, "onnx_opset", 11))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting checkpoint to ONNX: {}", output_path)
    model = exp.get_model()
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False

    export_device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
    model.to(export_device)

    dummy_input = torch.randn(
        1,
        3,
        exp.test_size[0],
        exp.test_size[1],
        device=export_device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["images"],
            output_names=["output"],
            opset_version=opset_version,
        )

    logger.info("Saved ONNX model to {}", output_path)
    return output_path


def export_ckpt_to_fp16_onnx(exp, ckpt_path: Path, output_path: Path, device: str = "cpu") -> Path | None:
    return export_ckpt_to_onnx(exp=exp, ckpt_path=ckpt_path, output_path=output_path, device=device)


def patch_trainer_for_onnx_export() -> None:
    from yolox.core import trainer as trainer_module

    if getattr(trainer_module, "_custom17_best_onnx_patch_applied", False):
        return

    original_after_train = trainer_module.Trainer.after_train

    def patched_after_train(self):
        if self.rank == 0:
            try:
                export_best_ckpt_to_onnx(self)
            except Exception:
                logger.exception("Failed to export best checkpoint to ONNX")
        original_after_train(self)

    trainer_module.Trainer.after_train = patched_after_train
    trainer_module._custom17_best_onnx_patch_applied = True


def patch_trainer_for_balanced_resample_length() -> None:
    from yolox.core import trainer as trainer_module

    if getattr(trainer_module, "_custom17_balanced_len_patch_applied", False):
        return

    original_before_train = trainer_module.Trainer.before_train
    original_train_in_iter = trainer_module.Trainer.train_in_iter
    original_after_iter = trainer_module.Trainer.after_iter

    def patched_before_train(self):
        if os.getenv("CUSTOM17_BALANCED_RESAMPLE", "").strip().lower() in {"1", "true", "yes", "on"}:
            self.exp.balanced_resample = True
            seed_raw = os.getenv("CUSTOM17_BALANCED_RESAMPLE_SEED", "").strip()
            if seed_raw:
                self.exp.balanced_resample_seed = int(seed_raw)
            logger.info(
                "Enabled balanced resampling from env before trainer setup: seed={}",
                getattr(self.exp, "balanced_resample_seed", 42),
            )
        original_before_train(self)
        if not getattr(self.exp, "balanced_resample", False):
            return
        batch_sampler = getattr(self.train_loader, "batch_sampler", None)
        if batch_sampler is None:
            return
        logger.info(
            "Balanced resampling batch sampler in use: {}",
            type(batch_sampler).__name__,
        )
        train_dataset = getattr(self.train_loader, "dataset", None)
        dataset_len = len(train_dataset) if train_dataset is not None else 0
        effective_batch_size = getattr(batch_sampler, "batch_size", None) or getattr(
            self.args, "batch_size", 1
        )
        patched_max_iter = math.ceil(dataset_len / max(int(effective_batch_size), 1))
        self.max_iter = patched_max_iter
        self._custom17_effective_max_iter = patched_max_iter
        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if getattr(self, "use_model_ema", False):
            self.ema_model.updates = self.max_iter * self.start_epoch
        logger.info(
            "Patched max_iter for balanced resampling: {} (dataset_len={}, effective_batch_size={})",
            self.max_iter,
            dataset_len,
            effective_batch_size,
        )

    def patched_train_in_iter(self):
        if not getattr(self.exp, "balanced_resample", False):
            return original_train_in_iter(self)
        effective_max_iter = getattr(self, "_custom17_effective_max_iter", self.max_iter)
        self.max_iter = effective_max_iter
        for self.iter in range(effective_max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def patched_after_iter(self):
        if getattr(self.exp, "balanced_resample", False):
            self.max_iter = getattr(self, "_custom17_effective_max_iter", self.max_iter)
        return original_after_iter(self)

    trainer_module.Trainer.before_train = patched_before_train
    trainer_module.Trainer.train_in_iter = patched_train_in_iter
    trainer_module.Trainer.after_iter = patched_after_iter
    trainer_module._custom17_balanced_len_patch_applied = True
