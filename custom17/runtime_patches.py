#!/usr/bin/env python3
"""Runtime patches for YOLOX evaluation output."""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import tempfile
from typing import Sequence

import numpy as np
from loguru import logger
from tabulate import tabulate


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
