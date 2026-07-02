#!/usr/bin/env python3
"""Evaluate a custom17 ONNX model with COCO metrics."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = REPO_ROOT / "upstream_yolox"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from custom17.common import CUSTOM17_CLASSES
from yolox.data.data_augment import preproc as preprocess
from yolox.exp import get_exp
from yolox.utils import demo_postprocess, multiclass_nms


def make_parser():
    parser = argparse.ArgumentParser("Custom17 ONNX evaluation")
    parser.add_argument("-m", "--model", required=True, type=str, help="Path to ONNX model.")
    parser.add_argument("-f", "--exp-file", required=True, type=str, help="Experiment file.")
    parser.add_argument("--annotation", type=str, default=None, help="Path to COCO-style val annotation JSON.")
    parser.add_argument("--image-dir", type=str, default=None, help="Validation image directory.")
    parser.add_argument("--conf", type=float, default=0.001, help="Score threshold for mAP evaluation.")
    parser.add_argument("--nms", type=float, default=None, help="NMS threshold. Defaults to exp.nmsthre.")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future use. Current evaluator uses 1.")
    parser.add_argument("--save-json", type=str, default=None, help="Optional output path for detection JSON.")
    return parser


def _build_metric_table(metric_by_class, headers, columns=6):
    num_cols = min(columns, len(metric_by_class) * len(headers))
    result_pair = [x for pair in metric_by_class.items() for x in pair]
    row_pair = zip(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    return tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left"
    )


def per_class_ap50_table(coco_eval: COCOeval, class_names):
    precisions = coco_eval.eval["precision"]
    assert len(class_names) == precisions.shape[2]

    iou_thresholds = np.array(coco_eval.params.iouThrs)
    iou_index = int(np.argmin(np.abs(iou_thresholds - 0.5)))
    if not np.isclose(iou_thresholds[iou_index], 0.5):
        raise ValueError(f"Unable to find IoU=0.50 in COCOeval thresholds: {iou_thresholds}")

    ap50_by_class = {}
    for idx, name in enumerate(class_names):
        precision = precisions[iou_index, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap50 = np.mean(precision) if precision.size else float("nan")
        ap50_by_class[name] = float(ap50 * 100.0)

    return _build_metric_table(ap50_by_class, headers=["class", "AP50"], columns=6)


class ONNXEvaluator:
    def __init__(self, model_path: str, exp, provider: str):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = self.session.get_inputs()[0].type
        self.test_size = exp.test_size

    def infer(self, image: np.ndarray):
        img, ratio = preprocess(image, self.test_size)
        dtype = np.float16 if "float16" in self.input_type else np.float32
        ort_inputs = {self.input_name: img[None, :, :, :].astype(dtype)}
        output = self.session.run(None, ort_inputs)[0]
        predictions = demo_postprocess(output, self.test_size)[0]
        return predictions, ratio


def run_eval(args):
    if args.batch_size != 1:
        logger.warning("batch-size is currently ignored; ONNX eval runs one image at a time.")

    exp = get_exp(args.exp_file, None)
    annotation_path = Path(args.annotation or Path(exp.data_dir) / "annotations" / exp.val_ann)
    image_dir = Path(args.image_dir or Path(exp.data_dir) / "val2017")
    nms_thr = float(args.nms if args.nms is not None else exp.nmsthre)

    coco_gt = COCO(str(annotation_path))
    image_ids = list(sorted(coco_gt.imgs.keys()))
    predictor = ONNXEvaluator(args.model, exp, provider=args.provider)

    data_list = []
    total_infer = 0.0
    total_nms = 0.0

    for index, image_id in enumerate(image_ids, start=1):
        img_info = coco_gt.loadImgs([image_id])[0]
        image_path = image_dir / img_info["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        infer_start = time.perf_counter()
        predictions, ratio = predictor.infer(image)
        infer_end = time.perf_counter()

        nms_start = time.perf_counter()
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=args.conf)
        nms_end = time.perf_counter()

        total_infer += infer_end - infer_start
        total_nms += nms_end - nms_start

        if dets is not None:
            final_boxes = dets[:, :4]
            final_scores = dets[:, 4]
            final_cls_inds = dets[:, 5].astype(int)
            for bbox, score, cls_ind in zip(final_boxes, final_scores, final_cls_inds):
                x0, y0, x1, y1 = bbox.tolist()
                data_list.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(cls_ind),
                        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                        "score": float(score),
                    }
                )

        if index % 100 == 0 or index == len(image_ids):
            logger.info("Evaluated {}/{} images", index, len(image_ids))

    if args.save_json:
        save_json_path = Path(args.save_json)
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        with save_json_path.open("w", encoding="utf-8") as fp:
            json.dump(data_list, fp)
        logger.info("Saved detection JSON to {}", save_json_path)

    if not data_list:
        raise RuntimeError("No detections were produced. Verify the ONNX model, class mapping, and eval thresholds.")

    coco_dt = coco_gt.loadRes(data_list)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    summary_buffer = io.StringIO()
    with contextlib.redirect_stdout(summary_buffer):
        coco_eval.summarize()

    cat_names = [coco_gt.cats[cat_id]["name"] for cat_id in sorted(coco_gt.cats.keys())]

    avg_infer_ms = 1000.0 * total_infer / max(len(image_ids), 1)
    avg_nms_ms = 1000.0 * total_nms / max(len(image_ids), 1)

    print(
        "\n".join(
            [
                f"Average forward time: {avg_infer_ms:.2f} ms",
                f"Average NMS time: {avg_nms_ms:.2f} ms",
                f"Average inference time: {avg_infer_ms + avg_nms_ms:.2f} ms",
                summary_buffer.getvalue().rstrip(),
                "per class AP:",
                _build_metric_table(
                    {
                        name: float(np.mean(coco_eval.eval["precision"][:, :, idx, 0, -1][coco_eval.eval["precision"][:, :, idx, 0, -1] > -1]) * 100.0)
                        if np.any(coco_eval.eval["precision"][:, :, idx, 0, -1] > -1)
                        else float("nan")
                        for idx, name in enumerate(cat_names)
                    },
                    headers=["class", "AP"],
                    columns=6,
                ),
                "per class AP50:",
                per_class_ap50_table(coco_eval, cat_names),
            ]
        )
    )

    print(f"mAP50_95: {coco_eval.stats[0] * 100.0:.3f}")
    print(f"mAP50: {coco_eval.stats[1] * 100.0:.3f}")


def main():
    args = make_parser().parse_args()
    if len(CUSTOM17_CLASSES) != 17:
        raise RuntimeError("custom17 class list is unexpectedly changed.")
    run_eval(args)


if __name__ == "__main__":
    main()
