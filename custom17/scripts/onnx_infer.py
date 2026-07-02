#!/usr/bin/env python3
"""Inference with an exported custom17 ONNX model."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = REPO_ROOT / "upstream_yolox"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from yolox.data.data_augment import preproc as preprocess
from yolox.exp import get_exp
from yolox.utils import demo_postprocess, mkdir, multiclass_nms, vis

IMAGE_EXT = {".jpg", ".jpeg", ".webp", ".bmp", ".png"}


def make_parser():
    parser = argparse.ArgumentParser("Custom17 ONNX inference")
    parser.add_argument("demo", choices=("image", "video", "webcam"), default="image")
    parser.add_argument("-m", "--model", required=True, type=str, help="Path to ONNX model.")
    parser.add_argument("-f", "--exp_file", required=True, type=str, help="Experiment file for class names and test size.")
    parser.add_argument("--path", default="./assets/dog.jpg", help="Path to image dir or video file.")
    parser.add_argument("--camid", type=int, default=0, help="Webcam camera id.")
    parser.add_argument("-o", "--output-dir", default="runs/onnx_infer", type=str)
    parser.add_argument("-s", "--score-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.45)
    parser.add_argument("--save-result", action="store_true", default=False)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, _, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1].lower()
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return sorted(image_names)


class ONNXPredictor:
    def __init__(self, model_path: str, exp, provider: str = "cpu"):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = self.session.get_inputs()[0].type
        self.test_size = exp.test_size
        self.class_names = getattr(exp, "class_names", None)
        if self.class_names is None:
            raise ValueError("exp.class_names is required")

    def inference(self, frame):
        img, ratio = preprocess(frame, self.test_size)
        dtype = np.float16 if "float16" in self.input_type else np.float32
        ort_inputs = {self.input_name: img[None, :, :, :].astype(dtype)}
        output = self.session.run(None, ort_inputs)[0]
        predictions = demo_postprocess(output, self.test_size)[0]
        return predictions, ratio

    def visual(self, frame, predictions, ratio, score_thr: float, nms_thr: float):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)
        if dets is None:
            return frame
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        return vis(frame, final_boxes, final_scores, final_cls_inds, conf=score_thr, class_names=self.class_names)


def image_demo(predictor, args):
    if os.path.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]

    if args.save_result:
        mkdir(args.output_dir)

    for image_name in files:
        origin_img = cv2.imread(image_name)
        predictions, ratio = predictor.inference(origin_img)
        result = predictor.visual(origin_img, predictions, ratio, args.score_thr, args.nms_thr)
        if args.save_result:
            output_path = os.path.join(args.output_dir, os.path.basename(image_name))
            cv2.imwrite(output_path, result)
        cv2.imshow("custom17-onnx", result)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def stream_demo(predictor, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    if not cap.isOpened():
        raise RuntimeError("Unable to open input stream")

    writer = None
    if args.save_result:
        mkdir(args.output_dir)
        save_path = os.path.join(
            args.output_dir,
            os.path.basename(args.path) if args.demo == "video" else f"webcam_{int(time.time())}.mp4",
        )
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        predictions, ratio = predictor.inference(frame)
        result = predictor.visual(frame, predictions, ratio, args.score_thr, args.nms_thr)
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(result, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        if writer is not None:
            writer.write(result)
        cv2.imshow("custom17-onnx", result)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, None)
    predictor = ONNXPredictor(args.model, exp, provider=args.provider)
    if args.demo == "image":
        image_demo(predictor, args)
    else:
        stream_demo(predictor, args)


if __name__ == "__main__":
    main()
