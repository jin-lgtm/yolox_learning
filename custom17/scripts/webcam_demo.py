#!/usr/bin/env python3
"""Webcam demo for the custom17 YOLOX model."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = REPO_ROOT / "upstream_yolox"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from custom17.runtime_patches import patch_torch_load_for_checkpoints


def make_parser():
    parser = argparse.ArgumentParser("Custom17 YOLOX webcam demo")
    parser.add_argument("-f", "--exp_file", required=True, type=str)
    parser.add_argument("-c", "--ckpt", required=True, type=str)
    parser.add_argument("--camid", type=int, default=0)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--conf", default=0.3, type=float)
    parser.add_argument("--nms", default=0.45, type=float)
    parser.add_argument("--tsize", default=None, type=int)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--legacy", action="store_true", default=False)
    parser.add_argument("--fuse", action="store_true", default=False)
    parser.add_argument("--save_result", action="store_true", default=False)
    return parser


class Predictor:
    def __init__(self, model, exp, cls_names, device="cpu", fp16=False, legacy=False):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, frame):
        img_info = {"id": 0, "file_name": None}
        height, width = frame.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = frame

        ratio = min(self.test_size[0] / frame.shape[0], self.test_size[1] / frame.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(frame, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.confthre,
                self.nmsthre,
                class_agnostic=True,
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        return vis(img, bboxes, scores, cls, cls_conf, self.cls_names)


def main(exp, args):
    patch_torch_load_for_checkpoints()

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()

    logger.info("loading checkpoint from {}", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("Fusing model...")
        model = fuse_model(model)

    cls_names = getattr(exp, "class_names", None)
    if cls_names is None:
        raise ValueError("exp.class_names is required for the webcam demo")

    predictor = Predictor(model, exp, cls_names, args.device, args.fp16, args.legacy)
    cap = cv2.VideoCapture(args.camid)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam camid={args.camid}")

    writer = None
    if args.save_result:
        output_dir = REPO_ROOT / "runs" / "webcam_demo"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(
            str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        logger.info("Saving webcam demo to {}", save_path)

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        outputs, img_info = predictor.inference(frame)
        result = predictor.visual(outputs[0], img_info, predictor.confthre)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            result,
            f"FPS: {fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(result)

        cv2.namedWindow("custom17-webcam", cv2.WINDOW_NORMAL)
        cv2.imshow("custom17-webcam", result)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, None)
    main(exp, args)
