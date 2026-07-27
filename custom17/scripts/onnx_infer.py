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
from yolox.utils import mkdir, multiclass_nms, vis

IMAGE_EXT = {".jpg", ".jpeg", ".webp", ".bmp", ".png"}


def concrete_input_size(shape) -> tuple[int, int]:
    dims = [str(dim) for dim in shape]
    if len(dims) != 4:
        raise ValueError(f"Expected 4D ONNX input shape, got: {dims}")
    if not dims[2].isdigit() or not dims[3].isdigit():
        raise ValueError(f"Dynamic spatial input shape is not supported: {dims}")
    return int(dims[2]), int(dims[3])


def infer_strides_from_output_count(img_size, output_count: int):
    candidate_stride_sets = (
        [8],
        [16],
        [32],
        [64],
        [8, 16],
        [8, 16, 32],
        [16, 32],
        [32, 64],
        [8, 16, 32, 64],
    )
    for strides in candidate_stride_sets:
        expected = sum((img_size[0] // stride) * (img_size[1] // stride) for stride in strides)
        if expected == output_count:
            return strides
    raise ValueError(
        f"Unable to infer strides for output_count={output_count} and img_size={img_size}. "
        f"Known candidates: {candidate_stride_sets}"
    )


def decode_outputs_with_strides(outputs: np.ndarray, img_size, strides) -> np.ndarray:
    grids = []
    expanded_strides = []
    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, axis=1)
    expanded_strides = np.concatenate(expanded_strides, axis=1)
    if outputs.shape[1] != grids.shape[1]:
        raise ValueError(
            f"ONNX output anchor count mismatch: got {outputs.shape[1]}, "
            f"expected {grids.shape[1]} for strides={strides} and img_size={img_size}"
        )
    outputs = outputs.copy()
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def make_parser():
    parser = argparse.ArgumentParser("Custom17 ONNX inference")
    parser.add_argument("demo", choices=("image", "video", "webcam"), default="image")
    parser.add_argument("-m", "--model", required=True, type=str, help="Path to ONNX model.")
    parser.add_argument("-f", "--exp_file", required=True, type=str, help="Experiment file for class names and test size.")
    parser.add_argument("--model-b", type=str, default=None, help="Optional second ONNX model for side-by-side comparison.")
    parser.add_argument("--exp-file-b", type=str, default=None, help="Optional second experiment file. Defaults to --exp-file.")
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
    def __init__(self, model_path: str, exp, provider: str = "cpu", name: str | None = None):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.model_path = model_path
        self.name = name or Path(model_path).stem
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = self.session.get_inputs()[0].type
        self.test_size = concrete_input_size(self.session.get_inputs()[0].shape)
        self.head_strides = getattr(exp, "head_strides", [8, 16, 32])
        self.class_names = getattr(exp, "class_names", None)
        if self.class_names is None:
            raise ValueError("exp.class_names is required")
        self.class_names = tuple(self.class_names)
        self._resolved_head_strides = None
        self._resolved_class_names = self.class_names

    def _ensure_class_names(self, num_classes: int):
        if len(self._resolved_class_names) >= num_classes:
            return
        padded = list(self.class_names)
        padded.extend(f"cls_{idx}" for idx in range(len(padded), num_classes))
        self._resolved_class_names = tuple(padded)

    def inference(self, frame):
        img, ratio = preprocess(frame, self.test_size)
        dtype = np.float16 if "float16" in self.input_type else np.float32
        ort_inputs = {self.input_name: img[None, :, :, :].astype(dtype)}
        output = self.session.run(None, ort_inputs)[0]
        self._ensure_class_names(int(output.shape[2] - 5))
        if self._resolved_head_strides is None:
            self._resolved_head_strides = infer_strides_from_output_count(self.test_size, int(output.shape[1]))
        predictions = decode_outputs_with_strides(output, self.test_size, self._resolved_head_strides)[0]
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
        return vis(
            frame,
            final_boxes,
            final_scores,
            final_cls_inds,
            conf=score_thr,
            class_names=self._resolved_class_names,
        )


def annotate_panel(
    frame,
    title: str,
    subtitle: str | None = None,
    fps: float | None = None,
):
    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 60), (25, 25, 25), thickness=-1)
    cv2.putText(
        canvas,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    info_parts = []
    if subtitle:
        info_parts.append(subtitle)
    if fps is not None:
        info_parts.append(f"FPS: {fps:.1f}")
    if info_parts:
        cv2.putText(
            canvas,
            " | ".join(info_parts),
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 255, 180),
            2,
            cv2.LINE_AA,
        )
    return canvas


def render_predictor_output(frame, predictor, score_thr: float, nms_thr: float):
    predictions, ratio = predictor.inference(frame)
    return predictor.visual(frame.copy(), predictions, ratio, score_thr, nms_thr)


def predictor_subtitle(predictor) -> str:
    return f"model={predictor.name} | input={predictor.test_size[0]}x{predictor.test_size[1]}"


def stack_results(results):
    if len(results) == 1:
        return results[0]
    height = max(result.shape[0] for result in results)
    normalized = []
    for result in results:
        if result.shape[0] == height:
            normalized.append(result)
            continue
        width = int(round(result.shape[1] * (height / result.shape[0])))
        normalized.append(cv2.resize(result, (width, height)))
    return cv2.hconcat(normalized)


def image_demo(predictors, args):
    if os.path.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]

    if args.save_result:
        mkdir(args.output_dir)

    for image_name in files:
        origin_img = cv2.imread(image_name)
        panels = []
        for predictor in predictors:
            result = render_predictor_output(origin_img, predictor, args.score_thr, args.nms_thr)
            panels.append(
                annotate_panel(
                    result,
                    "Detection",
                    subtitle=predictor_subtitle(predictor),
                )
            )
        result = stack_results(panels)
        if args.save_result:
            output_path = os.path.join(args.output_dir, os.path.basename(image_name))
            cv2.imwrite(output_path, result)
        cv2.imshow("custom17-onnx", result)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def stream_demo(predictors, args):
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
        width *= len(predictors)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prev_times = [time.time() for _ in predictors]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        panels = []
        for idx, predictor in enumerate(predictors):
            result = render_predictor_output(frame, predictor, args.score_thr, args.nms_thr)
            now = time.time()
            fps = 1.0 / max(now - prev_times[idx], 1e-6)
            prev_times[idx] = now
            panels.append(
                annotate_panel(
                    result,
                    "Detection",
                    subtitle=predictor_subtitle(predictor),
                    fps=fps,
                )
            )
        result = stack_results(panels)
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
    predictors = [ONNXPredictor(args.model, exp, provider=args.provider)]
    if args.model_b:
        exp_b = get_exp(args.exp_file_b or args.exp_file, None)
        predictors.append(ONNXPredictor(args.model_b, exp_b, provider=args.provider))
    if args.demo == "image":
        image_demo(predictors, args)
    else:
        stream_demo(predictors, args)


if __name__ == "__main__":
    main()
