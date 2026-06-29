#!/usr/bin/env python3
"""Render sample bounding boxes from filtered custom17 annotations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom17.common import CUSTOM17_CLASSES, categories_match_expected


COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/custom17_vis"))
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def draw_label(image, text: str, x1: int, y1: int, color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_top = max(y1 - text_h - baseline - 4, 0)
    cv2.rectangle(image, (x1, y_top), (x1 + text_w + 6, y_top + text_h + baseline + 6), color, -1)
    cv2.putText(image, text, (x1 + 3, y_top + text_h + 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    payload = load_json(args.annotation.resolve())
    categories = payload.get("categories", [])
    if not isinstance(categories, Sequence) or not categories_match_expected(categories):
        raise SystemExit("Annotation categories are not in custom17 id/name order.")

    images = payload["images"]
    annotations = payload["annotations"]

    ann_by_image: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for ann in annotations:
        ann_by_image[int(ann["image_id"])].append(ann)

    candidate_images = [image for image in images if ann_by_image.get(int(image["id"]))]
    random.seed(args.seed)
    random.shuffle(candidate_images)
    selected_images = candidate_images[: args.num_images]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_meta in selected_images:
        image_id = int(image_meta["id"])
        file_name = str(image_meta["file_name"])
        image_path = (args.image_dir / file_name).resolve()
        canvas = cv2.imread(str(image_path))
        if canvas is None:
            print(f"[warn] missing image: {image_path}")
            continue

        for ann in ann_by_image[image_id]:
            category_id = int(ann["category_id"])
            class_name = CUSTOM17_CLASSES[category_id]
            color = COLORS[category_id]
            x, y, w, h = map(float, ann["bbox"])
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            draw_label(canvas, f"{category_id}:{class_name}", x1, y1, color)

        target_path = output_dir / file_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target_path), canvas)
        print(f"[write] {target_path}")


if __name__ == "__main__":
    main()
