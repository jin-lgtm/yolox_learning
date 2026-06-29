#!/usr/bin/env python3
"""Validate remapped custom17 COCO annotations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom17.common import CUSTOM17_CLASSES, categories_match_expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    payload = load_json(args.annotation.resolve())

    categories = payload.get("categories", [])
    if not isinstance(categories, Sequence) or not categories_match_expected(categories):
        raise SystemExit(
            "categories field does not match the required 0..16 / class-name order for custom17"
        )

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    if not isinstance(images, Sequence) or not isinstance(annotations, Sequence):
        raise SystemExit("images or annotations field is missing or invalid")

    image_sizes = {
        int(image["id"]): (int(image["width"]), int(image["height"]))
        for image in images
    }
    per_class = Counter()
    per_image = Counter()
    out_of_bounds = defaultdict(int)
    empty_bbox_count = 0

    for ann in annotations:
        ann_id = int(ann["id"])
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        bbox = ann["bbox"]
        if image_id not in image_sizes:
            raise SystemExit(f"annotation {ann_id} references missing image_id={image_id}")
        if not 0 <= category_id < len(CUSTOM17_CLASSES):
            raise SystemExit(f"annotation {ann_id} has invalid category_id={category_id}")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise SystemExit(f"annotation {ann_id} has invalid bbox format: {bbox}")

        x, y, w, h = map(float, bbox)
        if w <= 0 or h <= 0:
            empty_bbox_count += 1
            continue

        img_w, img_h = image_sizes[image_id]
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            out_of_bounds[CUSTOM17_CLASSES[category_id]] += 1

        per_class[CUSTOM17_CLASSES[category_id]] += 1
        per_image[image_id] += 1

    print(f"[ok] {args.annotation}")
    print(f"     images={len(images)} annotations={len(annotations)}")
    print(f"     empty_bbox_count={empty_bbox_count}")
    print(f"     images_with_targets={sum(1 for count in per_image.values() if count > 0)}")
    print(f"     images_without_targets={len(images) - len(per_image)}")
    print("     per-class counts:")
    for class_name in CUSTOM17_CLASSES:
        print(f"       {class_name:12s}: {per_class[class_name]}")

    if out_of_bounds:
        print("     bbox boundary warnings:")
        for class_name in CUSTOM17_CLASSES:
            if out_of_bounds[class_name]:
                print(f"       {class_name:12s}: {out_of_bounds[class_name]}")
    else:
        print("     bbox boundary warnings: none")


if __name__ == "__main__":
    main()
