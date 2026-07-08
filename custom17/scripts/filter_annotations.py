#!/usr/bin/env python3
"""Filter COCO-style annotations down to the 17 target classes with 0..16 remapping."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom17.common import (
    CLASS_TO_NEW_ID,
    CUSTOM17_CLASSES,
    EXPECTED_CATEGORIES,
    build_source_category_remap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("coco", "objects365"), default="coco")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/custom17"))
    parser.add_argument(
        "--train-input",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--val-input",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--drop-empty-images", action="store_true")
    return parser.parse_args()


def resolve_default_annotation_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_root = args.dataset_root.resolve()
    output_dir = (args.output_dir or (dataset_root / "annotations")).resolve()
    if args.train_input is not None and args.val_input is not None:
        return args.train_input.resolve(), args.val_input.resolve(), output_dir

    if args.source == "coco":
        train_input = dataset_root / "raw_annotations" / "instances_train2017.json"
        val_input = dataset_root / "raw_annotations" / "instances_val2017.json"
    else:
        train_input = dataset_root / "raw_annotations" / "objects365_train.json"
        val_input = dataset_root / "raw_annotations" / "objects365_val.json"

    return train_input.resolve(), val_input.resolve(), output_dir


def load_json(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[write] {path}")


def normalize_image_file_name(file_name: str, source: str) -> str:
    if source != "objects365":
        return file_name

    normalized = file_name.replace("\\", "/").lstrip("./")
    for prefix in ("images/v1/", "images/v2/"):
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
    return normalized


def build_filtered_annotations(
    data: Mapping[str, object],
    source: str,
    drop_empty_images: bool = False,
) -> Dict[str, object]:
    images: Sequence[Mapping[str, object]] = data["images"]  # type: ignore[assignment]
    annotations: Iterable[Mapping[str, object]] = data["annotations"]  # type: ignore[assignment]
    categories: Sequence[Mapping[str, object]] = data["categories"]  # type: ignore[assignment]

    source_to_target = build_source_category_remap(categories)
    missing = [name for name in CUSTOM17_CLASSES if name not in CLASS_TO_NEW_ID]
    if missing:
        raise ValueError(f"Internal class config is invalid, missing names: {missing}")

    image_id_to_image = {}
    for image in images:
        copied_image = deepcopy(image)
        if "file_name" in copied_image:
            copied_image["file_name"] = normalize_image_file_name(str(copied_image["file_name"]), source)
        image_id_to_image[int(image["id"])] = copied_image
    kept_annotations: List[Dict[str, object]] = []
    positive_image_ids = set()
    per_class_counter: Counter[str] = Counter()

    next_ann_id = 1
    for ann in annotations:
        source_category_id = int(ann["category_id"])
        if source_category_id not in source_to_target:
            continue

        bbox = ann.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        _, _, width, height = bbox
        if float(width) <= 0 or float(height) <= 0:
            continue

        target_category_id = source_to_target[source_category_id]
        copied = deepcopy(ann)
        copied["id"] = next_ann_id
        copied["category_id"] = target_category_id
        copied["area"] = float(width) * float(height)
        kept_annotations.append(copied)
        next_ann_id += 1

        image_id = int(copied["image_id"])
        positive_image_ids.add(image_id)
        per_class_counter[CUSTOM17_CLASSES[target_category_id]] += 1

    if drop_empty_images:
        kept_images = [image_id_to_image[image_id] for image_id in sorted(positive_image_ids)]
    else:
        kept_images = [image_id_to_image[image_id] for image_id in sorted(image_id_to_image)]

    filtered = {
        "info": deepcopy(data.get("info", {})),
        "licenses": deepcopy(data.get("licenses", [])),
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": deepcopy(EXPECTED_CATEGORIES),
    }

    print(
        f"[summary] images={len(kept_images)} annotations={len(kept_annotations)} "
        f"drop_empty_images={drop_empty_images}"
    )
    for class_name in CUSTOM17_CLASSES:
        print(f"  - {class_name:12s}: {per_class_counter[class_name]}")
    return filtered


def main() -> None:
    args = parse_args()
    train_input, val_input, output_dir = resolve_default_annotation_paths(args)

    train_data = load_json(train_input)
    val_data = load_json(val_input)

    train_filtered = build_filtered_annotations(
        train_data,
        source=args.source,
        drop_empty_images=args.drop_empty_images,
    )
    val_filtered = build_filtered_annotations(
        val_data,
        source=args.source,
        drop_empty_images=args.drop_empty_images,
    )

    dump_json(train_filtered, output_dir / "train.json")
    dump_json(val_filtered, output_dir / "val.json")


if __name__ == "__main__":
    main()
