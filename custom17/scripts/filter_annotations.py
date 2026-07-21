#!/usr/bin/env python3
"""Filter COCO-style annotations down to the 17 target classes with 0..16 remapping."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
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
    parser.add_argument("--disable-coco-fallback", action="store_true")
    parser.add_argument("--balance-train", action="store_true")
    parser.add_argument("--balance-target-count", type=int, default=None)
    parser.add_argument("--balance-seed", type=int, default=42)
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
    return normalized


def resolve_image_file_name(file_name: str, source: str, image_root: Path | None = None) -> str:
    normalized = normalize_image_file_name(file_name, source)
    if source != "objects365":
        return normalized

    candidates = [normalized]
    for prefix in ("images/v1/", "images/v2/"):
        if normalized.startswith(prefix):
            candidates.append(normalized[len(prefix):])

    seen = set()
    deduped_candidates = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped_candidates.append(candidate)

    if image_root is not None:
        for candidate in deduped_candidates:
            if (image_root / candidate).exists():
                return candidate

    return deduped_candidates[0]


def collect_existing_image_ids(
    image_id_to_image: Mapping[int, Mapping[str, object]],
    image_root: Path | None,
) -> set[int]:
    if image_root is None:
        return set(image_id_to_image)

    existing_image_ids = set()
    for image_id, image in image_id_to_image.items():
        file_name = str(image.get("file_name", ""))
        if file_name and (image_root / file_name).exists():
            existing_image_ids.add(image_id)
    return existing_image_ids


def class_counter_from_filtered(data: Mapping[str, object]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for ann in data["annotations"]:  # type: ignore[index]
        class_id = int(ann["category_id"])
        counter[CUSTOM17_CLASSES[class_id]] += 1
    return counter


def build_filtered_annotations(
    data: Mapping[str, object],
    source: str,
    allowed_target_category_ids: set[int] | None = None,
    drop_empty_images: bool = False,
    image_root: Path | None = None,
    require_existing_images: bool = False,
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
            copied_image["file_name"] = resolve_image_file_name(
                str(copied_image["file_name"]),
                source,
                image_root=image_root,
            )
        image_id_to_image[int(image["id"])] = copied_image
    existing_image_ids = (
        collect_existing_image_ids(image_id_to_image, image_root)
        if require_existing_images
        else set(image_id_to_image)
    )
    kept_annotations: List[Dict[str, object]] = []
    positive_image_ids = set()
    per_class_counter: Counter[str] = Counter()
    clipped_bbox_count = 0

    next_ann_id = 1
    for ann in annotations:
        source_category_id = int(ann["category_id"])
        if source_category_id not in source_to_target:
            continue

        target_category_id = source_to_target[source_category_id]
        if allowed_target_category_ids is not None and target_category_id not in allowed_target_category_ids:
            continue
        image_id = int(ann["image_id"])
        if image_id not in existing_image_ids:
            continue
        bbox = ann.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        image_meta = image_id_to_image.get(image_id)
        if image_meta is None:
            continue
        img_w = float(image_meta["width"])
        img_h = float(image_meta["height"])
        x, y, width, height = map(float, bbox)
        if width <= 0 or height <= 0:
            continue

        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(img_w, x + width)
        y2 = min(img_h, y + height)
        clipped_width = x2 - x1
        clipped_height = y2 - y1
        if clipped_width <= 0 or clipped_height <= 0:
            continue
        if x1 != x or y1 != y or clipped_width != width or clipped_height != height:
            clipped_bbox_count += 1

        copied = deepcopy(ann)
        copied["id"] = next_ann_id
        copied["category_id"] = target_category_id
        copied["bbox"] = [x1, y1, clipped_width, clipped_height]
        copied["area"] = clipped_width * clipped_height
        kept_annotations.append(copied)
        next_ann_id += 1

        positive_image_ids.add(image_id)
        per_class_counter[CUSTOM17_CLASSES[target_category_id]] += 1

    if drop_empty_images:
        kept_images = [image_id_to_image[image_id] for image_id in sorted(positive_image_ids)]
    else:
        image_ids_to_keep = existing_image_ids if require_existing_images else set(image_id_to_image)
        kept_images = [image_id_to_image[image_id] for image_id in sorted(image_ids_to_keep)]

    filtered = {
        "info": deepcopy(data.get("info", {})),
        "licenses": deepcopy(data.get("licenses", [])),
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": deepcopy(EXPECTED_CATEGORIES),
    }

    print(
        f"[summary] images={len(kept_images)} annotations={len(kept_annotations)} "
        f"drop_empty_images={drop_empty_images} clipped_bboxes={clipped_bbox_count}"
    )
    for class_name in CUSTOM17_CLASSES:
        print(f"  - {class_name:12s}: {per_class_counter[class_name]}")
    return filtered


def merge_filtered_annotations(
    primary: Mapping[str, object],
    fallback: Mapping[str, object],
) -> Dict[str, object]:
    merged_images = [deepcopy(image) for image in primary["images"]]  # type: ignore[index]
    merged_annotations = [deepcopy(ann) for ann in primary["annotations"]]  # type: ignore[index]

    next_image_id = max((int(image["id"]) for image in merged_images), default=0) + 1
    next_ann_id = max((int(ann["id"]) for ann in merged_annotations), default=0) + 1

    image_id_remap: Dict[int, int] = {}
    for image in fallback["images"]:  # type: ignore[index]
        copied_image = deepcopy(image)
        old_image_id = int(copied_image["id"])
        copied_image["id"] = next_image_id
        image_id_remap[old_image_id] = next_image_id
        merged_images.append(copied_image)
        next_image_id += 1

    for ann in fallback["annotations"]:  # type: ignore[index]
        copied_ann = deepcopy(ann)
        copied_ann["id"] = next_ann_id
        copied_ann["image_id"] = image_id_remap[int(copied_ann["image_id"])]
        merged_annotations.append(copied_ann)
        next_ann_id += 1

    return {
        "info": deepcopy(primary.get("info", {})),
        "licenses": deepcopy(primary.get("licenses", [])),
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": deepcopy(EXPECTED_CATEGORIES),
    }


def maybe_apply_coco_fallback(
    filtered: Mapping[str, object],
    dataset_root: Path,
    split: str,
    disable_coco_fallback: bool,
) -> Dict[str, object]:
    if disable_coco_fallback:
        return dict(filtered)

    class_counter = class_counter_from_filtered(filtered)
    missing_class_ids = {
        idx for idx, class_name in enumerate(CUSTOM17_CLASSES) if class_counter[class_name] == 0
    }
    if not missing_class_ids:
        return dict(filtered)

    coco_raw_path = dataset_root / "raw_annotations" / (
        "instances_train2017.json" if split == "train" else "instances_val2017.json"
    )
    if not coco_raw_path.exists():
        print(
            f"[warn] Missing COCO fallback annotation for split={split}: {coco_raw_path}. "
            f"Classes with zero instances remain missing: {[CUSTOM17_CLASSES[idx] for idx in sorted(missing_class_ids)]}"
        )
        return dict(filtered)

    print(
        f"[fallback] split={split} source=coco classes="
        + ", ".join(CUSTOM17_CLASSES[idx] for idx in sorted(missing_class_ids))
    )
    image_root = dataset_root / ("train2017" if split == "train" else "val2017")
    coco_filtered = build_filtered_annotations(
        load_json(coco_raw_path),
        source="coco",
        allowed_target_category_ids=missing_class_ids,
        drop_empty_images=True,
        image_root=image_root,
        require_existing_images=True,
    )
    if not coco_filtered["annotations"]:  # type: ignore[index]
        print(f"[fallback-skip] split={split} no matching COCO images found under {image_root}")
        return dict(filtered)
    return merge_filtered_annotations(filtered, coco_filtered)


def balance_train_subset(
    filtered: Mapping[str, object],
    target_count: int | None = None,
    seed: int = 42,
) -> Dict[str, object]:
    images: Sequence[Mapping[str, object]] = filtered["images"]  # type: ignore[assignment]
    annotations: Sequence[Mapping[str, object]] = filtered["annotations"]  # type: ignore[assignment]
    if not images or not annotations:
        print("[balance] skip because filtered dataset is empty")
        return dict(filtered)

    original_counts = class_counter_from_filtered(filtered)
    positive_classes = [name for name in CUSTOM17_CLASSES if original_counts[name] > 0]
    if not positive_classes:
        print("[balance] skip because no positive classes were found")
        return dict(filtered)

    resolved_target = target_count or min(original_counts[name] for name in positive_classes)
    if resolved_target <= 0:
        print("[balance] skip because resolved target count is not positive")
        return dict(filtered)

    image_id_to_image = {int(image["id"]): deepcopy(image) for image in images}
    anns_by_image: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    class_ids_by_image: dict[int, set[int]] = defaultdict(set)
    for ann in annotations:
        image_id = int(ann["image_id"])
        anns_by_image[image_id].append(ann)
        class_ids_by_image[image_id].add(int(ann["category_id"]))

    rng = random.Random(seed)
    image_ids = list(anns_by_image)
    rng.shuffle(image_ids)
    image_ids.sort(
        key=lambda image_id: (
            min(original_counts[CUSTOM17_CLASSES[class_id]] for class_id in class_ids_by_image[image_id]),
            -len(class_ids_by_image[image_id]),
        )
    )

    selected_counts: Counter[str] = Counter()
    selected_image_ids: list[int] = []
    selected_annotations: list[Dict[str, object]] = []

    def needs_more_examples(image_id: int) -> bool:
        return any(
            selected_counts[CUSTOM17_CLASSES[int(ann["category_id"])]] < resolved_target
            for ann in anns_by_image[image_id]
        )

    for image_id in image_ids:
        if not needs_more_examples(image_id):
            continue
        selected_image_ids.append(image_id)
        for ann in anns_by_image[image_id]:
            copied_ann = deepcopy(ann)
            selected_annotations.append(copied_ann)
            selected_counts[CUSTOM17_CLASSES[int(copied_ann["category_id"])]] += 1
        if all(selected_counts[class_name] >= resolved_target for class_name in positive_classes):
            break

    selected_images = [image_id_to_image[image_id] for image_id in selected_image_ids]
    balanced = {
        "info": deepcopy(filtered.get("info", {})),
        "licenses": deepcopy(filtered.get("licenses", [])),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": deepcopy(filtered.get("categories", EXPECTED_CATEGORIES)),
    }

    print(
        f"[balance] target_count={resolved_target} seed={seed} "
        f"images={len(selected_images)} annotations={len(selected_annotations)}"
    )
    final_counts = class_counter_from_filtered(balanced)
    for class_name in CUSTOM17_CLASSES:
        if original_counts[class_name] == 0:
            continue
        print(
            f"  - {class_name:12s}: original={original_counts[class_name]} balanced={final_counts[class_name]}"
        )
    under_target = [class_name for class_name in positive_classes if final_counts[class_name] < resolved_target]
    if under_target:
        print(
            "[balance-warn] unable to hit target for classes: "
            + ", ".join(
                f"{class_name}({final_counts[class_name]}/{resolved_target})" for class_name in under_target
            )
        )
    return balanced


def main() -> None:
    args = parse_args()
    train_input, val_input, output_dir = resolve_default_annotation_paths(args)
    dataset_root = args.dataset_root.resolve()
    train_image_root = dataset_root / "train2017"
    val_image_root = dataset_root / "val2017"

    train_data = load_json(train_input)
    val_data = load_json(val_input)

    train_filtered = build_filtered_annotations(
        train_data,
        source=args.source,
        drop_empty_images=args.drop_empty_images,
        image_root=train_image_root,
        require_existing_images=(args.source == "objects365"),
    )
    val_filtered = build_filtered_annotations(
        val_data,
        source=args.source,
        drop_empty_images=args.drop_empty_images,
        image_root=val_image_root,
        require_existing_images=(args.source == "objects365"),
    )

    if args.source == "objects365":
        train_filtered = maybe_apply_coco_fallback(
            train_filtered,
            dataset_root=dataset_root,
            split="train",
            disable_coco_fallback=args.disable_coco_fallback,
        )
        val_filtered = maybe_apply_coco_fallback(
            val_filtered,
            dataset_root=dataset_root,
            split="val",
            disable_coco_fallback=args.disable_coco_fallback,
        )

    if args.balance_train:
        train_filtered = balance_train_subset(
            train_filtered,
            target_count=args.balance_target_count,
            seed=args.balance_seed,
        )

    dump_json(train_filtered, output_dir / "train.json")
    dump_json(val_filtered, output_dir / "val.json")


if __name__ == "__main__":
    main()
