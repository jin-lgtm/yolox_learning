#!/usr/bin/env python3
"""Download COCO or Objects365 assets for custom17 training."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

YOLOX_TINY_COCO_CKPT_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("coco", "objects365"), default="coco")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/custom17"))
    parser.add_argument("--downloads-dir", type=Path, default=None)
    parser.add_argument("--weights-dir", type=Path, default=Path("pretrained_models"))
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-pretrained", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-image-src", type=str, default=None)
    parser.add_argument("--val-image-src", type=str, default=None)
    parser.add_argument("--train-ann-src", type=str, default=None)
    parser.add_argument("--val-ann-src", type=str, default=None)
    parser.add_argument("--train-image-strip-prefix", type=str, default="")
    parser.add_argument("--val-image-strip-prefix", type=str, default="")
    parser.add_argument("--train-ann-member", type=str, default=None)
    parser.add_argument("--val-ann-member", type=str, default=None)
    parser.add_argument("--train-ann-output-name", type=str, default=None)
    parser.add_argument("--val-ann-output-name", type=str, default=None)
    return parser.parse_args()


def download_file(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"[skip] {destination} already exists")
        return

    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        percent = downloaded * 100.0 / total_size
        sys.stdout.write(
            f"\r[download] {destination.name}: {downloaded / 1e6:.1f}MB / {total_size / 1e6:.1f}MB ({percent:5.1f}%)"
        )
        sys.stdout.flush()

    print(f"[download] {url}")
    urllib.request.urlretrieve(url, destination, reporthook=report)
    sys.stdout.write("\n")


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def get_source_value(cli_value: str | None, env_key: str) -> str | None:
    return cli_value or os.environ.get(env_key)


def stage_source(
    src: str,
    downloads_dir: Path,
    default_filename: str,
    force: bool = False,
) -> Path:
    if is_url(src):
        destination = downloads_dir / default_filename
        download_file(src, destination, force=force)
        return destination

    local_path = Path(src).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Source not found: {local_path}")
    print(f"[use-local] {local_path}")
    return local_path


def normalize_member_path(member_name: str, strip_prefix: str = "") -> Path | None:
    normalized = member_name.lstrip("./")
    if normalized.endswith("/"):
        normalized = normalized[:-1]
    strip_prefix = strip_prefix.strip("/")
    if strip_prefix:
        prefix = strip_prefix + "/"
        if normalized == strip_prefix:
            return None
        if not normalized.startswith(prefix):
            return None
        normalized = normalized[len(prefix):]
    if not normalized:
        return None
    return Path(normalized)


def extract_zip_tree(zip_path: Path, output_dir: Path, strip_prefix: str = "", force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            relative_path = normalize_member_path(member, strip_prefix=strip_prefix)
            if relative_path is None:
                continue
            target_path = output_dir / relative_path
            if member.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() and not force:
                continue
            with zf.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        print(f"[extract] {zip_path.name} -> {output_dir}")


def extract_tar_tree(
    tar_path: Path,
    output_dir: Path,
    strip_prefix: str = "",
    force: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            relative_path = normalize_member_path(member.name, strip_prefix=strip_prefix)
            if relative_path is None:
                continue
            target_path = output_dir / relative_path
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() and not force:
                continue
            with extracted as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"[extract] {tar_path.name} -> {output_dir}")


def extract_archive_tree(
    archive_path: Path,
    output_dir: Path,
    strip_prefix: str = "",
    force: bool = False,
) -> None:
    if archive_path.suffix == ".zip":
        extract_zip_tree(archive_path, output_dir, strip_prefix=strip_prefix, force=force)
        return
    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        extract_tar_tree(archive_path, output_dir, strip_prefix=strip_prefix, force=force)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def extract_archive_member(
    archive_path: Path,
    output_path: Path,
    member_name: str | None = None,
    force: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"[skip] {output_path} already exists")
        return

    if archive_path.suffix == ".json":
        shutil.copy2(archive_path, output_path)
        print(f"[copy] {archive_path} -> {output_path}")
        return

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            selected = member_name
            if selected is None:
                json_members = [name for name in zf.namelist() if name.endswith(".json")]
                if len(json_members) != 1:
                    raise ValueError(
                        f"Unable to infer a unique json member from {archive_path}; "
                        f"please pass --train-ann-member/--val-ann-member."
                    )
                selected = json_members[0]
            with zf.open(selected) as src, output_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"[extract] {selected} -> {output_path}")
        return

    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path) as tf:
            selected = member_name
            if selected is None:
                json_members = [member.name for member in tf.getmembers() if member.name.endswith(".json")]
                if len(json_members) != 1:
                    raise ValueError(
                        f"Unable to infer a unique json member from {archive_path}; "
                        f"please pass --train-ann-member/--val-ann-member."
                    )
                selected = json_members[0]
            extracted = tf.extractfile(selected)
            if extracted is None:
                raise FileNotFoundError(f"Unable to extract member {selected} from {archive_path}")
            with extracted as src, output_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"[extract] {selected} -> {output_path}")
        return

    raise ValueError(f"Unsupported annotation source format: {archive_path}")


def default_objects365_output_name(split: str) -> str:
    return f"objects365_{split}.json"


def prepare_coco_assets(dataset_root: Path, downloads_dir: Path, force: bool, skip_images: bool) -> None:
    raw_annotations_dir = dataset_root / "raw_annotations"
    raw_annotations_dir.mkdir(parents=True, exist_ok=True)

    annotations_zip = downloads_dir / "annotations_trainval2017.zip"
    download_file(COCO_URLS["annotations"], annotations_zip, force=force)
    extract_archive_member(
        annotations_zip,
        raw_annotations_dir / "instances_train2017.json",
        member_name="annotations/instances_train2017.json",
        force=force,
    )
    extract_archive_member(
        annotations_zip,
        raw_annotations_dir / "instances_val2017.json",
        member_name="annotations/instances_val2017.json",
        force=force,
    )

    if skip_images:
        return

    train_zip = downloads_dir / "train2017.zip"
    val_zip = downloads_dir / "val2017.zip"
    download_file(COCO_URLS["train_images"], train_zip, force=force)
    download_file(COCO_URLS["val_images"], val_zip, force=force)
    extract_archive_tree(train_zip, dataset_root, strip_prefix="", force=force)
    extract_archive_tree(val_zip, dataset_root, strip_prefix="", force=force)


def prepare_objects365_assets(args: argparse.Namespace, dataset_root: Path, downloads_dir: Path) -> None:
    raw_annotations_dir = dataset_root / "raw_annotations"
    raw_annotations_dir.mkdir(parents=True, exist_ok=True)

    train_image_src = get_source_value(args.train_image_src, "OBJECTS365_TRAIN_IMAGE_SRC")
    val_image_src = get_source_value(args.val_image_src, "OBJECTS365_VAL_IMAGE_SRC")
    train_ann_src = get_source_value(args.train_ann_src, "OBJECTS365_TRAIN_ANN_SRC")
    val_ann_src = get_source_value(args.val_ann_src, "OBJECTS365_VAL_ANN_SRC")

    missing = []
    if train_ann_src is None:
        missing.append("--train-ann-src or OBJECTS365_TRAIN_ANN_SRC")
    if val_ann_src is None:
        missing.append("--val-ann-src or OBJECTS365_VAL_ANN_SRC")
    if not args.skip_images:
        if train_image_src is None:
            missing.append("--train-image-src or OBJECTS365_TRAIN_IMAGE_SRC")
        if val_image_src is None:
            missing.append("--val-image-src or OBJECTS365_VAL_IMAGE_SRC")
    if missing:
        raise ValueError(
            "Objects365 source selected but required sources are missing:\n- " + "\n- ".join(missing)
        )

    train_ann_path = stage_source(
        train_ann_src,
        downloads_dir,
        default_filename=Path(urlparse(train_ann_src).path).name or "objects365_train_ann",
        force=args.force,
    )
    val_ann_path = stage_source(
        val_ann_src,
        downloads_dir,
        default_filename=Path(urlparse(val_ann_src).path).name or "objects365_val_ann",
        force=args.force,
    )
    extract_archive_member(
        train_ann_path,
        raw_annotations_dir / (args.train_ann_output_name or default_objects365_output_name("train")),
        member_name=args.train_ann_member,
        force=args.force,
    )
    extract_archive_member(
        val_ann_path,
        raw_annotations_dir / (args.val_ann_output_name or default_objects365_output_name("val")),
        member_name=args.val_ann_member,
        force=args.force,
    )

    if args.skip_images:
        return

    train_image_path = stage_source(
        train_image_src,
        downloads_dir,
        default_filename=Path(urlparse(train_image_src).path).name or "objects365_train_images",
        force=args.force,
    )
    val_image_path = stage_source(
        val_image_src,
        downloads_dir,
        default_filename=Path(urlparse(val_image_src).path).name or "objects365_val_images",
        force=args.force,
    )
    extract_archive_tree(
        train_image_path,
        dataset_root / "train2017",
        strip_prefix=args.train_image_strip_prefix,
        force=args.force,
    )
    extract_archive_tree(
        val_image_path,
        dataset_root / "val2017",
        strip_prefix=args.val_image_strip_prefix,
        force=args.force,
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    downloads_dir = (args.downloads_dir or dataset_root / "_downloads").resolve()
    weights_dir = args.weights_dir.resolve()

    dataset_root.mkdir(parents=True, exist_ok=True)
    if args.source == "coco":
        prepare_coco_assets(dataset_root, downloads_dir, force=args.force, skip_images=args.skip_images)
    else:
        prepare_objects365_assets(args, dataset_root, downloads_dir)

    if not args.skip_pretrained:
        ckpt_path = weights_dir / "yolox_tiny.pth"
        download_file(YOLOX_TINY_COCO_CKPT_URL, ckpt_path, force=args.force)

    print(f"[done] Dataset assets are ready for source={args.source}.")
    print(f"        raw annotations: {dataset_root / 'raw_annotations'}")
    print(f"        train images:    {dataset_root / 'train2017'}")
    print(f"        val images:      {dataset_root / 'val2017'}")
    if not args.skip_pretrained:
        print(f"        pretrained:      {weights_dir / 'yolox_tiny.pth'}")


if __name__ == "__main__":
    main()
