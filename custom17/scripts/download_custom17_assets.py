#!/usr/bin/env python3
"""Download COCO assets and YOLOX-Tiny pretrained weights for custom17 training."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


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
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/custom17"))
    parser.add_argument("--downloads-dir", type=Path, default=None)
    parser.add_argument("--weights-dir", type=Path, default=Path("pretrained_models"))
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-pretrained", action="store_true")
    parser.add_argument("--force", action="store_true")
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


def extract_zip_member(zip_path: Path, member_prefix: str, output_dir: Path, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.startswith(member_prefix)]
        for member in members:
            target_path = output_dir / Path(member).name
            if target_path.exists() and not force:
                print(f"[skip] {target_path} already exists")
                continue
            with zf.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"[extract] {target_path}")


def extract_zip_tree(zip_path: Path, member_prefix: str, output_dir: Path, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.startswith(member_prefix)]
        for member in members:
            target_path = output_dir / Path(member)
            if member.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() and not force:
                continue
            with zf.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        print(f"[extract] {zip_path.name} -> {output_dir}")


def maybe_extract_archive(archive_path: Path, output_dir: Path, force: bool = False) -> None:
    if archive_path.suffix == ".zip":
        extract_zip_tree(archive_path, archive_path.stem + "/", output_dir, force=force)
        return
    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        output_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path) as tf:
            tf.extractall(path=output_dir)
        print(f"[extract] {archive_path.name} -> {output_dir}")
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    downloads_dir = (args.downloads_dir or dataset_root / "_downloads").resolve()
    raw_annotations_dir = dataset_root / "raw_annotations"
    weights_dir = args.weights_dir.resolve()

    dataset_root.mkdir(parents=True, exist_ok=True)
    raw_annotations_dir.mkdir(parents=True, exist_ok=True)

    annotations_zip = downloads_dir / "annotations_trainval2017.zip"
    download_file(COCO_URLS["annotations"], annotations_zip, force=args.force)
    extract_zip_member(
        annotations_zip,
        "annotations/instances_",
        raw_annotations_dir,
        force=args.force,
    )

    if not args.skip_images:
        train_zip = downloads_dir / "train2017.zip"
        val_zip = downloads_dir / "val2017.zip"
        download_file(COCO_URLS["train_images"], train_zip, force=args.force)
        download_file(COCO_URLS["val_images"], val_zip, force=args.force)
        maybe_extract_archive(train_zip, dataset_root, force=args.force)
        maybe_extract_archive(val_zip, dataset_root, force=args.force)

    if not args.skip_pretrained:
        ckpt_path = weights_dir / "yolox_tiny.pth"
        download_file(YOLOX_TINY_COCO_CKPT_URL, ckpt_path, force=args.force)

    print("[done] Dataset assets are ready.")
    print(f"        raw annotations: {raw_annotations_dir}")
    print(f"        train images:    {dataset_root / 'train2017'}")
    print(f"        val images:      {dataset_root / 'val2017'}")
    if not args.skip_pretrained:
        print(f"        pretrained:      {weights_dir / 'yolox_tiny.pth'}")


if __name__ == "__main__":
    main()
