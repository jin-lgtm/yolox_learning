#!/usr/bin/env python3
"""Custom train loader helpers for custom17 experiments."""

from __future__ import annotations

from collections import Counter
import os
import random

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data.sampler import Sampler


class BalancedInfiniteSampler(Sampler):
    """Build a smaller balanced subset each epoch and repeat indefinitely."""

    def __init__(
        self,
        image_classes: list[set[int]],
        target_count: int | None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self._image_classes = image_classes
        self._size = len(image_classes)
        assert self._size > 0
        self._target_count = target_count
        self._seed = int(seed)
        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size
        self._epoch_indices = self._build_epoch_indices(self._seed)
        self._epoch_size = len(self._epoch_indices)

    def _build_epoch_indices(self, seed: int) -> list[int]:
        rng = random.Random(seed)
        class_image_counts: Counter[int] = Counter()
        positive_classes: set[int] = set()
        for classes in self._image_classes:
            for class_id in classes:
                class_image_counts[class_id] += 1
                positive_classes.add(class_id)

        if not positive_classes:
            return list(range(self._size))

        resolved_target = self._target_count or min(class_image_counts[class_id] for class_id in positive_classes)
        selected_counts: Counter[int] = Counter()
        candidate_indices = list(range(self._size))
        rng.shuffle(candidate_indices)
        candidate_indices.sort(
            key=lambda image_idx: (
                min((class_image_counts[class_id] for class_id in self._image_classes[image_idx]), default=10**9),
                -len(self._image_classes[image_idx]),
            )
        )

        selected_indices: list[int] = []
        for image_idx in candidate_indices:
            classes = self._image_classes[image_idx]
            if not classes:
                continue
            if not any(selected_counts[class_id] < resolved_target for class_id in classes):
                continue
            selected_indices.append(image_idx)
            for class_id in classes:
                if selected_counts[class_id] < resolved_target:
                    selected_counts[class_id] += 1
            if all(selected_counts[class_id] >= resolved_target for class_id in positive_classes):
                break

        if not selected_indices:
            return candidate_indices
        return selected_indices

    def __iter__(self):
        epoch = 0
        while True:
            epoch_indices = self._epoch_indices if epoch == 0 else self._build_epoch_indices(self._seed + epoch)
            yield from epoch_indices[self._rank :: self._world_size]
            epoch += 1

    def __len__(self):
        return self._epoch_size // self._world_size


def _collect_image_classes(base_dataset) -> list[set[int]]:
    image_classes: list[set[int]] = []

    for labels, *_ in base_dataset.annotations:
        if getattr(labels, "size", 0) == 0:
            classes = set()
        else:
            classes = {int(class_id) for class_id in labels[:, 4].tolist()}
        image_classes.append(classes)
    return image_classes


def _env_enabled(env_key: str) -> bool:
    return os.getenv(env_key, "").strip().lower() in {"1", "true", "yes", "on"}


def build_custom17_train_loader(exp, batch_size, is_distributed, no_aug=False, cache_img: str = None):
    from yolox.data import (
        TrainTransform,
        YoloBatchSampler,
        DataLoader,
        InfiniteSampler,
        MosaicDetection,
        worker_init_reset_seed,
    )
    from yolox.utils import wait_for_the_master

    if exp.dataset is None:
        with wait_for_the_master():
            assert cache_img is None, "cache_img must be None if you didn't create self.dataset before launch"
            exp.dataset = exp.get_dataset(cache=False, cache_type=cache_img)

    base_dataset = exp.dataset
    dataset = MosaicDetection(
        dataset=base_dataset,
        mosaic=not no_aug,
        img_size=exp.input_size,
        preproc=TrainTransform(
            max_labels=120,
            flip_prob=exp.flip_prob,
            hsv_prob=exp.hsv_prob,
        ),
        degrees=exp.degrees,
        translate=exp.translate,
        mosaic_scale=exp.mosaic_scale,
        mixup_scale=exp.mixup_scale,
        shear=exp.shear,
        enable_mixup=exp.enable_mixup,
        mosaic_prob=exp.mosaic_prob,
        mixup_prob=exp.mixup_prob,
    )
    exp.dataset = dataset

    if is_distributed:
        batch_size = batch_size // dist.get_world_size()

    balanced_resample = getattr(exp, "balanced_resample", False) or _env_enabled("CUSTOM17_BALANCED_RESAMPLE")
    balanced_resample_seed = getattr(exp, "balanced_resample_seed", 42)
    balanced_target_count = getattr(exp, "balanced_target_count", None)
    if os.getenv("CUSTOM17_BALANCED_RESAMPLE_SEED", "").strip():
        balanced_resample_seed = int(os.getenv("CUSTOM17_BALANCED_RESAMPLE_SEED", "42"))
    if os.getenv("CUSTOM17_BALANCED_TARGET_COUNT", "").strip():
        balanced_target_count = int(os.getenv("CUSTOM17_BALANCED_TARGET_COUNT", "0")) or None

    if balanced_resample:
        image_classes = _collect_image_classes(base_dataset)
        class_image_counts: Counter[int] = Counter()
        for classes in image_classes:
            for class_id in classes:
                class_image_counts[class_id] += 1
        positive_counts = [count for count in class_image_counts.values() if count > 0]
        resolved_target = balanced_target_count or (min(positive_counts) if positive_counts else len(base_dataset))
        sampler = BalancedInfiniteSampler(
            image_classes=image_classes,
            target_count=resolved_target,
            seed=balanced_resample_seed,
        )
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        logger.info(
            "Using balanced subset resampling for training: dataset_size={}, target_count={}, epoch_size_per_rank={}, batches_per_epoch={}, seed={}, class_image_counts={}",
            len(base_dataset),
            resolved_target,
            len(sampler),
            len(batch_sampler),
            balanced_resample_seed,
            dict(sorted(class_image_counts.items())),
        )
    else:
        sampler = InfiniteSampler(len(dataset), seed=exp.seed if exp.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

    dataloader_kwargs = {"num_workers": exp.data_num_workers, "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
    return DataLoader(dataset, **dataloader_kwargs)
