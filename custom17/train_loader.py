#!/usr/bin/env python3
"""Custom train loader helpers for custom17 experiments."""

from __future__ import annotations

from collections import Counter
import math
import os

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data.sampler import BatchSampler, Sampler


class BalancedYoloBatchSampler(BatchSampler):
    """Yield balanced mini-batches with replacement using image-level rarity weights."""

    def __init__(
        self,
        weights: torch.Tensor,
        batch_size: int,
        epoch_size: int,
        seed: int = 0,
        mosaic: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self._weights = weights.to(dtype=torch.double, device="cpu")
        self.batch_size = int(batch_size)
        self._epoch_size = int(epoch_size)
        self._num_batches = math.ceil(self._epoch_size / self.batch_size)
        self._seed = int(seed)
        self.mosaic = mosaic
        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self._seed + self._rank)
        while True:
            for _ in range(self._num_batches):
                sampled = torch.multinomial(
                    self._weights,
                    self.batch_size,
                    replacement=True,
                    generator=generator,
                )
                yield [(self.mosaic, idx) for idx in sampled.tolist()]

    def __len__(self):
        return self._num_batches


class BalancedInfiniteSampler(Sampler):
    """Backward-compatible balanced sampler that yields single indices."""

    def __init__(
        self,
        weights: torch.Tensor,
        epoch_size: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self._weights = weights.to(dtype=torch.double, device="cpu")
        self._epoch_size = int(epoch_size)
        self._seed = int(seed)
        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self._seed + self._rank)
        while True:
            sampled = torch.multinomial(
                self._weights,
                self._epoch_size,
                replacement=True,
                generator=generator,
            )
            yield from sampled.tolist()

    def __len__(self):
        return self._epoch_size


def _compute_balanced_weights(base_dataset) -> torch.Tensor:
    class_counts: Counter[int] = Counter()
    image_classes: list[set[int]] = []

    for labels, *_ in base_dataset.annotations:
        if getattr(labels, "size", 0) == 0:
            classes = set()
        else:
            classes = {int(class_id) for class_id in labels[:, 4].tolist()}
        image_classes.append(classes)
        for class_id in classes:
            class_counts[class_id] += 1

    if not class_counts:
        return torch.ones(len(image_classes), dtype=torch.double)

    min_positive_weight = min(1.0 / count for count in class_counts.values() if count > 0)
    negative_weight = min_positive_weight * 0.5
    weights = []
    for classes in image_classes:
        if classes:
            weight = max(1.0 / class_counts[class_id] for class_id in classes)
        else:
            weight = negative_weight
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.double)


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
    if os.getenv("CUSTOM17_BALANCED_RESAMPLE_SEED", "").strip():
        balanced_resample_seed = int(os.getenv("CUSTOM17_BALANCED_RESAMPLE_SEED", "42"))

    if balanced_resample:
        weights = _compute_balanced_weights(base_dataset)
        batch_sampler = BalancedYoloBatchSampler(
            weights=weights,
            batch_size=batch_size,
            epoch_size=len(dataset),
            seed=balanced_resample_seed,
            mosaic=not no_aug,
        )
        logger.info(
            "Using balanced online resampling for training: epoch_size={}, batches_per_epoch={}, seed={}",
            len(dataset),
            len(batch_sampler),
            balanced_resample_seed,
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
