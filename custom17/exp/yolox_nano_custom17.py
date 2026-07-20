#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
from pathlib import Path

import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = REPO_ROOT / "upstream_yolox"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from yolox.exp import Exp as MyExp

from custom17.common import (
    CUSTOM17_CLASSES,
    resolve_bool_env,
    resolve_int_env,
    resolve_size_override,
)
from custom17.train_loader import build_custom17_train_loader


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.num_classes = 17
        self.class_names = tuple(CUSTOM17_CLASSES)

        self.data_dir = "datasets/custom17"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        resolved_size = resolve_size_override((640, 640))
        self.input_size = resolved_size
        self.test_size = resolved_size
        self.multiscale_range = 0
        self.random_size = (20, 20)

        self.max_epoch = 50
        self.no_aug_epochs = 10
        self.warmup_epochs = 3
        self.data_num_workers = 4
        self.eval_interval = 1

        self.mosaic_prob = 0.5
        self.mixup_prob = 0.0
        self.enable_mixup = False
        self.mosaic_scale = (0.8, 1.2)
        self.degrees = 5.0
        self.translate = 0.05
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.basic_lr_per_img = 0.01 / 64.0
        self.test_conf = 0.001
        self.nmsthre = 0.65
        self.balanced_resample = resolve_bool_env("CUSTOM17_BALANCED_RESAMPLE", False)
        self.balanced_resample_seed = resolve_int_env("CUSTOM17_BALANCED_RESAMPLE_SEED", 42)

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        return build_custom17_train_loader(
            self,
            batch_size=batch_size,
            is_distributed=is_distributed,
            no_aug=no_aug,
            cache_img=cache_img,
        )

    def get_model(self, sublinear=False):
        def init_yolo(module):
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
