#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = REPO_ROOT / "upstream_yolox"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from yolox.exp import Exp as MyExp

from custom17.common import CUSTOM17_CLASSES, resolve_size_override


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
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

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
