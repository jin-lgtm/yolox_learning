#!/usr/bin/env python3
"""Run YOLOX evaluation with the upstream submodule on sys.path."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from custom17.runtime_patches import patch_coco_evaluator_output


REPO_ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = REPO_ROOT / "upstream_yolox"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

patch_coco_evaluator_output()
runpy.run_path(str(YOLOX_ROOT / "tools" / "eval.py"), run_name="__main__")
