# YOLOX Custom17 Fine-tuning Guide

This repository now contains a YOLOX-compatible pipeline for fine-tuning `YOLOX-Tiny` on the following 17 classes, in this exact order:

1. `person`
2. `bottle`
3. `wine glass`
4. `cup`
5. `bowl`
6. `chair`
7. `couch`
8. `bed`
9. `tv`
10. `laptop`
11. `mouse`
12. `remote`
13. `keyboard`
14. `cell phone`
15. `book`
16. `clock`
17. `vase`

The implementation enforces the same class order in:

- `annotation categories`
- remapped `category_id` values `0..16`
- `YOLOX Exp.num_classes`
- `YOLOX Exp.class_names`

If these diverge, mAP drops sharply and the first thing to verify is class remapping, bbox format, and eval threshold.

## Added files

- `custom17/scripts/download_custom17_assets.py`
- `custom17/scripts/filter_annotations.py`
- `custom17/scripts/validate_annotations.py`
- `custom17/scripts/visualize_annotations.py`
- `custom17/common.py`
- `custom17/exp/yolox_tiny_custom17.py`

## Recommended directory layout

```text
datasets/custom17/
  annotations/
    train.json
    val.json
  raw_annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
  val2017/
```

## 1. Install dependencies

```bash
uv venv
source .venv/bin/activate
```

Install PyTorch first.

CUDA 11.8 example:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
```

CPU-only example:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Then install YOLOX and the remaining dependencies:

```bash
uv pip install --no-build-isolation -e ./upstream_yolox
uv pip install pycocotools opencv-python tabulate tensorboard
```

`torch` must be installed before `upstream_yolox`, otherwise editable install fails while building YOLOX ops.

## 2. Download COCO images, COCO annotations, and YOLOX-Tiny pretrained weights

```bash
uv run python custom17/scripts/download_custom17_assets.py
```

This downloads:

- `datasets/custom17/train2017`
- `datasets/custom17/val2017`
- `datasets/custom17/raw_annotations/instances_train2017.json`
- `datasets/custom17/raw_annotations/instances_val2017.json`
- `pretrained_models/yolox_tiny.pth`

Optional flags:

```bash
uv run python custom17/scripts/download_custom17_assets.py --skip-pretrained
uv run python custom17/scripts/download_custom17_assets.py --skip-images
uv run python custom17/scripts/download_custom17_assets.py --force
```

## 3. Filter COCO annotations to the target 17 classes and remap ids to 0..16

```bash
uv run python custom17/scripts/filter_annotations.py
```

This generates:

- `datasets/custom17/annotations/train.json`
- `datasets/custom17/annotations/val.json`

Important behavior:

- only the 17 target classes are kept
- `category_id` is always remapped to `0..16`
- category order is always the class order listed above
- bbox format stays COCO `[x, y, w, h]`
- image entries are kept even when a selected target class is absent, so negatives remain available

If you want to drop empty images:

```bash
uv run python custom17/scripts/filter_annotations.py --drop-empty-images
```

## 4. Validate class mapping and bbox integrity

Run structural validation:

```bash
uv run python custom17/scripts/validate_annotations.py --annotation datasets/custom17/annotations/train.json
uv run python custom17/scripts/validate_annotations.py --annotation datasets/custom17/annotations/val.json
```

This checks:

- categories are exactly `0..16`
- category names match the required class order
- every annotation points to an existing image
- bbox format is valid
- bbox boundary warnings are reported
- per-class instance counts are printed

## 5. Visualize remapped bounding boxes

Train split visualization:

```bash
uv run python custom17/scripts/visualize_annotations.py \
  --annotation datasets/custom17/annotations/train.json \
  --image-dir datasets/custom17/train2017 \
  --output-dir runs/custom17_vis/train \
  --num-images 30
```

Val split visualization:

```bash
uv run python custom17/scripts/visualize_annotations.py \
  --annotation datasets/custom17/annotations/val.json \
  --image-dir datasets/custom17/val2017 \
  --output-dir runs/custom17_vis/val \
  --num-images 30
```

Before training, inspect a few rendered samples and verify:

- each class label text matches the drawn object
- `category_id` and class name are aligned
- bbox corners are reasonable
- no obvious class swaps exist, especially `couch`, `tv`, and `cell phone`

## 6. Custom YOLOX-Tiny experiment

`custom17/exp/yolox_tiny_custom17.py` is configured as follows:

- `num_classes = 17`
- `depth = 0.33`
- `width = 0.375`
- `input_size = (640, 640)`
- `test_size = (640, 640)`
- `max_epoch = 50`
- `no_aug_epochs = 10`
- `warmup_epochs = 3`
- `mixup_prob = 0.0`
- `enable_mixup = False`
- `mosaic_prob = 0.5`
- `mosaic_scale = (0.8, 1.2)` for weaker mosaic
- `test_conf = 0.001` for mAP evaluation
- `eval_interval = 1`

`no_aug_epochs` disables mosaic in the final stage through the normal YOLOX training flow.

## 7. Fine-tune from YOLOX-Tiny COCO pretrained weights

Single-node multi-GPU example:

```bash
uv run python custom17/scripts/train.py \
  -f custom17/exp/yolox_tiny_custom17.py \
  -d 1 \
  -b 32 \
  --fp16 \
  -o \
  -c pretrained_models/yolox_tiny.pth
```

CPU-visible GPU count can be adjusted with `-d`.

Notes:

- `--fp16` is supported
- using a COCO pretrained checkpoint is recommended for this 17-class setup
- classification head shape mismatch during checkpoint load is expected and normal
- YOLOX handles partial checkpoint loading for fine-tuning with `strict=False`
- the new 17-class detection head is learned during training
- `custom17/scripts/train.py` injects `upstream_yolox` into `sys.path`, so you do not need to set `PYTHONPATH` manually

## 8. Evaluate with low confidence threshold for mAP

Run evaluation with the same experiment file and a low eval confidence threshold:

```bash
uv run python custom17/scripts/eval.py \
  -f custom17/exp/yolox_tiny_custom17.py \
  -d 1 \
  -b 32 \
  --fp16 \
  -c YOLOX_outputs/yolox_tiny_custom17/best_ckpt.pth \
  --conf 0.001
```

YOLOX COCO evaluation summary prints both:

- `AP@[IoU=0.50:0.95]`
- `AP@[IoU=0.50]`

If `mAP@0.5` is unexpectedly low, for example around `43`, verify in this order:

1. `category_id` remapping is exactly `0..16`
2. annotation `categories` order matches the class order
3. YOLOX `num_classes` and `class_names` match the same order
4. bbox format is still COCO `[x, y, w, h]`
5. eval is run with low `conf`, for example `0.001`

## 9. Objects365 usage

The filtering code works on generic COCO-style JSON as long as:

- `images`, `annotations`, and `categories` fields exist
- target class names match the expected names or aliases in `custom17/common.py`

For Objects365-based data preparation:

1. place the raw COCO-style annotation JSON under a local path
2. update `--train-input` and `--val-input` to those files
3. run the same filtering script
4. validate and visualize before training

Example:

```bash
uv run python custom17/scripts/filter_annotations.py \
  --train-input /path/to/objects365_train.json \
  --val-input /path/to/objects365_val.json
```

If your Objects365 export uses slightly different class names, extend the aliases in `custom17/common.py` before filtering.

## 10. Optional teacher-student distillation structure

Final deployment model should remain `YOLOX-Tiny`. A practical path is:

1. train a `YOLOX-S` teacher on the same `custom17` dataset
2. freeze or partially freeze the teacher during student training
3. train `YOLOX-Tiny` as student with:
   - standard detection loss on ground-truth labels
   - feature-level distillation on FPN outputs
   - logit distillation on classification/objectness heads
4. evaluate and deploy only the student checkpoint

Suggested file layout if you add this later:

```text
custom17/exp/yolox_s_custom17_teacher.py
custom17/scripts/train_distill.py
upstream_yolox/yolox/models/distill_losses.py
```

Keep teacher and student annotation class order identical. Distillation does not fix broken class remapping.
