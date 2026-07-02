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
- `custom17/scripts/webcam_demo.py`
- `custom17/scripts/onnx_infer.py`
- `custom17/common.py`
- `custom17/exp/yolox_nano_custom17.py`
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
uv pip install onnx onnxruntime
```

`torch` must be installed before `upstream_yolox`, otherwise editable install fails while building YOLOX ops.

## 2. Choose a dataset source

The pipeline supports two raw-data sources:

- `coco`
- `objects365`

Both are normalized into the same final layout:

```text
datasets/custom17/
  annotations/
    train.json
    val.json
  train2017/
  val2017/
```

The training Exp only sees the normalized output above. The difference is in how `raw_annotations/` and source images are prepared.

## 3. Download COCO images, COCO annotations, and pretrained weights

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

To download the Nano checkpoint instead:

```bash
uv run python custom17/scripts/download_custom17_assets.py --model nano
```

To download both Tiny and Nano checkpoints:

```bash
uv run python custom17/scripts/download_custom17_assets.py --model both
```

## 4. Download or stage Objects365 assets

For `Objects365`, use `--source objects365` and pass either URLs or local archive/json paths.

Expected inputs:

- `--train-image-src`
- `--val-image-src`
- `--train-ann-src`
- `--val-ann-src`

These can be:

- direct `http(s)` URLs
- local archive paths
- local `.json` annotation files

Example:

```bash
uv run python custom17/scripts/download_custom17_assets.py \
  --source objects365 \
  --train-image-src /data/objects365/train_images.zip \
  --val-image-src /data/objects365/val_images.zip \
  --train-ann-src /data/objects365/zhiyuan_objv2_train.json \
  --val-ann-src /data/objects365/zhiyuan_objv2_val.json \
  --train-image-strip-prefix train \
  --val-image-strip-prefix val
```

If your image archive contains an extra top-level directory, use:

- `--train-image-strip-prefix`
- `--val-image-strip-prefix`

If your annotation json lives inside a zip or tar, use:

- `--train-ann-member`
- `--val-ann-member`

You can also provide the Objects365 source paths via environment variables:

```bash
export OBJECTS365_TRAIN_IMAGE_SRC=/data/objects365/train_images.zip
export OBJECTS365_VAL_IMAGE_SRC=/data/objects365/val_images.zip
export OBJECTS365_TRAIN_ANN_SRC=/data/objects365/zhiyuan_objv2_train.json
export OBJECTS365_VAL_ANN_SRC=/data/objects365/zhiyuan_objv2_val.json
uv run python custom17/scripts/download_custom17_assets.py --source objects365
```

The script writes normalized files under:

- `datasets/custom17/raw_annotations/objects365_train.json`
- `datasets/custom17/raw_annotations/objects365_val.json`
- `datasets/custom17/train2017/`
- `datasets/custom17/val2017/`

## 5. Filter source annotations to the target 17 classes and remap ids to 0..16

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

Objects365 example:

```bash
uv run python custom17/scripts/filter_annotations.py --source objects365
```

## 6. Validate class mapping and bbox integrity

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

## 7. Visualize remapped bounding boxes

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

## 8. Custom YOLOX experiments

Tiny deploy model:

- exp file: `custom17/exp/yolox_tiny_custom17.py`
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

Nano option:

- exp file: `custom17/exp/yolox_nano_custom17.py`
- `num_classes = 17`
- `depth = 0.33`
- `width = 0.25`
- uses `depthwise=True` backbone/head like standard YOLOX-Nano
- `input_size = (640, 640)`
- `test_size = (640, 640)`
- same dataset, augmentation, and evaluation defaults as the Tiny setup

Tiny settings summary:

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

## 9. Fine-tune from YOLOX-Tiny COCO pretrained weights

Tiny single-node example:

```bash
uv run python custom17/scripts/train.py \
  -f custom17/exp/yolox_tiny_custom17.py \
  -d 1 \
  -b 32 \
  --fp16 \
  -o \
  -c pretrained_models/yolox_tiny.pth
```

Nano single-node example:

```bash
uv run python custom17/scripts/train.py \
  -f custom17/exp/yolox_nano_custom17.py \
  -d 1 \
  -b 32 \
  --fp16 \
  -o \
  -c /path/to/yolox_nano.pth
```

CPU-visible GPU count can be adjusted with `-d`.

Notes:

- `--fp16` is supported
- using a COCO pretrained checkpoint is recommended for this 17-class setup
- Tiny uses the YOLOX-Tiny checkpoint, Nano should use the YOLOX-Nano checkpoint
- classification head shape mismatch during checkpoint load is expected and normal
- YOLOX handles partial checkpoint loading for fine-tuning with `strict=False`
- the new 17-class detection head is learned during training
- `custom17/scripts/train.py` injects `upstream_yolox` into `sys.path`, so you do not need to set `PYTHONPATH` manually
- evaluation output now includes both `per class AP` (`AP50:95`) and `per class AP50`
- after training ends, `best_ckpt.pth` is automatically exported to `best_ckpt_fp16.onnx`

## 10. Evaluate with low confidence threshold for mAP

Run evaluation with the same experiment file and a low eval confidence threshold:

Tiny:

```bash
uv run python custom17/scripts/eval.py \
  -f custom17/exp/yolox_tiny_custom17.py \
  -d 1 \
  -b 32 \
  --fp16 \
  -c YOLOX_outputs/yolox_tiny_custom17/best_ckpt.pth \
  --conf 0.001
```

Nano:

```bash
uv run python custom17/scripts/eval.py \
  -f custom17/exp/yolox_nano_custom17.py \
  -d 1 \
  -b 32 \
  --fp16 \
  -c YOLOX_outputs/yolox_nano_custom17/best_ckpt.pth \
  --conf 0.001
```

YOLOX COCO evaluation summary prints both:

- `AP@[IoU=0.50:0.95]`
- `AP@[IoU=0.50]`
- `per class AP` as class-wise `AP50:95`
- `per class AP50` as class-wise `AP@0.5`

If `mAP@0.5` is unexpectedly low, for example around `43`, verify in this order:

1. `category_id` remapping is exactly `0..16`
2. annotation `categories` order matches the class order
3. YOLOX `num_classes` and `class_names` match the same order
4. bbox format is still COCO `[x, y, w, h]`
5. eval is run with low `conf`, for example `0.001`

## 11. Objects365 notes

The filtering code works on generic COCO-style JSON as long as:

- `images`, `annotations`, and `categories` fields exist
- target class names match the expected names or aliases in `custom17/common.py`

For Objects365-based training, make sure:

1. the image files extracted under `datasets/custom17/train2017` and `val2017` match the `file_name` fields in the Objects365 json
2. you run `download_custom17_assets.py --source objects365`
3. you run `filter_annotations.py --source objects365`
4. you validate and visualize before training

If your Objects365 export uses slightly different class names, extend the aliases in `custom17/common.py` before filtering.

## 12. Webcam demo

Run a real-time webcam demo with the trained checkpoint:

```bash
uv run python custom17/scripts/webcam_demo.py \
  -f custom17/exp/yolox_tiny_custom17.py \
  -c YOLOX_outputs/yolox_tiny_custom17/best_ckpt.pth \
  --device gpu \
  --fp16 \
  --conf 0.3 \
  --nms 0.45
```

Useful options:

- `--camid 0` to choose the webcam device
- `--device cpu` if GPU is not available
- `--save_result` to write an mp4 under `runs/webcam_demo/`
- press `q` or `Esc` to exit

## 13. ONNX inference

After training finishes, the best checkpoint is automatically exported to:

```text
YOLOX_outputs/<exp_name>/best_ckpt_fp16.onnx
```

Image inference with the exported ONNX:

```bash
uv run python custom17/scripts/onnx_infer.py image \
  -m YOLOX_outputs/yolox_tiny_custom17/best_ckpt_fp16.onnx \
  -f custom17/exp/yolox_tiny_custom17.py \
  --path /path/to/image_or_dir \
  --provider cpu \
  --save-result
```

Webcam inference with the exported ONNX:

```bash
uv run python custom17/scripts/onnx_infer.py webcam \
  -m YOLOX_outputs/yolox_tiny_custom17/best_ckpt_fp16.onnx \
  -f custom17/exp/yolox_tiny_custom17.py \
  --provider cpu
```

Video inference with the exported ONNX:

```bash
uv run python custom17/scripts/onnx_infer.py video \
  -m YOLOX_outputs/yolox_tiny_custom17/best_ckpt_fp16.onnx \
  -f custom17/exp/yolox_tiny_custom17.py \
  --path /path/to/video.mp4 \
  --save-result
```

## 14. Optional teacher-student distillation structure

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
