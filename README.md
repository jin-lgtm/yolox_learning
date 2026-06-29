# yolox_learning

This repository uses the official YOLOX source as a git submodule at `upstream_yolox/`.

Custom fine-tuning artifacts for the 17-class deployment target live in:

- `custom17/`
- `README_custom17.md`

## Layout

```text
.
├── custom17/
├── datasets/
├── upstream_yolox/
├── README.md
└── README_custom17.md
```

## Setup

Initialize the submodule:

```bash
git submodule update --init --recursive
```

Install YOLOX dependencies from the submodule:

```bash
pip install -U pip
pip install -v -e ./upstream_yolox
pip install pycocotools opencv-python
```

## Custom17 workflow

See `README_custom17.md` for:

- dataset download
- COCO/Objects365 annotation filtering
- class id remapping
- bbox visualization and validation
- YOLOX-Tiny fine-tuning
- evaluation with low confidence threshold
