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

Create a virtual environment with `uv`:

```bash
uv venv
source .venv/bin/activate
```

Install PyTorch first, then install YOLOX and the remaining dependencies.

CUDA 11.8 example:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
```

CPU-only example:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Then install YOLOX from the submodule:

```bash
uv pip install --no-build-isolation -e ./upstream_yolox
uv pip install pycocotools opencv-python tabulate tensorboard
```

Use `uv run` for scripts in this repository.

## Custom17 workflow

See `README_custom17.md` for:

- dataset download
- COCO/Objects365 annotation filtering
- class id remapping
- bbox visualization and validation
- YOLOX-Tiny fine-tuning
- evaluation with low confidence threshold
