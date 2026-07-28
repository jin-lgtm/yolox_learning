#!/usr/bin/env bash
set -euo pipefail

resolve_exp_name() {
  local variant="$1"
  case "${variant}" in
    nano_std_640)
      printf 'yolox_nano_custom17_%s\n' "${variant}"
      ;;
    *)
      printf 'yolox_nano_fusion_custom17_%s\n' "${variant}"
      ;;
  esac
}

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  custom17/scripts/run_ablation_nano.sh <variant> [train args...]

Variants:
  nano_std_640
  nano_fuse_p4p5_640
  nano_fuse_p5_640
  nano_fuse_p4ctxp5_on_640
  nano_fuse_p4ctxp5_off_640
  nano_fuse_p4p5_res_on_640
  nano_fuse_p4p5_res_off_640

Example:
  custom17/scripts/run_ablation_nano.sh nano_fuse_p4p5_640 \
    -d 1 -b 32 --fp16 -o -c pretrained_models/yolox_nano.pth
EOF
  exit 1
fi

variant="$1"
shift

export CUSTOM17_ABLATION_TAG="${variant}"
export YOLOX_MLFLOW_RUN_NAME="${YOLOX_MLFLOW_RUN_NAME:-${variant}}"

train_cmd=(
  uv run python custom17/scripts/train.py
)

case "${variant}" in
  nano_std_640)
    export CUSTOM17_INPUT_SIZE=640
    train_cmd+=(-f custom17/exp/yolox_nano_custom17.py)
    ;;

  nano_fuse_p4p5_640)
    export CUSTOM17_INPUT_SIZE=640
    export CUSTOM17_FUSION_PREDICTION_MODE=p4p5
    export CUSTOM17_FUSION_USE_P5=1
    export CUSTOM17_FUSION_P4_RESIDUAL=0
    export CUSTOM17_FUSION_P5_RESIDUAL=0
    train_cmd+=(-f custom17/exp/yolox_nano_fusion_custom17.py)
    ;;

  nano_fuse_p5_640)
    export CUSTOM17_INPUT_SIZE=640
    export CUSTOM17_FUSION_PREDICTION_MODE=p5
    export CUSTOM17_FUSION_USE_P5=1
    export CUSTOM17_FUSION_P4_RESIDUAL=0
    export CUSTOM17_FUSION_P5_RESIDUAL=0
    train_cmd+=(-f custom17/exp/yolox_nano_fusion_custom17.py)
    ;;

  nano_fuse_p4ctxp5_on_640)
    export CUSTOM17_INPUT_SIZE=640
    export CUSTOM17_FUSION_PREDICTION_MODE=p4
    export CUSTOM17_FUSION_USE_P5=1
    export CUSTOM17_FUSION_P4_RESIDUAL=0
    export CUSTOM17_FUSION_P5_RESIDUAL=0
    train_cmd+=(-f custom17/exp/yolox_nano_fusion_custom17.py)
    ;;

  nano_fuse_p4ctxp5_off_640)
    export CUSTOM17_INPUT_SIZE=640
    export CUSTOM17_FUSION_PREDICTION_MODE=p4
    export CUSTOM17_FUSION_USE_P5=0
    export CUSTOM17_FUSION_P4_RESIDUAL=0
    export CUSTOM17_FUSION_P5_RESIDUAL=0
    train_cmd+=(-f custom17/exp/yolox_nano_fusion_custom17.py)
    ;;

  nano_fuse_p4p5_res_on_640)
    export CUSTOM17_INPUT_SIZE=640
    export CUSTOM17_FUSION_PREDICTION_MODE=p4p5
    export CUSTOM17_FUSION_USE_P5=1
    export CUSTOM17_FUSION_P4_RESIDUAL=1
    export CUSTOM17_FUSION_P5_RESIDUAL=1
    train_cmd+=(-f custom17/exp/yolox_nano_fusion_custom17.py)
    ;;

  nano_fuse_p4p5_res_off_640)
    export CUSTOM17_INPUT_SIZE=640
    export CUSTOM17_FUSION_PREDICTION_MODE=p4p5
    export CUSTOM17_FUSION_USE_P5=1
    export CUSTOM17_FUSION_P4_RESIDUAL=0
    export CUSTOM17_FUSION_P5_RESIDUAL=0
    train_cmd+=(-f custom17/exp/yolox_nano_fusion_custom17.py)
    ;;

  *)
    echo "Unknown variant: ${variant}" >&2
    exit 1
    ;;
esac

exp_name="$(resolve_exp_name "${variant}")"
printf 'Running ablation variant: %s\n' "${variant}"
printf 'Experiment name: %s\n' "${exp_name}"
printf 'Output dir: YOLOX_outputs/%s\n' "${exp_name}"

"${train_cmd[@]}" "$@"
