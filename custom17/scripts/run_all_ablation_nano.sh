#!/usr/bin/env bash
set -euo pipefail

variants=(
  nano_std_640
  nano_fuse_p4p5_640
  nano_fuse_p5_640
  nano_fuse_p4ctxp5_on_640
  nano_fuse_p4ctxp5_off_640
  nano_fuse_p4p5_res_on_640
  nano_fuse_p4p5_res_off_640
)

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

resolve_exp_file() {
  local variant="$1"
  case "${variant}" in
    nano_std_640)
      printf 'custom17/exp/yolox_nano_custom17.py\n'
      ;;
    *)
      printf 'custom17/exp/yolox_nano_fusion_custom17.py\n'
      ;;
  esac
}

timestamp="$(date +%Y%m%d_%H%M%S)"
log_root="runs/ablation_nano_${timestamp}"
mkdir -p "${log_root}"

summary_file="${log_root}/summary.txt"
: > "${summary_file}"
summary_csv="${log_root}/summary.csv"
: > "${summary_file}"

printf 'Ablation log root: %s\n' "${log_root}"
printf 'Summary file: %s\n' "${summary_file}"
printf 'Summary csv: %s\n' "${summary_csv}"

fail_count=0

for variant in "${variants[@]}"; do
  log_file="${log_root}/${variant}.log"
  eval_log="${log_root}/${variant}_eval.log"
  bench_log="${log_root}/${variant}_bench.log"
  metrics_json="${log_root}/${variant}_eval_metrics.json"
  detection_json="${log_root}/${variant}_detections.json"
  bench_json="${log_root}/${variant}_benchmark.json"
  exp_name="$(resolve_exp_name "${variant}")"
  exp_file="$(resolve_exp_file "${variant}")"
  output_dir="YOLOX_outputs/${exp_name}"
  onnx_path="${output_dir}/best_ckpt.onnx"
  printf '\n[%s] start %s\n' "$(date '+%F %T')" "${variant}" | tee -a "${summary_file}"

  if custom17/scripts/run_ablation_nano.sh "${variant}" "$@" 2>&1 | tee "${log_file}"; then
    status="success"
    if [[ ! -f "${onnx_path}" ]]; then
      status="missing_onnx"
      fail_count=$((fail_count + 1))
      printf '[%s] missing onnx %s\n' "$(date '+%F %T')" "${onnx_path}" | tee -a "${summary_file}"
    else
      if uv run python custom17/scripts/eval_onnx.py \
        -m "${onnx_path}" \
        -f "${exp_file}" \
        --conf 0.001 \
        --provider cpu \
        --save-json "${detection_json}" \
        --save-metrics-json "${metrics_json}" 2>&1 | tee "${eval_log}"; then
        :
      else
        status="eval_failed"
        fail_count=$((fail_count + 1))
      fi

      if [[ "${status}" == "success" ]]; then
        if uv run python custom17/scripts/benchmark_onnx.py \
          -m "${onnx_path}" \
          --provider cpu \
          --score-thr 0.2 \
          --nms-thr 0.45 \
          --image-dir datasets/custom17/val2017 \
          --max-images 50 \
          --output-json "${bench_json}" 2>&1 | tee "${bench_log}"; then
          :
        else
          status="benchmark_failed"
          fail_count=$((fail_count + 1))
        fi
      fi
    fi

    uv run python custom17/scripts/append_ablation_summary.py \
      --csv-path "${summary_csv}" \
      --variant "${variant}" \
      --status "${status}" \
      --exp-name "${exp_name}" \
      --output-dir "${output_dir}" \
      --onnx-path "${onnx_path}" \
      --metrics-json "${metrics_json}" \
      --benchmark-json "${bench_json}"

    printf '[%s] %s %s\n' "$(date '+%F %T')" "${status}" "${variant}" | tee -a "${summary_file}"
  else
    status=$?
    fail_count=$((fail_count + 1))
    printf '[%s] failed %s (exit=%s)\n' "$(date '+%F %T')" "${variant}" "${status}" | tee -a "${summary_file}"
    uv run python custom17/scripts/append_ablation_summary.py \
      --csv-path "${summary_csv}" \
      --variant "${variant}" \
      --status "train_failed" \
      --exp-name "${exp_name}" \
      --output-dir "${output_dir}" \
      --onnx-path "${onnx_path}"
  fi
done

printf '\nCompleted at %s\n' "$(date '+%F %T')" | tee -a "${summary_file}"
printf 'Failed runs: %s\n' "${fail_count}" | tee -a "${summary_file}"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
