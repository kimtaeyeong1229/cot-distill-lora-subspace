#!/usr/bin/env bash
# One machine's share of the inference-time baselines for the think-off edit (eval_baselines.py): the
# unedited lambda0 adapter with a NoThinking prefill, and cut at the region removal's mean length.
# Results go next to the section-5 edits, <cell>/eval_main_s<seed>/lambda0.0_{nothink,budget}.json.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
for entry in ${CELLS}; do
  tag="${entry%%:*}"; student="${entry#*:}"
  maxlen=8192; newtok=7680
  [[ "${student}" == *OLMo* ]] && maxlen=4096 && newtok=3840
  mkdir -p "${E}/${tag}/logs"
  for seed in ${SEEDS:-0 1 2}; do
    for mode in ${MODES:-nothink budget}; do
      extra=()
      if [[ "${mode}" == budget ]]; then
        b=$(python3 -c "import json; print(round(json.load(open('${E}/${tag}/eval_main_s${seed}/lambda0.0_rm_pre.json'))['avg_tokens']))") || exit 1
        extra=(--budget "${b}")
      fi
      echo "[$(date '+%F %T')] ${tag} s${seed} ${mode} ${extra[*]}"
      python3 "${E}/eval_baselines.py" --base "models/${student}" \
        --adapter "${E}/${tag}/runs/lambda0.0_seed${seed}/adapter" --mode "${mode}" "${extra[@]}" \
        --name "lambda0.0_${mode}" --data "${E}/data/math500.parquet" \
        --out_dir "${E}/${tag}/eval_main_s${seed}" \
        --max_model_len "${maxlen}" --max_new_tokens "${newtok}" \
        >> "${E}/${tag}/logs/eval_baselines.log" 2>&1 || exit 1
    done
  done
  chown -R "$(stat -c '%u:%g' .)" "${E}/${tag}" 2>/dev/null || true
done
echo "[$(date '+%F %T')] 완료"
