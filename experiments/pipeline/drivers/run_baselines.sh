#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/pipeline
source "${E}/config.sh"
for entry in ${CELLS}; do
  tag="${entry%%:*}"; student="${entry#*:}"
  gen_limits "${student}"; maxlen="${MAXLEN}"; newtok="${NEWTOK}"
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
echo "[$(date '+%F %T')] done"
