#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/pipeline
source "${E}/config.sh"
for student in Qwen2.5-0.5B-Instruct Llama-3.2-1B-Instruct OLMo-2-1B-Instruct; do
  gen_limits "${student}"; maxlen="${MAXLEN}"; newtok="${NEWTOK}"
  mkdir -p "${E}/base_eval/${student}"
  for seed in 0 1 2; do
    echo "[$(date '+%F %T')] ${student} s${seed}"
    python3 "${E}/eval_ablation.py" --base "models/${student}" --adapters "base=" \
      --data "${E}/data/math500.parquet" --out_dir "${E}/base_eval/${student}/eval_main_s${seed}" \
      --max_model_len "${maxlen}" --max_new_tokens "${newtok}" --seed "${seed}" \
      >> "${E}/base_eval/${student}/eval.log" 2>&1 || exit 1
  done
done
chown -R "$(stat -c '%u:%g' .)" "${E}/base_eval" 2>/dev/null || true
echo "[$(date '+%F %T')] done"
