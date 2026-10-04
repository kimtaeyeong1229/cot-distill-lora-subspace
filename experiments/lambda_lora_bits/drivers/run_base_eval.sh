#!/usr/bin/env bash
# The untrained students under the paper's evaluation (first 200 MATH-500 problems x 2 samples,
# 7,680 new tokens, OLMo 3,840), so the think-off edits of section 5 can be read against the model
# before any distillation. With no adapter there is no training seed; the three replicates vary the
# sampling seed instead. Results: base_eval/<student>/eval_main_s<seed>/base.json
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
source "${E}/config.sh"   # gen_limits, PAPER_SWITCH, PAPER_SEEDS
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
echo "[$(date '+%F %T')] 완료"
