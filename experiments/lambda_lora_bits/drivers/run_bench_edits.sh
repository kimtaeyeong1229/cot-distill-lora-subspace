#!/usr/bin/env bash
# One machine's share of the out-of-domain check (eval_bench_edits.py): for each cell, seed and benchmark,
# the lambda0 adapter, the shared region removed, the same energy removed elsewhere, and the lambda1 adapter.
# Results: <cell>/eval_<bench>_s<seed>/{lambda0.0,lambda0.0_rm_pre,lambda0.0_rm_decisive,lambda1.0}.json
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
source "${E}/config.sh"   # gen_limits, PAPER_SWITCH, PAPER_SEEDS
for entry in ${CELLS}; do
  tag="${entry%%:*}"; student="${entry#*:}"
  gen_limits "${student}"; maxlen="${MAXLEN}"; newtok="${NEWTOK}"
  mkdir -p "${E}/${tag}/logs"
  for bench in ${BENCHES:-gpqa_diamond gsm8k}; do
    for seed in 0 1 2; do
      echo "[$(date '+%F %T')] ${tag} ${bench} s${seed}"
      python3 "${E}/eval_bench_edits.py" --base "models/${student}" --bench "${bench}" \
        --adapters "lambda0.0=${E}/${tag}/runs/lambda0.0_seed${seed}/adapter" \
                   "lambda0.0_rm_pre=${E}/${tag}/runs_edit_s${seed}/lambda0.0_rm_pre/adapter" \
                   "lambda0.0_rm_decisive=${E}/${tag}/runs_edit_s${seed}/lambda0.0_rm_decisive/adapter" \
                   "lambda1.0=${E}/${tag}/runs/lambda1.0_seed${seed}/adapter" \
        --out_dir "${E}/${tag}/eval_${bench}_s${seed}" \
        --max_model_len "${maxlen}" --max_new_tokens "${newtok}" \
        >> "${E}/${tag}/logs/eval_bench_edits.log" 2>&1 || exit 1
    done
  done
  chown -R "$(stat -c '%u:%g' .)" "${E}/${tag}" 2>/dev/null || true
done
echo "[$(date '+%F %T')] done"
