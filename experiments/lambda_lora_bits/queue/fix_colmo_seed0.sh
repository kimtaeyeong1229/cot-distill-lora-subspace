#!/usr/bin/env bash
# C/OLMo seed 0 at lambda 0.0/0.2/0.4 was trained with max_len 8,192, beyond OLMo-2's 4,096 context,
# while every other OLMo run (all seeds, halves, and the other lambdas) used 4,096. Those three runs
# predate the OLMo length fix and were skipped when the cell was re-trained. This retrains them at
# 4,096, re-evaluates them under the shared student setting, and rebuilds the cell's section-5 edits,
# whose region includes these adapters. The stale runs are kept under runs_stale_maxlen8192/.
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits; C=${E}/pairD/olmo1b
mkdir -p "${C}/runs_stale_maxlen8192" "${C}/eval_math500_stale_maxlen8192"
for lam in 0.0 0.2 0.4; do
  run="${C}/runs/lambda${lam}_seed0"
  if [[ -d "${run}" ]] && grep -q '"max_len": 8192' "${run}/train_args.json"; then
    mv "${run}" "${C}/runs_stale_maxlen8192/"
    [[ -f "${C}/eval_math500/lambda${lam}_seed0.json" ]] && mv "${C}/eval_math500/lambda${lam}_seed0.json" "${C}/eval_math500_stale_maxlen8192/"
  fi
done
SEEDS=0 LAMBDAS="0.0 0.2 0.4" STAGES=train MAX_MODEL_LEN=4096 TRAIN_MAX_LEN=4096 TRAIN_TOKEN_BUDGET=8192 \
  STUDENT=models/OLMo-2-1B-Instruct TAG=pairD/olmo1b SPLITS=${E}/pairD/splits bash ${E}/run_student.sh
grep -q '"max_len": 4096' ${C}/runs/lambda0.0_seed0/train_args.json
python3 ${E}/eval_student.py --base models/OLMo-2-1B-Instruct \
  --adapters lambda0.0_seed0=${C}/runs/lambda0.0_seed0/adapter lambda0.2_seed0=${C}/runs/lambda0.2_seed0/adapter \
             lambda0.4_seed0=${C}/runs/lambda0.4_seed0/adapter \
  --data ${E}/data/math500.parquet --out_dir ${C}/eval_math500 --limit 200 --n_samples 2 \
  --max_new_tokens 3840 --max_model_len 4096
rm -rf ${C}/runs_edit_s0 ${C}/runs_edit_s1 ${C}/runs_edit_s2
SPECS="pairD/olmo1b:0.6" bash ${E}/queue/make_main_edits.sh
chown -R "$(stat -c '%u:%g' .)" "${C}" 2>/dev/null || true
echo "C/OLMo seed 0 fixed"
