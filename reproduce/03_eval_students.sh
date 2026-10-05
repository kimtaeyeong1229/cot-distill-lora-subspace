#!/usr/bin/env bash
# Stage 3: MATH-500 generations of the unedited students, the untrained students and the seed-0
# reference adapters. Sampling: temperature 0.6, top-p 0.95, 7,680 new tokens (OLMo 3,840).
#
# Paper artifacts:
#   <cell>/eval_math500/ and, for A/Qwen and A/Llama, <cell>/eval_math500_7680/
#       student markers, length, truncation and accuracy across lambda: Table 6; the lambda0 / lambda1
#       rows of Tables 2, 3, 10, 12-15 and 17, Figure 3 (gray bars); Table 16
#   base_eval/<student>/eval_main_s{0,1,2}/base.json       untrained students: Tables 3, 10 and 17
#   <cell>/eval_ablation_loo/lambda0.0_base.json          seed-0 lambda0 reference of the beta = 2, 4, 8 grafts (App. C)
#
# Run inside the container: docker exec cot-distill-lora-bit bash reproduce/03_eval_students.sh
# The analyses read the first 200 problems x 2 samples of every evaluation (the shared student setting).
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/lambda_lora_bits
source "${E}/config.sh"

for entry in ${CELLS:-${PAPER_CELLS}}; do
  tag="${entry%%:*}"; student="${entry#*:}"
  gen_limits "${student}"
  case "${tag}" in
    sweep|sweep_llama)
      # pair A Qwen and Llama: re-run at the shared 7,680-token limit (their first run used 8,192)
      CELL="${tag}" STUDENT="${student}" SEEDS="${PAPER_SEEDS}" bash "${E}/drivers/run_math500.sh" ;;
    *)
      MAX_MODEL_LEN="${MAXLEN}" MAX_NEW_TOKENS="${NEWTOK}" STAGES=eval500 SEEDS="${PAPER_SEEDS}" \
        STUDENT="models/${student}" TAG="${tag}" SPLITS="$(cell_splits "${tag}")" bash "${E}/run_student.sh" ;;
  esac
  # unedited seed-0 lambda0 / lambda1 adapters at 200 x 2 (eval_ablation_loo/lambda{0.0,1.0}_base.json)
  RUNS=runs_ablation_loo EVAL=eval_ablation_loo CELLS="${entry}" bash "${E}/drivers/run_ablation.sh"
done

# untrained students, three sampling seeds
bash "${E}/drivers/run_base_eval.sh"
