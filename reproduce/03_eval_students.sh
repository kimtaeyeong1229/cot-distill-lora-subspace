#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/pipeline
source "${E}/config.sh"

for entry in ${CELLS:-${PAPER_CELLS}}; do
  tag="${entry%%:*}"; student="${entry#*:}"
  gen_limits "${student}"
  case "${tag}" in
    pairA/qwen0.5b|pairA/llama1b)
      CELL="${tag}" STUDENT="${student}" SEEDS="${PAPER_SEEDS}" bash "${E}/drivers/run_math500.sh" ;;
    *)
      MAX_MODEL_LEN="${MAXLEN}" MAX_NEW_TOKENS="${NEWTOK}" STAGES=eval500 SEEDS="${PAPER_SEEDS}" \
        STUDENT="models/${student}" TAG="${tag}" SPLITS="$(cell_splits "${tag}")" bash "${E}/run_student.sh" ;;
  esac
  RUNS=runs_ablation_loo EVAL=eval_ablation_loo CELLS="${entry}" bash "${E}/drivers/run_ablation.sh"
done

bash "${E}/drivers/run_base_eval.sh"
