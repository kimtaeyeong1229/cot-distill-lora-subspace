#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/pipeline
source "${E}/config.sh"

for entry in ${CELLS:-${PAPER_CELLS}}; do
  tag="${entry%%:*}"; student="${entry#*:}"
  splits="$(cell_splits "${tag}")"
  if [[ "${student}" == *OLMo* ]]; then
    train=(MAX_MODEL_LEN=4096 TRAIN_MAX_LEN=4096 TRAIN_TOKEN_BUDGET=8192); halves=(TRAIN_MAX_LEN=4096 TRAIN_TOKEN_BUDGET=4096)
  else
    train=(TRAIN_MAX_LEN=8192 TRAIN_TOKEN_BUDGET=16384); halves=(TRAIN_MAX_LEN=8192 TRAIN_TOKEN_BUDGET=8192)
  fi
  env "${train[@]}" SEEDS="${PAPER_SEEDS}" STAGES=train STUDENT="models/${student}" TAG="${tag}" SPLITS="${splits}" \
    bash "${E}/run_student.sh"
  for seed in ${PAPER_SEEDS}; do
    env "${halves[@]}" SEED="${seed}" STUDENT="models/${student}" TAG="${tag}" SPLITS="${splits}" \
      bash "${E}/run_halves.sh"
  done
done
