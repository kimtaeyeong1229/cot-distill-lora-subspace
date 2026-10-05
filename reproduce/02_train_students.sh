#!/usr/bin/env bash
# Stage 2: LoRA students (rank 16, alpha 32, 3 epochs, lr 2e-4, 32-sample batches) for the nine cells.
#
# Paper artifacts: every adapter the paper analyses.
#   full-data adapters   <cell>/runs/lambda<l>_seed{0,1,2}/      every table; Figures 2-5
#   data-half adapters   <cell>/runs/lambda<l>_half{A,B}[_seed{1,2}]/   data-only baseline B(lambda) of R(lambda)
#                        (Figure 2 middle; Tables 9 and 12) and the half-data reference of the share of the
#                        lambda0 region kept (Figure 2 bottom; Tables 1 and 9)
#
# Run inside the container: docker exec cot-distill-lora-bit bash reproduce/02_train_students.sh
# CELLS (default: all nine, as "<cell dir>:<student>") restricts the run. train_lora.py skips any run
# whose adapter already exists, so the script can be resumed.
#
# Training length: OLMo-2-1B has a 4,096-token context, so its runs use max_len 4096. TRAIN_TOKEN_BUDGET
# only sets the micro-batch size (the 32-sample optimizer batch and the loss are unchanged); the shipped
# runs used 4,096-16,384 depending on the GPU, and each run's value is recorded in its train_args.json.
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/lambda_lora_bits
source "${E}/config.sh"

for entry in ${CELLS:-${PAPER_CELLS}}; do
  tag="${entry%%:*}"; student="${entry#*:}"
  splits="$(cell_splits "${tag}")"
  if [[ "${student}" == *OLMo* ]]; then
    train=(MAX_MODEL_LEN=4096 TRAIN_MAX_LEN=4096 TRAIN_TOKEN_BUDGET=8192); halves=(TRAIN_MAX_LEN=4096 TRAIN_TOKEN_BUDGET=4096)
  else
    train=(TRAIN_MAX_LEN=8192 TRAIN_TOKEN_BUDGET=16384); halves=(TRAIN_MAX_LEN=8192 TRAIN_TOKEN_BUDGET=8192)
  fi
  # three seeds on the full training set, every lambda
  env "${train[@]}" SEEDS="${PAPER_SEEDS}" STAGES=train STUDENT="models/${student}" TAG="${tag}" SPLITS="${splits}" \
    bash "${E}/run_student.sh"
  # two disjoint halves per lambda, three seeds (seed 0 keeps the names lambda<l>_half{A,B})
  for seed in ${PAPER_SEEDS}; do
    env "${halves[@]}" SEED="${seed}" STUDENT="models/${student}" TAG="${tag}" SPLITS="${splits}" \
      bash "${E}/run_halves.sh"
  done
done
