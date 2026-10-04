#!/usr/bin/env bash
# Think-off edit with the region built only from the other two seeds' lambda0 adapters (eigenvalue > 0.75),
# and its energy-matched control. Output: <cell>/runs_edit_s<seed>/lambda0.0_{rmseed,rmseed_other}
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
source "${E}/config.sh"   # gen_limits, PAPER_SWITCH, PAPER_SEEDS
for spec in ${SPECS:-${PAPER_SWITCH}}; do
  tag="${spec%%:*}"; cliff="${spec#*:}"
  for seed in ${PAPER_SEEDS}; do
    python3 "${E}/ablate_subspace.py" --tag "${tag}" --cliff "${cliff}" --seed "${seed}" --loo --region seed \
      --out "runs_edit_s${seed}" --variants rm_pre rm_decisive || exit 1
  done
done
