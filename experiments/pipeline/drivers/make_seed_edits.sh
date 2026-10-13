#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/pipeline
source "${E}/config.sh"
for spec in ${SPECS:-${PAPER_SWITCH}}; do
  tag="${spec%%:*}"; cliff="${spec#*:}"
  for seed in ${PAPER_SEEDS}; do
    python3 "${E}/ablate_subspace.py" --tag "${tag}" --cliff "${cliff}" --seed "${seed}" --loo --region seed \
      --out "runs_edit_s${seed}" --variants rm_pre rm_decisive || exit 1
  done
done
