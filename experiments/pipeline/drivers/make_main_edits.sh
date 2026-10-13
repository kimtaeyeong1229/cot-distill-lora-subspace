#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/pipeline
source "${E}/config.sh"
SPECS="${SPECS:-${PAPER_SWITCH}}"
for spec in ${SPECS}; do
  tag="${spec%%:*}"; cliff="${spec#*:}"
  for seed in ${PAPER_SEEDS}; do
    python3 "${E}/ablate_subspace.py" --tag "${tag}" --cliff "${cliff}" --seed "${seed}" --loo \
      --out "runs_edit_s${seed}" --variants rm_pre rm_decisive rm_scale rm_spanrand rm_post || exit 1
    python3 "${E}/steer_subspace.py" --tag "${tag}" --cliff "${cliff}" --seed "${seed}" --loo \
      --out "runs_edit_s${seed}" --variants ampn ampV graft --betas 1 2 || exit 1
  done
done
chown -R "$(stat -c '%u:%g' .)" "${E}" 2>/dev/null || true
echo "main-setting edits done"
