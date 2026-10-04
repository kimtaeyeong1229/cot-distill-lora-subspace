#!/usr/bin/env bash
# Edited adapters for the section-5 re-run under the main evaluation setting (seeds 0-2). Each seed's
# edits use a region built without that seed's edited adapters (--loo), exactly as in runs_*_loo.
# Output: <cell>/runs_edit_s<seed>/{lambda0.0_rm_*, lambda1.0_rm_*, graft*, ampn*, ampV}
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
source "${E}/config.sh"   # gen_limits, PAPER_SWITCH, PAPER_SEEDS
SPECS="${SPECS:-${PAPER_SWITCH}}"   # <cell dir>:<switch lambda>
for spec in ${SPECS}; do
  tag="${spec%%:*}"; cliff="${spec#*:}"
  for seed in ${PAPER_SEEDS}; do
    python3 "${E}/ablate_subspace.py" --tag "${tag}" --cliff "${cliff}" --seed "${seed}" --loo \
      --out "runs_edit_s${seed}" --variants rm_pre rm_decisive rm_scale || exit 1
    python3 "${E}/steer_subspace.py" --tag "${tag}" --cliff "${cliff}" --seed "${seed}" --loo \
      --out "runs_edit_s${seed}" --variants ampn ampV graft --betas 1 2 || exit 1
  done
done
chown -R "$(stat -c '%u:%g' .)" "${E}" 2>/dev/null || true
echo "main-setting edits done"
