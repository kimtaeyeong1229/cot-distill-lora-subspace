#!/usr/bin/env bash
# Edited adapters for the section-5 re-run under the main evaluation setting (seeds 0-2). Each seed's
# edits use a region built without that seed's edited adapters (--loo), exactly as in runs_*_loo.
# Output: <cell>/runs_edit_s<seed>/{lambda0.0_rm_*, lambda1.0_rm_*, graft*, ampn*, ampV}
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
SPECS="${SPECS:-sweep:0.4 sweep_llama:0.4 sweep_olmo2:0.4 pairB/qwen0.5b:0.6 pairB/llama1b:0.6 pairB/olmo1b:0.6 pairD/qwen0.5b:0.6 pairD/llama1b:0.6 pairD/olmo1b:0.6}"
for spec in ${SPECS}; do
  tag="${spec%%:*}"; cliff="${spec#*:}"
  for seed in 0 1 2; do
    python3 "${E}/ablate_subspace.py" --tag "${tag}" --cliff "${cliff}" --seed "${seed}" --loo \
      --out "runs_edit_s${seed}" --variants rm_pre rm_decisive rm_scale || exit 1
    python3 "${E}/steer_subspace.py" --tag "${tag}" --cliff "${cliff}" --seed "${seed}" --loo \
      --out "runs_edit_s${seed}" --variants ampn ampV graft --betas 1 2 || exit 1
  done
done
chown -R "$(stat -c '%u:%g' .)" "${E}" 2>/dev/null || true
echo "main-setting edits done"
