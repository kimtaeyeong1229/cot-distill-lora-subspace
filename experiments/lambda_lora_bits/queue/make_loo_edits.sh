#!/usr/bin/env bash
# Writes the leave-one-out edited adapters for all nine cells: the regions are rebuilt without the
# seed-0 adapters that get edited, and pair C (directory pairD) uses the 0.6 transition in all cells.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
for spec in sweep:0.4 sweep_llama:0.4 sweep_olmo2:0.4 pairB/qwen0.5b:0.6 pairB/llama1b:0.6 pairB/olmo1b:0.6 \
            pairD/qwen0.5b:0.6 pairD/llama1b:0.6 pairD/olmo1b:0.6; do
  tag="${spec%%:*}"; cliff="${spec#*:}"
  python3 "${E}/ablate_subspace.py" --tag "${tag}" --cliff "${cliff}" --loo --out runs_ablation_loo || exit 1
  python3 "${E}/steer_subspace.py" --tag "${tag}" --cliff "${cliff}" --loo --out runs_steer_loo \
    --variants amp ampn ampV graft || exit 1
done
chown -R "$(stat -c '%u:%g' .)" "${E}" 2>/dev/null || true
echo "LOO edits done"
