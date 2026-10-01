#!/usr/bin/env bash
# Think-off edit with the region built only from the other two seeds' lambda0 adapters (eigenvalue > 0.75),
# and its energy-matched control. Output: <cell>/runs_edit_s<seed>/lambda0.0_{rmseed,rmseed_other}
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
for spec in sweep:0.4 sweep_llama:0.4 sweep_olmo2:0.4 pairB/qwen0.5b:0.6 pairB/llama1b:0.6 pairB/olmo1b:0.6 pairD/qwen0.5b:0.6 pairD/llama1b:0.6 pairD/olmo1b:0.6; do
  tag="${spec%%:*}"; cliff="${spec#*:}"
  for seed in 0 1 2; do
    python3 "${E}/ablate_subspace.py" --tag "${tag}" --cliff "${cliff}" --seed "${seed}" --loo --region seed \
      --out "runs_edit_s${seed}" --variants rm_pre rm_decisive || exit 1
  done
done
