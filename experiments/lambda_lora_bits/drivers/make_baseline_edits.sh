#!/usr/bin/env bash
# Training-free comparison methods for the think-off edit: Gold-Switch top
# removal of the lambda0 adapter (gs_tau at tau = 0.6, gs_energy matched to the shared-region removal) and
# linear merges of the lambda0 and lambda1 adapters (merge0.5, merge0.7).
# Output: <cell>/runs_edit_s<seed>/lambda0.0_{gs_tau,gs_energy}, <cell>/runs_edit_s<seed>/merge{0.5,0.7}
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
source "${E}/config.sh"   # gen_limits, PAPER_SWITCH, PAPER_SEEDS
for spec in ${SPECS:-${PAPER_SWITCH}}; do
  tag=${spec%%:*}; cliff=${spec#*:}
  for seed in ${PAPER_SEEDS}; do
    python3 $E/ablate_subspace.py --tag $tag --cliff $cliff --seed $seed --loo --sources 0.0 --out runs_edit_s$seed --variants gs_tau gs_energy || exit 1
    python3 $E/merge_adapters.py --tag $tag --seed $seed || exit 1
  done
done
