#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/pipeline
source "${E}/config.sh"
for spec in ${SPECS:-${PAPER_SWITCH}}; do
  tag=${spec%%:*}; cliff=${spec#*:}
  for seed in ${PAPER_SEEDS}; do
    python3 $E/ablate_subspace.py --tag $tag --cliff $cliff --seed $seed --loo --sources 0.0 --out runs_edit_s$seed --variants gs_tau gs_energy || exit 1
    python3 $E/merge_adapters.py --tag $tag --seed $seed || exit 1
  done
done
