#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/pipeline
mkdir -p "${E}/pairB" "${E}/pairC"

if [[ "${PILOT:-0}" == 1 ]]; then
  bash "${E}/pilot_template_pair_a.sh"
  PAIR=B STAGE=pilot GPU_UTIL=0.90 bash "${E}/run_teachers_pair_bc.sh" > "${E}/pairB/pilot.log" 2>&1
  PAIR=C STAGE=pilot GPU_UTIL=0.90 bash "${E}/run_teachers_pair_bc.sh" > "${E}/pairC/pilot.log" 2>&1
fi

TEMPLATE=reasoning LAMBDAS="0.0 0.1 0.2 0.3 0.4 0.6 0.8 1.0" bash "${E}/run_teachers_pair_a.sh"

PAIR=B STAGE=full TEMPLATE=reasoning GPU_UTIL=0.90 bash "${E}/run_teachers_pair_bc.sh" > "${E}/pairB/full.log" 2>&1
PAIR=C STAGE=full TEMPLATE=reasoning GPU_UTIL=0.90 bash "${E}/run_teachers_pair_bc.sh" > "${E}/pairC/full.log" 2>&1

for pair in pairA pairB pairC; do
  python "${E}/make_halves.py" "${E}/${pair}/splits"
done
