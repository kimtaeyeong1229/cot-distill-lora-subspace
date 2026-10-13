#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/pipeline
source "${E}/config.sh"

MAIN="lambda0.0_rm_pre lambda0.0_rm_decisive lambda0.0_rm_scale lambda1.0_rm_pre graft1 graft_orth1 graft_full1 graft2 graft_orth2 graft_full2 ampn1 ampn4 ampV"
EXTRA="lambda0.0_rmseed lambda0.0_rmseed_other lambda0.0_gs_tau lambda0.0_gs_energy merge0.5 merge0.7 lambda0.0_rm_spanrand lambda0.0_rm_post"
BETA="graft2 graft4 graft8 graft_orth2 graft_orth4 graft_orth8 graft_full2 graft_full4 graft_full8"

SPECS="${PAPER_SWITCH}" bash "${E}/drivers/make_main_edits.sh"
bash "${E}/drivers/make_seed_edits.sh"
bash "${E}/drivers/make_baseline_edits.sh"
for spec in ${PAPER_SWITCH}; do
  python3 "${E}/steer_subspace.py" --tag "${spec%%:*}" --cliff "${spec#*:}" --seed 0 --loo \
    --variants graft --betas 2 4 8 --out runs_graft_beta
done

for entry in ${CELLS:-${PAPER_CELLS}}; do
  for s in ${PAPER_SEEDS}; do
    LIMIT=200 SAMPLES=2 RUNS="runs_edit_s${s}" EVAL="eval_main_s${s}" VARIANTS="${MAIN} ${EXTRA}" CELLS="${entry}" \
      bash "${E}/drivers/run_steer.sh"
  done
  RUNS=runs_graft_beta EVAL=eval_graft_beta VARIANTS="${BETA}" CELLS="${entry}" bash "${E}/drivers/run_steer.sh"
  CELLS="${entry}" bash "${E}/drivers/run_baselines.sh"
  CELLS="${entry}" bash "${E}/drivers/run_bench_edits.sh"
done
