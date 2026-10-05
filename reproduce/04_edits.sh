#!/usr/bin/env bash
# Stage 4: edit the adapters in weight space (remove / graft / amplify the shared region and the controls)
# and evaluate every edited adapter on MATH-500, GSM8K and GPQA-Diamond.
#
# Paper artifacts:
#   runs_edit_s{0,1,2}/ + eval_main_s{0,1,2}/, all seeds:
#     shared-region removal, controls, grafts (beta 1, 2), amplify   Figure 3; Tables 2-5, 7, 8, 11, 13, 15 and 17
#     lambda0.0_rmseed / rmseed_other     two-seed region and its control       Table 3
#     lambda0.0_gs_tau                    Gold-Switch top removal, tau = 0.6    Table 3
#     lambda0.0_gs_energy                 top removal, energy-matched           Tables 2, 11 and 15; App. A
#     merge0.5 / merge0.7                 merged with the lambda1 adapter       Table 3
#     lambda0.0_nothink                   think-skip prompt                     Table 3
#     lambda0.0_budget                    length-matched cut                    Tables 2, 3, 11 and 15
#   runs_graft_beta/ + eval_graft_beta/   grafts at beta = 2, 4, 8, seed 0      App. C
#   eval_gsm8k_s{0,1,2}/, eval_gpqa_diamond_s{0,1,2}/                           Table 3; Figure 4
#
# Run inside the container after stages 2 and 3: docker exec cot-distill-lora-bit bash reproduce/04_edits.sh
# The edits need all three seeds of a cell: each seed's region is built without that seed's own adapter.
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/lambda_lora_bits
source "${E}/config.sh"

MAIN="lambda0.0_rm_pre lambda0.0_rm_decisive lambda0.0_rm_scale lambda1.0_rm_pre graft1 graft_orth1 graft_full1 graft2 graft_orth2 graft_full2 ampn1 ampn4 ampV"
EXTRA="lambda0.0_rmseed lambda0.0_rmseed_other lambda0.0_gs_tau lambda0.0_gs_energy merge0.5 merge0.7"
BETA="graft2 graft4 graft8 graft_orth2 graft_orth4 graft_orth8 graft_full2 graft_full4 graft_full8"

# --- 4a. edited adapters (CPU/GPU linear algebra, minutes per cell) ---------------------------------
SPECS="${PAPER_SWITCH}" bash "${E}/drivers/make_main_edits.sh"     # rm_pre, rm_decisive, rm_scale, ampn, ampV, graft beta 1 and 2
bash "${E}/drivers/make_seed_edits.sh"                               # rmseed, rmseed_other
bash "${E}/drivers/make_baseline_edits.sh"                           # gs_tau, gs_energy, merge0.5, merge0.7
for spec in ${PAPER_SWITCH}; do                                      # grafts at beta 2, 4, 8 for seed 0
  python3 "${E}/steer_subspace.py" --tag "${spec%%:*}" --cliff "${spec#*:}" --seed 0 --loo \
    --variants graft --betas 2 4 8 --out runs_graft_beta
done

# --- 4b. evaluation (vLLM, first 200 problems x 2 samples) ------------------------------------------
for entry in ${CELLS:-${PAPER_CELLS}}; do
  for s in ${PAPER_SEEDS}; do
    LIMIT=200 SAMPLES=2 RUNS="runs_edit_s${s}" EVAL="eval_main_s${s}" VARIANTS="${MAIN} ${EXTRA}" CELLS="${entry}" \
      bash "${E}/drivers/run_steer.sh"
  done
  RUNS=runs_graft_beta EVAL=eval_graft_beta VARIANTS="${BETA}" CELLS="${entry}" bash "${E}/drivers/run_steer.sh"
  CELLS="${entry}" bash "${E}/drivers/run_baselines.sh"              # needs eval_main_s*/lambda0.0_rm_pre.json
  CELLS="${entry}" bash "${E}/drivers/run_bench_edits.sh"            # GSM8K and GPQA-Diamond
done
