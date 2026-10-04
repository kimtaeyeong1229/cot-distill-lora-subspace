#!/usr/bin/env bash
# Stage 7: the paper's figures.
#
# Paper artifacts:
#   Figure 1  paper_fig1_cde.py -> fig1_cde.pptx (schematic; no data; export to PDF with PowerPoint or
#             LibreOffice). Needs Arial metrics: put the TTFs in FIG1_FONT_DIR or use Liberation Sans.
#   Figure 2  paper_figures.py  -> fig2_switch.pdf   (think_rate, phi_resample_s012, coverage_by_lambda)
#   Figure 3  paper_figures.py  -> fig3_states.pdf   (seed_spread, baselines)
#
# Only the result JSONs in experiments/ are needed, so this runs on the host:
#   FIG_OUT=/tmp/figs bash reproduce/07_figures.sh
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/lambda_lora_bits
export FIG_OUT="${FIG_OUT:-figures}"
export FIG1_OUT="${FIG1_OUT:-${FIG_OUT}}"
mkdir -p "${FIG_OUT}"
python3 "${E}/paper_figures.py"
python3 "${E}/paper_fig1_cde.py"
