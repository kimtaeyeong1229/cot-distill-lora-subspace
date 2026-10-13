#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/pipeline
export FIG_OUT="${FIG_OUT:-figures}"
export FIG1_OUT="${FIG1_OUT:-${FIG_OUT}}"
mkdir -p "${FIG_OUT}"
python3 "${E}/paper_figures.py"
python3 "${E}/paper_appendix_figures.py"
python3 "${E}/paper_fig1_cde.py"
