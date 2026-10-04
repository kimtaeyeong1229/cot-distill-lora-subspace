#!/usr/bin/env bash
# Stage 6: every number in the paper's tables, as result JSONs in experiments/ (no generation; CPU is
# enough, the adapter analyses run faster with a GPU-enabled torch).
#
# Paper artifacts (printed numbers; README.md has the full map):
#   think_rate.json                       Figure 2 (top); Tables 2, 3 and 18
#   main_density.json, main_edits.json    per-seed inputs of seed_spread.json
#   seed_spread.json                      Tables 2, 6-11 and 20; Figure 3
#   teacher_density.json                  Table 2 (teacher markers, r)
#   phi_resample_s012.json                Figure 2 (middle); Tables 3 and 16
#   alignment_s012.json                   Table 4 (think rate)
#   alignment_markers_s012.json           Table 4 (teacher markers)
#   consensus_pre.json, consensus_post.json   Table 5
#   coverage_by_lambda.json               Figure 2 (bottom); Table 18
#   phi_xseed.json                        Table 16 (lambda0 cross-seed overlap)
#   baselines.json                        Figure 3; Tables 1, 7, 8, 15 and 19
#   base_students.json                    Tables 1 and 15 (untrained students)
#   graft_beta.json                       Table 21
#   ood.json                              Tables 1, 13 and 14
#   sae_llama.json                        Table 12
#   seed_region_overlap.json              Section 6 / appendix (two-seed region covers 54-60%)
#   top_overlap.json                      Table 19
#   threshold_sensitivity.json            Tables 16 and 17
#   degeneration.json                     Tables 20 and 21
#   edit_stats.json                       Table 22
#   length_confound.json                  Table 18
#   reproducibility.lock.json             model commits and library versions (appendix)
#
# Run from anywhere: bash reproduce/06_analysis.sh  (inside the container for the adapter analyses).
# REPO_ROOT and RESULTS_DIR (see experiments/lambda_lora_bits/config.py) redirect inputs and outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/lambda_lora_bits
py() { echo "== $*"; python3 "${E}/$@"; }

# teacher side
py measure_think_rate.py
# student outputs, unedited and edited (seed_spread runs the two scripts above it once per seed)
py analyze_main_density.py
py analyze_main_edits.py
py analyze_seed_spread.py
py analyze_teacher_density.py                       # needs seed_spread.json
py analyze_baselines.py
py analyze_base_students.py
py analyze_graft_beta.py
py analyze_ood.py
py analyze_sae_llama.py
py analyze_degeneration.py
py analyze_edit_stats.py                             # needs seed_spread.json and baselines.json
# adapter geometry (reads the LoRA weights)
py analyze_resample.py                               # seeds 0-2 -> phi_resample_s012.json
py analyze_alignment.py                              # think rate -> alignment_s012.json
py analyze_alignment.py --teacher markers            # -> alignment_markers_s012.json
py analyze_consensus.py --side pre
py analyze_consensus.py --side post
py analyze_coverage_by_lambda.py
py analyze_phi_xseed.py
py analyze_seed_region_overlap.py
py analyze_top_overlap.py
py analyze_threshold_sensitivity.py
py analyze_length_confound.py                        # needs think_rate, phi_resample_s012, coverage_by_lambda
# environment record
py make_lockfile.py
