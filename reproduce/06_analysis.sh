#!/usr/bin/env bash
# Stage 6: every number in the paper's tables, captions and text, as result JSONs in experiments/ (no
# generation; CPU is enough, the adapter analyses run faster with a GPU-enabled torch).
#
# Paper artifacts (printed numbering of the current manuscript; README.md has the full map with JSON keys):
#   think_rate.json                       Figure 2 (top); input of Tables 1, 9 and 12
#   main_density.json, main_edits.json    Tables 2, 10, 11 and 17; per-seed inputs of seed_spread.json
#   seed_spread.json                      Tables 2, 5, 10, 11 and 17; Figures 1 (caption) and 3
#   teacher_density.json                  Table 10 (teacher-student r); input of Tables 6 and 12
#   baselines.json                        Tables 2, 3 and 11; Figures 3 and 5
#   base_students.json                    Table 3 (untrained students)
#   graft_beta.json                       App. C (beta = 4, 8)
#   ood.json                              Table 3; Figure 4
#   sae_llama.json                        Table 13 (bottom)
#   degeneration.json                     Table 17 (loops)
#   edit_stats.json                       Tables 2 and 15
#   phi_resample_s012.json                Figure 2 (middle); App. B
#   alignment_s012.json                   Table 12 (think rate)
#   alignment_markers_s012.json           Table 12 (teacher markers)
#   alignment_length_s012.json            Table 12 (training length)
#   consensus_pre.json, consensus_post.json   Table 16; Figure 1 caption
#   coverage_by_lambda.json               input of length_confound.json
#   phi_xseed.json                        Sec. 4.3, App. B
#   seed_region_overlap.json              not printed (equals threshold_sensitivity two_seed_region 0.75)
#   top_overlap.json                      App. A
#   threshold_sensitivity.json            Table 16; App. A; Sec. 6.2
#   length_confound.json                  Tables 1 and 9
#   lambda0_retention.json                Figure 2 (bottom); App. B
#   region_anatomy.json                   Table 4; App. A
#   energy_match.json                     Tables 2, 5 and 8; Figure 1 caption
#   paired_bootstrap.json                 Table 15; Sec. 5.1 and App. C (grades cached in
#                                         paired_bootstrap_grades.json, rebuilt when that file is absent)
#   symmetric_R.json                      App. B
#   switch_regression.json                App. B
#   pairA_seed_ratio.json                 App. B
#   pairA_crossseed.json                  Table 1 (caption); App. B
#   two_seed_floor.json                   Table 1 (caption); App. B
#   supplementary_stats.json              App. A and E
#   think_rate_completed.json             Table 12 (completed and training subsets)
#   teacher_structure.json                Table 6
#   student_structure.json                Table 13 (top)
#   removal_vs_untrained.json             Table 7
#   region_logit_lens.json                App. D
#   review_stats.json                     Tables 2, 5, 15 and 17; Sec. 5 and Limitations
#   c13_additions.json                    App. B and C
#   agreement_normalised.json             Sec. 4.3
#   examples.json (+ examples_table.tex)  Table 14
#   reproducibility.lock.json             model commits and library versions (appendix)
#
# Run from anywhere: bash reproduce/06_analysis.sh  (inside the container for the adapter analyses and the
# scripts that read parquet files). REPO_ROOT and RESULTS_DIR (see experiments/lambda_lora_bits/config.py)
# redirect inputs and outputs; experiments/manuscript_audit/run_all.sh then checks the manuscript against them.
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
py analyze_alignment.py --teacher length             # -> alignment_length_s012.json
py analyze_consensus.py --side pre
py analyze_consensus.py --side post
py analyze_coverage_by_lambda.py
py analyze_phi_xseed.py
py analyze_seed_region_overlap.py
py analyze_top_overlap.py
py analyze_threshold_sensitivity.py
py analyze_length_confound.py                        # needs think_rate, teacher_density, phi_resample_s012, coverage_by_lambda
py analyze_lambda0_retention.py                      # half-data reference of the lambda0 region (Figure 2 bottom)
py analyze_region_anatomy.py
py analyze_energy_match.py                           # reads the edited adapters of stage 4
# interval and seed-level checks built on length_confound.json
py analyze_symmetric_R.py
py analyze_switch_regression.py
py analyze_pairA_seed_ratio.py
py analyze_pairA_crossseed.py                        # also needs teacher_density.json
py analyze_two_seed_floor.py
py analyze_supplementary_stats.py
py analyze_think_rate_completed.py                   # needs alignment_s012.json
py analyze_agreement_normalised.py                   # needs phi_xseed.json
# response-level analyses of the stored generations
py analyze_paired_bootstrap.py
py analyze_teacher_structure.py
py analyze_student_structure.py                      # needs the untrained-student generations (base_eval/)
py analyze_removal_vs_untrained.py
py analyze_region_logit_lens.py                      # reads the student checkpoints in models/
py analyze_review_stats.py                           # needs main_edits, sae_llama, paired_bootstrap, length_confound, energy_match, seed_spread, degeneration
py analyze_c13_additions.py                          # needs lambda0_retention, length_confound, main_edits
py make_examples_table.py                            # Table 14: examples.json and examples_table.tex
py paper_tables.py                                   # Tables 1, 2, 9 and 15 into $MANUSCRIPT_DIR/tables/
# environment record
py make_lockfile.py
