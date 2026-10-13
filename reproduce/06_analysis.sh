#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/pipeline
py() { echo "== $*"; python3 "${E}/$@"; }

py measure_think_rate.py
py analyze_main_density.py
py analyze_main_edits.py
py analyze_seed_spread.py
py analyze_teacher_density.py
py analyze_baselines.py
py analyze_base_students.py
py analyze_graft_beta.py
py analyze_ood.py
py analyze_sae_llama.py
py analyze_degeneration.py
py analyze_edit_stats.py
py analyze_resample.py
py analyze_alignment.py
py analyze_alignment.py --teacher markers
py analyze_alignment.py --teacher length
py analyze_consensus.py --side pre
py analyze_consensus.py --side post
py analyze_coverage_by_lambda.py
py analyze_phi_xseed.py
py analyze_seed_region_overlap.py
py analyze_top_overlap.py
py analyze_threshold_sensitivity.py
py analyze_length_confound.py
py analyze_lambda0_retention.py
py analyze_region_anatomy.py
py analyze_energy_match.py
py analyze_symmetric_R.py
py analyze_switch_regression.py
py analyze_pairA_seed_ratio.py
py analyze_pairA_crossseed.py
py analyze_two_seed_floor.py
py analyze_supplementary_stats.py
py analyze_think_rate_completed.py
py analyze_agreement_normalised.py
py analyze_paired_bootstrap.py
py analyze_teacher_structure.py
py analyze_student_structure.py
py analyze_removal_vs_untrained.py
py analyze_region_logit_lens.py
py analyze_summary_stats.py
py analyze_additional_stats.py
py make_examples_table.py
py paper_tables.py
py make_lockfile.py
