#!/usr/bin/env bash
# Pair A teachers: interpolate DeepSeek-R1-Distill-Qwen-1.5B (lambda = 0) and Qwen2.5-Math-1.5B-Instruct
# (lambda = 1), generate one CoT per pool problem at every lambda, and build the shared train/held splits.
# Pairs B and C use run_teachers_pair_bc.sh, which follows the same procedure. Students are trained afterwards with
# run_student.sh (reproduce/02_train_students.sh).
#
# The template was chosen with pilot_template_pair_a.sh: every pair A teacher uses the R1 (reasoning) tokenizer
# and chat template. The shared split was built over eight lambdas (0.1 and 0.3 included), so the same
# list must be passed to rebuild sweep/splits/split_ids.json exactly:
#
#   docker exec cot-distill-lora-bit bash -c "TEMPLATE=reasoning LAMBDAS='0.0 0.1 0.2 0.3 0.4 0.6 0.8 1.0' \
#     bash experiments/lambda_lora_bits/run_teachers_pair_a.sh"
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/lambda_lora_bits
REASONING=models/DeepSeek-R1-Distill-Qwen-1.5B
INSTRUCT=models/Qwen2.5-Math-1.5B-Instruct
STUDENT=models/Qwen2.5-0.5B-Instruct   # tokenizer for the length filter in build_splits.py

TEMPLATE="${TEMPLATE:?set TEMPLATE=reasoning or TEMPLATE=instruct (chosen by pilot_template_pair_a.sh; the paper uses reasoning)}"
LAMBDAS="${LAMBDAS:-0.0 0.1 0.2 0.3 0.4 0.6 0.8 1.0}"
N_HELD="${N_HELD:-300}"
N_TRAIN="${N_TRAIN:-}"

SWEEP="${EXP}/sweep"
TEACHERS="${SWEEP}/teachers"
GEN="${SWEEP}/cot"
SPLITS="${SWEEP}/splits"
LOGS="${SWEEP}/logs"
mkdir -p "${TEACHERS}" "${GEN}" "${LOGS}"

stage() { echo "[$(date '+%F %T')] $*"; }

for model in "${REASONING}" "${INSTRUCT}" "${STUDENT}"; do
  [[ -f "${model}/config.json" ]] || python "${EXP}/fetch_model.py" "${model}"
done

# --- 1. teachers and CoT pool ----------------------------------------------
for lam in ${LAMBDAS}; do
  if [[ ! -f "${TEACHERS}/lambda${lam}/model.safetensors" ]]; then
    stage "interpolate lambda=${lam}"
    python "${EXP}/interpolate.py" --reasoning "${REASONING}" --instruct "${INSTRUCT}" \
      --lam "${lam}" --output "${TEACHERS}/lambda${lam}" --tokenizer_from "${TEMPLATE}"
  fi
  stage "generate lambda=${lam}"
  python "${EXP}/generate.py" --model "${TEACHERS}/lambda${lam}" --problems "${EXP}/data/pool.parquet" \
    --output "${GEN}/lambda${lam}.jsonl" > "${LOGS}/generate_lambda${lam}.log" 2>&1
  grep -E "^\[" "${LOGS}/generate_lambda${lam}.log" | tail -1 || true
done

# --- 2. shared splits ----------------------------------------------------------
if [[ ! -f "${SPLITS}/split_stats.json" ]]; then
  stage "build splits"
  python "${EXP}/build_splits.py" --gen_dir "${GEN}" --lambdas "${LAMBDAS}" --student "${STUDENT}" \
    --out_dir "${SPLITS}" --n_held "${N_HELD}" ${N_TRAIN:+--n_train "${N_TRAIN}"} > "${LOGS}/build_splits.log" 2>&1
fi

# The container runs as root; hand the outputs back to the owner of the project directory.
chown -R "$(stat -c '%u:%g' .)" "${SWEEP}" 2>/dev/null || true
stage "pair A teachers done"
