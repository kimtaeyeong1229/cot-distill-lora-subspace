#!/usr/bin/env bash
# Full experiment (v2 ordering): Long CoT (lambda=0.0) and Short CoT (lambda=1.0) are trained,
# measured and compared first; the other lambdas then follow the same process.
#
# Full experiment: interpolated teachers -> CoT pool -> shared splits -> LoRA per lambda
# -> bits (CL, W, dW) + accuracy -> best interior lambda -> extra seeds for the three arms.
#
# Arms: Long CoT (lambda=0.0), Short CoT (lambda=1.0) and the interior lambda with the best
# held-out accuracy at seed 0. Test accuracy (MATH-500) is only read after that choice.
#
#   docker exec -d cot-distill-lora-bit bash -c \
#     "TEMPLATE=instruct bash experiments/lambda_lora_bits/run_sweep_v2.sh > experiments/logs/sweep.log 2>&1"
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/lambda_lora_bits
REASONING=models/DeepSeek-R1-Distill-Qwen-1.5B
INSTRUCT=models/Qwen2.5-Math-1.5B-Instruct
STUDENT=models/Qwen2.5-0.5B-Instruct

TEMPLATE="${TEMPLATE:?set TEMPLATE=instruct or TEMPLATE=reasoning (chosen by pilot_template.sh)}"
LAMBDAS="${LAMBDAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
EXTRA_SEEDS="${EXTRA_SEEDS:-1 2}"
N_HELD="${N_HELD:-300}"
N_TRAIN="${N_TRAIN:-}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"

SWEEP="${EXP}/sweep"
TEACHERS="${SWEEP}/teachers"
GEN="${SWEEP}/cot"
SPLITS="${SWEEP}/splits"
RUNS="${SWEEP}/runs"
BITS="${SWEEP}/bits"
LOGS="${SWEEP}/logs"
mkdir -p "${TEACHERS}" "${GEN}" "${RUNS}" "${BITS}" "${LOGS}"

stage() { echo "[$(date '+%F %T')] $*"; }

# --- 1. teachers and CoT pool ----------------------------------------------
# GEN_LAMBDAS may extend LAMBDAS so extra spectrum points are ready if the grid grows.
GEN_LAMBDAS="${GEN_LAMBDAS:-${LAMBDAS}}"
for lam in ${GEN_LAMBDAS}; do
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

if [[ "${STOP_AFTER_GENERATION:-0}" == "1" ]]; then
  stage "generation finished (STOP_AFTER_GENERATION=1)"
  exit 0
fi

# --- 2. shared splits ----------------------------------------------------------
if [[ ! -f "${SPLITS}/split_stats.json" ]]; then
  stage "build splits"
  python "${EXP}/build_splits.py" --gen_dir "${GEN}" --lambdas "${LAMBDAS}" --student "${STUDENT}" \
    --out_dir "${SPLITS}" --n_held "${N_HELD}" ${N_TRAIN:+--n_train "${N_TRAIN}"} > "${LOGS}/build_splits.log" 2>&1
fi

CROSS=()
for lam in ${LAMBDAS}; do CROSS+=("lambda${lam}=${SPLITS}/lambda${lam}/held.jsonl"); done

train_and_measure() {
  local lam="$1" seed="$2"
  local name="lambda${lam}_seed${seed}"
  stage "train ${name}"
  python "${EXP}/train_lora.py" --base "${STUDENT}" --train "${SPLITS}/lambda${lam}/train.jsonl" \
    --output "${RUNS}/${name}" --seed "${seed}" > "${LOGS}/train_${name}.log" 2>&1
  if [[ ! -f "${BITS}/${name}.json" ]]; then
    stage "bits ${name}"
    python "${EXP}/measure_bits.py" --base "${STUDENT}" --run_dir "${RUNS}/${name}" \
      --train "${SPLITS}/lambda${lam}/train.jsonl" --held "${SPLITS}/lambda${lam}/held.jsonl" \
      --cross "${CROSS[@]}" --cache_dir "${SWEEP}/cache/base_bits" \
      --output "${BITS}/${name}.json" > "${LOGS}/bits_${name}.log" 2>&1
  fi
}

evaluate() {  # evaluate <split> <data> <names...>
  local split="$1" data="$2"; shift 2
  local adapters=()
  for name in "$@"; do adapters+=("${name}=${RUNS}/${name}/adapter"); done
  stage "eval ${split}: $*"
  python "${EXP}/eval_student.py" --base "${STUDENT}" --include_base --adapters "${adapters[@]}" \
    --data "${data}" --out_dir "${SWEEP}/eval_${split}" --n_samples "${EVAL_SAMPLES}" \
    > "${LOGS}/eval_${split}_$(date +%s).log" 2>&1
}

# --- 3a. Long (0.0) and Short (1.0) CoT first, with an early bits comparison -----
ENDPOINTS="${ENDPOINTS:-0.0 1.0}"
EARLY_RUNS=()
for lam in ${ENDPOINTS}; do
  train_and_measure "${lam}" 0
  EARLY_RUNS+=("lambda${lam}_seed0")
done
evaluate val "${SPLITS}/lambda0.0/held.jsonl" "${EARLY_RUNS[@]}"
stage "early report: Long vs Short CoT"
python "${EXP}/analyze.py" --sweep_dir "${SWEEP}" --out_dir "${SWEEP}/early_long_vs_short" > "${LOGS}/early_report.log" 2>&1
stage "early Long-vs-Short report ready: ${SWEEP}/early_long_vs_short/report.md"

# --- 3b. remaining lambdas at seed 0 (same process) --------------------------------
SWEEP_RUNS=()
for lam in ${LAMBDAS}; do
  if [[ " ${ENDPOINTS} " != *" ${lam} "* ]]; then
    train_and_measure "${lam}" 0
  fi
  SWEEP_RUNS+=("lambda${lam}_seed0")
done
evaluate val "${SPLITS}/lambda0.0/held.jsonl" "${SWEEP_RUNS[@]}"

# --- 4. choose the best interior lambda on held-out accuracy ---------------------
BEST=$(python - "${SWEEP}" "${LAMBDAS}" "${ENDPOINTS}" <<'PY'
import json, sys
from pathlib import Path
sweep, lambdas = Path(sys.argv[1]), sys.argv[2].split()
acc = {lam: json.loads((sweep / "eval_val" / f"lambda{lam}_seed0.json").read_text())["summary"]["accuracy"] for lam in lambdas}
endpoints = sys.argv[3].split()
interior = {lam: a for lam, a in acc.items() if lam not in endpoints}
best = max(interior, key=interior.get)
choice = {"val_accuracy_seed0": acc, "best_interior_lambda": best, "best_overall_lambda": max(acc, key=acc.get)}
(sweep / "best_lambda.json").write_text(json.dumps(choice, indent=2))
print(best)
PY
)
stage "best interior lambda = ${BEST}"

# --- 5. extra seeds for the three arms, then test --------------------------------
ARMS="0.0 ${BEST} 1.0"
ARM_RUNS=()
for lam in ${ARMS}; do
  ARM_RUNS+=("lambda${lam}_seed0")
  for seed in ${EXTRA_SEEDS}; do
    train_and_measure "${lam}" "${seed}"
    ARM_RUNS+=("lambda${lam}_seed${seed}")
  done
done
evaluate test "${EXP}/data/math500.parquet" "${ARM_RUNS[@]}"

stage "analyze"
python "${EXP}/analyze.py" --sweep_dir "${SWEEP}"

# The container runs as root; hand every file in the mounted project back to its owner (lab).
OWNER="$(stat -c '%u:%g' .)"
stage "chown project files to ${OWNER}"
chown -R "${OWNER}" .
stage "done"
