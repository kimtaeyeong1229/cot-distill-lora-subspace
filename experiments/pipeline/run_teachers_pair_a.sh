#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/pipeline
REASONING=models/DeepSeek-R1-Distill-Qwen-1.5B
INSTRUCT=models/Qwen2.5-Math-1.5B-Instruct
STUDENT=models/Qwen2.5-0.5B-Instruct

TEMPLATE="${TEMPLATE:?set TEMPLATE=reasoning or TEMPLATE=instruct (chosen by pilot_template_pair_a.sh; the paper uses reasoning)}"
LAMBDAS="${LAMBDAS:-0.0 0.1 0.2 0.3 0.4 0.6 0.8 1.0}"
N_HELD="${N_HELD:-300}"
N_TRAIN="${N_TRAIN:-}"

PAIR_A="${EXP}/pairA"
TEACHERS="${PAIR_A}/teachers"
GEN="${PAIR_A}/cot"
SPLITS="${PAIR_A}/splits"
LOGS="${PAIR_A}/logs"
mkdir -p "${TEACHERS}" "${GEN}" "${LOGS}"

stage() { echo "[$(date '+%F %T')] $*"; }

for model in "${REASONING}" "${INSTRUCT}" "${STUDENT}"; do
  [[ -f "${model}/config.json" ]] || python "${EXP}/fetch_model.py" "${model}"
done

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

if [[ ! -f "${SPLITS}/split_stats.json" ]]; then
  stage "build splits"
  python "${EXP}/build_splits.py" --gen_dir "${GEN}" --lambdas "${LAMBDAS}" --student "${STUDENT}" \
    --out_dir "${SPLITS}" --n_held "${N_HELD}" ${N_TRAIN:+--n_train "${N_TRAIN}"} > "${LOGS}/build_splits.log" 2>&1
fi

chown -R "$(stat -c '%u:%g' .)" "${PAIR_A}" 2>/dev/null || true
stage "pair A teachers done"
