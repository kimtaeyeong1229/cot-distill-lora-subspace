#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/pipeline
STUDENT="${STUDENT:?set STUDENT to the student checkpoint}"
TAG="${TAG:?set TAG for the output directory}"
LAMBDAS="${LAMBDAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
SEEDS="${SEEDS:-0}"
STAGES="${STAGES:-train,eval500}"
SPLITS="${SPLITS:-${EXP}/pairA/splits}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-7680}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-8192}"
TRAIN_TOKEN_BUDGET="${TRAIN_TOKEN_BUDGET:-16384}"

OUT="${EXP}/${TAG}"
RUNS="${OUT}/runs"; LOGS="${OUT}/logs"
mkdir -p "${RUNS}" "${LOGS}"
ln -sfn "$(realpath --relative-to="${OUT}" "${SPLITS}")" "${OUT}/splits"

stage() { echo "[$(date '+%F %T')] $*"; }

if [[ ! -f "${STUDENT}/config.json" ]]; then
  stage "download ${STUDENT}"
  python "${EXP}/fetch_model.py" "${STUDENT}"
fi

stage "tokenizer length check for ${STUDENT}"
python - "${STUDENT}" "${SPLITS}" "${LAMBDAS}" "${EXP}" "${TRAIN_MAX_LEN}" <<'PY'
import sys
sys.path.insert(0, sys.argv[4])
from common import encode, load_tokenizer, read_jsonl

tokenizer = load_tokenizer(sys.argv[1])
limit = int(sys.argv[5])
for lam in sys.argv[3].split():
    rows = read_jsonl(f"{sys.argv[2]}/lambda{lam}/train.jsonl")
    lengths = [len(encode(tokenizer, r["problem"], r["response"], 10 ** 9)[0]) for r in rows]
    over = sum(length >= limit for length in lengths)
    lost = sum(max(0, length - limit) for length in lengths) / sum(lengths)
    print(f"lambda={lam}: mean={sum(lengths)/len(lengths):.0f} max={max(lengths)} "
          f"over_{limit}={over} ({over/len(lengths):.1%}) truncated_tokens={lost:.1%}", flush=True)
PY

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

for seed in ${SEEDS}; do
  for lam in ${LAMBDAS}; do
    has_stage train || continue
    name="lambda${lam}_seed${seed}"
    stage "train ${name}"
    python "${EXP}/train_lora.py" --base "${STUDENT}" --train "${SPLITS}/lambda${lam}/train.jsonl" \
      --output "${RUNS}/${name}" --seed "${seed}" --max_len "${TRAIN_MAX_LEN}" --token_budget "${TRAIN_TOKEN_BUDGET}" > "${LOGS}/train_${name}.log" 2>&1
  done
done

ALL_ADAPTERS=()
for seed in ${SEEDS}; do
  for lam in ${LAMBDAS}; do ALL_ADAPTERS+=("lambda${lam}_seed${seed}=${RUNS}/lambda${lam}_seed${seed}/adapter"); done
done

if has_stage eval500; then
  stage "eval math500 (${#ALL_ADAPTERS[@]} adapters)"
  python "${EXP}/eval_student.py" --base "${STUDENT}" --include_base --adapters "${ALL_ADAPTERS[@]}" \
    --data "${EXP}/data/math500.parquet" --out_dir "${OUT}/eval_math500" --n_samples "${EVAL_SAMPLES}" \
    --max_model_len "${MAX_MODEL_LEN}" --max_new_tokens "${MAX_NEW_TOKENS}" > "${LOGS}/eval_math500.log" 2>&1
fi

OWNER="$(stat -c '%u:%g' .)"
chown -R "${OWNER}" "${OUT}"
stage "done"
