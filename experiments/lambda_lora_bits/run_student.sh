#!/usr/bin/env bash
# Repeat the lambda sweep with a different student family, reusing the teacher CoT splits.
#
# Only the student changes: the problems, the teacher CoTs and every training
# hyper-parameter stay as in the Qwen run, so the two students are comparable.
#
#   docker exec cot-distill-lora-bit bash -c \
#     "STUDENT=models/SmolLM2-1.7B-Instruct TAG=sweep_smollm2 bash experiments/lambda_lora_bits/run_student.sh"
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/lambda_lora_bits
STUDENT="${STUDENT:?set STUDENT to the student checkpoint}"
TAG="${TAG:?set TAG for the output directory}"
LAMBDAS="${LAMBDAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
SPLITS="${SPLITS:-${EXP}/sweep/splits}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"       # SmolLM2 tops out at 8192
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-7680}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-8192}"  # lower it for students with a shorter context (OLMo-2: 4096)

OUT="${EXP}/${TAG}"
RUNS="${OUT}/runs"; BITS="${OUT}/bits"; LOGS="${OUT}/logs"
mkdir -p "${RUNS}" "${BITS}" "${LOGS}"
ln -sfn ../sweep/splits "${OUT}/splits"      # analyze.py reads split_stats.json from here

stage() { echo "[$(date '+%F %T')] $*"; }

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

CROSS=()
for lam in ${LAMBDAS}; do CROSS+=("lambda${lam}=${SPLITS}/lambda${lam}/held.jsonl"); done

for lam in ${LAMBDAS}; do
  name="lambda${lam}_seed0"
  stage "train ${name}"
  python "${EXP}/train_lora.py" --base "${STUDENT}" --train "${SPLITS}/lambda${lam}/train.jsonl" \
    --output "${RUNS}/${name}" --seed 0 --max_len "${TRAIN_MAX_LEN}" > "${LOGS}/train_${name}.log" 2>&1
  if [[ ! -f "${BITS}/${name}.json" ]]; then
    stage "bits ${name}"
    python "${EXP}/measure_bits.py" --base "${STUDENT}" --run_dir "${RUNS}/${name}" \
      --train "${SPLITS}/lambda${lam}/train.jsonl" --held "${SPLITS}/lambda${lam}/held.jsonl" \
      --cross "${CROSS[@]}" --cache_dir "${OUT}/cache/base_bits" \
      --max_len "${TRAIN_MAX_LEN}" --output "${BITS}/${name}.json" > "${LOGS}/bits_${name}.log" 2>&1
  fi
done

ADAPTERS=()
for lam in ${LAMBDAS}; do ADAPTERS+=("lambda${lam}_seed0=${RUNS}/lambda${lam}_seed0/adapter"); done
stage "eval val"
python "${EXP}/eval_student.py" --base "${STUDENT}" --include_base --adapters "${ADAPTERS[@]}" \
  --data "${SPLITS}/lambda0.0/held.jsonl" --out_dir "${OUT}/eval_val" --n_samples "${EVAL_SAMPLES}" \
  --max_model_len "${MAX_MODEL_LEN}" --max_new_tokens "${MAX_NEW_TOKENS}" > "${LOGS}/eval_val.log" 2>&1

stage "analyze"
python "${EXP}/analyze.py" --sweep_dir "${OUT}" --out_dir "${OUT}" > "${LOGS}/analyze.log" 2>&1

OWNER="$(stat -c '%u:%g' .)"
chown -R "${OWNER}" "${OUT}"
stage "done"
