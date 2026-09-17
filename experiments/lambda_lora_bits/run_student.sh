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
SEEDS="${SEEDS:-0}"                          # extra seeds give the null distribution for subspace overlap
STAGES="${STAGES:-train,bits,eval,analyze}"  # drop stages for seed replicates that only need the adapter
SPLITS="${SPLITS:-${EXP}/sweep/splits}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"       # SmolLM2 tops out at 8192
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-7680}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-8192}"
TRAIN_TOKEN_BUDGET="${TRAIN_TOKEN_BUDGET:-16384}"  # micro-batch size in padded tokens; lower it to fit memory, the 32-sample optimizer batch is unchanged  # lower it for students with a shorter context (OLMo-2: 4096)

OUT="${EXP}/${TAG}"
RUNS="${OUT}/runs"; BITS="${OUT}/bits"; LOGS="${OUT}/logs"
mkdir -p "${RUNS}" "${BITS}" "${LOGS}"
ln -sfn "$(realpath --relative-to="${OUT}" "${SPLITS}")" "${OUT}/splits"   # analyze.py reads split_stats.json from here

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

CROSS=()
for lam in ${LAMBDAS}; do CROSS+=("lambda${lam}=${SPLITS}/lambda${lam}/held.jsonl"); done

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

for seed in ${SEEDS}; do
  for lam in ${LAMBDAS}; do
    has_stage train || continue
    name="lambda${lam}_seed${seed}"
    stage "train ${name}"   # train_lora.py skips runs whose adapter already exists
    python "${EXP}/train_lora.py" --base "${STUDENT}" --train "${SPLITS}/lambda${lam}/train.jsonl" \
      --output "${RUNS}/${name}" --seed "${seed}" --max_len "${TRAIN_MAX_LEN}" --token_budget "${TRAIN_TOKEN_BUDGET}" > "${LOGS}/train_${name}.log" 2>&1
    if has_stage bits && [[ ! -f "${BITS}/${name}.json" ]]; then
      stage "bits ${name}"
      python "${EXP}/measure_bits.py" --base "${STUDENT}" --run_dir "${RUNS}/${name}" \
        --train "${SPLITS}/lambda${lam}/train.jsonl" --held "${SPLITS}/lambda${lam}/held.jsonl" \
        --cross "${CROSS[@]}" --cache_dir "${OUT}/cache/base_bits" \
        --max_len "${TRAIN_MAX_LEN}" --output "${BITS}/${name}.json" > "${LOGS}/bits_${name}.log" 2>&1
    fi
  done
done

# One vLLM engine per benchmark covers every seed and lambda: starting the engine costs minutes,
# so the adapters are handed over as a single list rather than one process per seed.
ALL_ADAPTERS=()
for seed in ${SEEDS}; do
  for lam in ${LAMBDAS}; do ALL_ADAPTERS+=("lambda${lam}_seed${seed}=${RUNS}/lambda${lam}_seed${seed}/adapter"); done
done

if has_stage eval; then
  stage "eval val (${#ALL_ADAPTERS[@]} adapters)"
  python "${EXP}/eval_student.py" --base "${STUDENT}" --include_base --adapters "${ALL_ADAPTERS[@]}" \
    --data "${SPLITS}/lambda0.0/held.jsonl" --out_dir "${OUT}/eval_val" --n_samples "${EVAL_SAMPLES}" \
    --max_model_len "${MAX_MODEL_LEN}" --max_new_tokens "${MAX_NEW_TOKENS}" > "${LOGS}/eval_val.log" 2>&1
fi

if has_stage eval500; then
  stage "eval math500 (${#ALL_ADAPTERS[@]} adapters)"
  python "${EXP}/eval_student.py" --base "${STUDENT}" --include_base --adapters "${ALL_ADAPTERS[@]}" \
    --data "${EXP}/data/math500.parquet" --out_dir "${OUT}/eval_math500" --n_samples "${EVAL_SAMPLES}" \
    --max_model_len "${MAX_MODEL_LEN}" --max_new_tokens "${MAX_NEW_TOKENS}" > "${LOGS}/eval_math500.log" 2>&1
fi

if has_stage analyze; then
  stage "analyze"
  python "${EXP}/analyze.py" --sweep_dir "${OUT}" --out_dir "${OUT}" > "${LOGS}/analyze.log" 2>&1
fi

OWNER="$(stat -c '%u:%g' .)"
chown -R "${OWNER}" "${OUT}"
stage "done"
