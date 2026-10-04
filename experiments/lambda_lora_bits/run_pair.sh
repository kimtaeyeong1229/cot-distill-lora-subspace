#!/usr/bin/env bash
# Teacher side of pairs B and C: download -> template pilot -> full CoT pool -> shared splits.
#
# The paper's pair C is PAIR=D here (directory pairD/, key "D" in some result files).
# Same procedure as pair A (run_sweep_v2.sh and pilot_template.sh); only the endpoints change.
# STAGE=pilot stops after the template comparison so the template can be agreed before the
# 4,000-problem pool is generated with STAGE=full TEMPLATE=<reasoning|instruct>. Both pairs use
# TEMPLATE=reasoning (pairB/template.txt, pairD/template.txt).
#
#   docker exec -d cot-distill-lora-bit bash -c "PAIR=B STAGE=pilot bash experiments/lambda_lora_bits/run_pair.sh > experiments/lambda_lora_bits/pairB/pilot.log 2>&1"
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/lambda_lora_bits
PAIR="${PAIR:?set PAIR=B or D (D is pair C in the paper)}"
STAGE="${STAGE:?set STAGE=pilot or STAGE=full}"
case "${PAIR}" in
  B) REASONING_ID=Qwen/Qwen3-4B-Thinking-2507;              INSTRUCT_ID=Qwen/Qwen3-4B-Instruct-2507 ;;
  D) REASONING_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-8B; INSTRUCT_ID=meta-llama/Llama-3.1-8B-Instruct ;;
  *) echo "unknown PAIR=${PAIR}"; exit 1 ;;
esac
REASONING="models/${REASONING_ID##*/}"
INSTRUCT="models/${INSTRUCT_ID##*/}"
STUDENT=models/Qwen2.5-0.5B-Instruct   # length filter tokenizer, as in pair A
LAMBDAS="${LAMBDAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
PILOT_LAMBDAS="${PILOT_LAMBDAS:-0.0 0.4 1.0}"
LIMIT="${LIMIT:-200}"
N_HELD="${N_HELD:-300}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

OUT="${EXP}/pair${PAIR}"
mkdir -p "${OUT}/teachers" "${OUT}/logs" "${OUT}/pilot"
stage() { echo "[$(date '+%F %T')] $*"; }

# --- download ------------------------------------------------------------------
[[ -f hf.key ]] && export HF_TOKEN="$(cat hf.key)"   # optional; otherwise HF_TOKEN or the cached login
for repo in "${REASONING_ID}" "${INSTRUCT_ID}" Qwen/Qwen2.5-0.5B-Instruct; do
  dir="models/${repo##*/}"
  if [[ ! -f "${dir}/.download_done" ]]; then
    stage "download ${repo}"
    python - "${repo}" "${dir}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*", "*.model"])
PY
    touch "${dir}/.download_done"
  fi
done
[[ -f hf.key ]] && unset HF_TOKEN

has_weights() { [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]]; }

# interpolate <lam> <template> : weights once per lambda, tokenizer swapped to the template's endpoint
teacher() {
  local lam="$1" template="$2" dir="${OUT}/teachers/lambda$1"
  if ! has_weights "${dir}"; then
    stage "interpolate lambda=${lam} (${template} template)"
    python "${EXP}/interpolate.py" --reasoning "${REASONING}" --instruct "${INSTRUCT}" \
      --lam "${lam}" --output "${dir}" --tokenizer_from "${template}" | tail -1
  fi
}

# view <lam> <template> : a directory with the shared weights and the template's tokenizer
view() {
  local lam="$1" template="$2" src tok view
  src="$(cd "${OUT}/teachers/lambda${lam}" && pwd)"
  tok="$([[ "${template}" == reasoning ]] && echo "${REASONING}" || echo "${INSTRUCT}")"
  view="${OUT}/pilot/teachers_${template}/lambda${lam}"
  mkdir -p "${view}"
  for f in "${src}"/model*.safetensors* "${src}/config.json" "${src}/generation_config.json"; do
    ln -sfn "${f}" "${view}/$(basename "${f}")"
  done
  for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json chat_template.jinja; do
    rm -f "${view}/${f}"
    [[ -f "${tok}/${f}" ]] && cp "${tok}/${f}" "${view}/"
  done
  echo "${view}"
}

if [[ "${STAGE}" == pilot ]]; then
  for lam in ${PILOT_LAMBDAS}; do
    teacher "${lam}" instruct
    for template in reasoning instruct; do
      v="$(view "${lam}" "${template}")"
      stage "pilot generate lambda=${lam} template=${template}"
      python "${EXP}/generate.py" --model "${v}" --problems "${EXP}/data/pool.parquet" --limit "${LIMIT}" \
        --gpu_memory_utilization "${GPU_UTIL}" --max_num_seqs "${MAX_NUM_SEQS}" \
        --output "${OUT}/pilot/${template}_lambda${lam}.jsonl" > "${OUT}/logs/pilot_${template}_lambda${lam}.log" 2>&1
      grep -E "^\[|stop_token_ids" "${OUT}/logs/pilot_${template}_lambda${lam}.log" | tail -2
    done
  done
  for template in reasoning instruct; do
    stage "pilot report: ${template} template"
    python "${EXP}/build_splits.py" --gen_dir "${OUT}/pilot" --pattern "${template}_lambda{lam}.jsonl" \
      --lambdas "${PILOT_LAMBDAS}" --student "${STUDENT}" --out_dir "${OUT}/pilot/splits_${template}" --report_only \
      2>&1 | grep -v "it/s\]" | tee "${OUT}/pilot/report_${template}.txt"
  done
  stage "pilot done"
  exit 0
fi

TEMPLATE="${TEMPLATE:?set TEMPLATE=reasoning or TEMPLATE=instruct (agreed from the pilot)}"
mkdir -p "${OUT}/cot"
for lam in ${LAMBDAS}; do
  teacher "${lam}" "${TEMPLATE}"
  v="$(view "${lam}" "${TEMPLATE}")"
  stage "generate lambda=${lam} (${TEMPLATE} template)"
  python "${EXP}/generate.py" --model "${v}" --problems "${EXP}/data/pool.parquet" \
    --gpu_memory_utilization "${GPU_UTIL}" --max_num_seqs "${MAX_NUM_SEQS}" \
    --output "${OUT}/cot/lambda${lam}.jsonl" > "${OUT}/logs/generate_lambda${lam}.log" 2>&1
  grep -E "^\[" "${OUT}/logs/generate_lambda${lam}.log" | tail -1 || true
done

if [[ ! -f "${OUT}/splits/split_stats.json" ]]; then
  stage "build splits"
  python "${EXP}/build_splits.py" --gen_dir "${OUT}/cot" --lambdas "${LAMBDAS}" --student "${STUDENT}" \
    --out_dir "${OUT}/splits" --n_held "${N_HELD}" > "${OUT}/logs/build_splits.log" 2>&1
fi
echo "${TEMPLATE}" > "${OUT}/template.txt"
chown -R "$(stat -c '%u:%g' .)" "${OUT}" models 2>/dev/null || true
stage "pair ${PAIR} teachers done"
