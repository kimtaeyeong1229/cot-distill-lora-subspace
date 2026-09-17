#!/usr/bin/env bash
# Generate a subset of one pair's teacher CoT pool on a second machine, to split a slow pair's
# generation. Uses the same interpolation, template view and generate.py settings as run_pair.sh,
# but never builds splits; the machine running run_pair.sh for the full pair does that.
#
#   PAIR=B TEMPLATE=reasoning LAMBDAS="0.4 0.6 0.8 1.0" PUSH_TO=server1 bash experiments/lambda_lora_bits/gen_pair_lambdas.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP=experiments/lambda_lora_bits
PAIR="${PAIR:?}"; TEMPLATE="${TEMPLATE:?}"; LAMBDAS="${LAMBDAS:?}"
case "${PAIR}" in
  B) REASONING_ID=Qwen/Qwen3-4B-Thinking-2507; INSTRUCT_ID=Qwen/Qwen3-4B-Instruct-2507 ;;
  *) echo "only pair B is split this way"; exit 1 ;;
esac
REASONING="models/${REASONING_ID##*/}"; INSTRUCT="models/${INSTRUCT_ID##*/}"
OUT="${EXP}/pair${PAIR}"
mkdir -p "${OUT}/teachers" "${OUT}/logs" "${OUT}/cot" "${OUT}/view"
stage() { echo "[$(date '+%F %T')] $*"; }

export HF_TOKEN="$(cat hf.key)"
for repo in "${REASONING_ID}" "${INSTRUCT_ID}"; do
  dir="models/${repo##*/}"
  if [[ ! -f "${dir}/.download_done" ]]; then
    stage "download ${repo}"
    python -c "import sys; from huggingface_hub import snapshot_download as s; s(sys.argv[1], local_dir=sys.argv[2], allow_patterns=['*.safetensors','*.json','*.jinja','*.txt','tokenizer*','*.model'])" "${repo}" "${dir}"
    touch "${dir}/.download_done"
  fi
done
unset HF_TOKEN

tok="$([[ "${TEMPLATE}" == reasoning ]] && echo "${REASONING}" || echo "${INSTRUCT}")"
for lam in ${LAMBDAS}; do
  dir="${OUT}/teachers/lambda${lam}"
  if [[ ! -f "${dir}/model.safetensors" && ! -f "${dir}/model.safetensors.index.json" ]]; then
    stage "interpolate lambda=${lam}"
    python "${EXP}/interpolate.py" --reasoning "${REASONING}" --instruct "${INSTRUCT}" --lam "${lam}" \
      --output "${dir}" --tokenizer_from "${TEMPLATE}" | tail -1
  fi
  view="${OUT}/view/lambda${lam}"; mkdir -p "${view}"
  src="$(cd "${dir}" && pwd)"
  for f in "${src}"/model*.safetensors* "${src}/config.json" "${src}/generation_config.json"; do ln -sfn "${f}" "${view}/$(basename "${f}")"; done
  for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json chat_template.jinja; do
    rm -f "${view}/${f}"; [[ -f "${tok}/${f}" ]] && cp "${tok}/${f}" "${view}/"
  done
  stage "generate lambda=${lam} (${TEMPLATE} template)"
  python "${EXP}/generate.py" --model "${view}" --problems "${EXP}/data/pool.parquet" \
    --output "${OUT}/cot/lambda${lam}.jsonl" > "${OUT}/logs/generate_lambda${lam}.server2.log" 2>&1
  grep -E "^\[" "${OUT}/logs/generate_lambda${lam}.server2.log" | tail -1 || true
  n=$(wc -l < "${OUT}/cot/lambda${lam}.jsonl")
  if [[ "${n}" -ge 4000 && -n "${PUSH_TO:-}" ]]; then
    stage "complete (${n} rows); marker for ${PUSH_TO}"
    touch "${OUT}/cot/lambda${lam}.server2.done"
  fi
done
stage "done"
