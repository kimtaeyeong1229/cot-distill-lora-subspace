#!/usr/bin/env bash
# Collect one (student, pair) cell's adapters onto the machine that will evaluate it.
#
# Training was split by seed across machines, so a cell's 18 adapters can live on three hosts.
# Evaluation wants them together: one vLLM engine serves every adapter of the cell, and starting
# the engine costs minutes. This copies adapter directories only — never runs, logs or caches.
#
#   bash gather.sh <target-host|local> <relative TAG path>
#   bash gather.sh local pairB/llama1b
#   bash gather.sh rtx3090 pairD/olmo1b
set -uo pipefail
cd "$(dirname "$0")/../../.."

TARGET="${1:?target host or 'local'}"
TAG="${2:?tag path, e.g. pairB/llama1b or sweep_llama}"
SOURCES=("local" "leap@100.71.184.57" "rtx3090" "leap@100.126.236.97")
REL="experiments/lambda_lora_bits/${TAG}/runs"
REMOTE_ROOT="workspace/cot-distill-lora-bit"

copy_from() {
  local src="$1"
  [[ "${src}" == "${TARGET}" ]] && return 0
  if [[ "${src}" == "local" ]]; then
    rsync -a --ignore-existing --include='*/' --include='adapter/***' --include='adapter_config.json' \
      --exclude='*' "${REL}/" "${TARGET}:${REMOTE_ROOT}/${REL}/" 2>/dev/null
  elif [[ "${TARGET}" == "local" ]]; then
    rsync -a --ignore-existing --include='*/' --include='adapter/***' --include='adapter_config.json' \
      --exclude='*' "${src}:${REMOTE_ROOT}/${REL}/" "${REL}/" 2>/dev/null
  else
    # hop through this machine: the two remotes cannot reach each other directly
    local tmp; tmp="$(mktemp -d)"
    rsync -a --ignore-existing --include='*/' --include='adapter/***' --include='adapter_config.json' \
      --exclude='*' "${src}:${REMOTE_ROOT}/${REL}/" "${tmp}/" 2>/dev/null
    rsync -a --ignore-existing "${tmp}/" "${TARGET}:${REMOTE_ROOT}/${REL}/" 2>/dev/null
    rm -rf "${tmp}"
  fi
}

mkdir -p "${REL}"
for src in "${SOURCES[@]}"; do copy_from "${src}"; done

if [[ "${TARGET}" == "local" ]]; then
  found=$(ls -d "${REL}"/lambda*_seed*/adapter 2>/dev/null | wc -l)
else
  found=$(ssh -o ConnectTimeout=10 "${TARGET}" "ls -d ${REMOTE_ROOT}/${REL}/lambda*_seed*/adapter 2>/dev/null | wc -l")
fi
echo "${TAG} -> ${TARGET}: ${found}/18 adapters"
