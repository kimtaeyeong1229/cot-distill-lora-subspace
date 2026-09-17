#!/usr/bin/env bash
# Distinct completed work across the fleet. sync_hub replicates adapters between machines, so the
# per-host counts overlap and must be de-duplicated by path rather than summed.
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
REMOTE=("rtx3090" "leap@100.126.236.97" "leap@100.71.184.57")
R=workspace/cot-distill-lora-bit

m500=$( { ls ${E}/pairD/llama1b/eval_math500/*.json 2>/dev/null
  for h in "${REMOTE[@]}"; do ssh -o ConnectTimeout=8 -o BatchMode=yes "$h" \
    "ls ${R}/${E}/pairD/llama1b/eval_math500/*.json 2>/dev/null" 2>/dev/null; done
  } | xargs -n1 basename 2>/dev/null | sort -u | grep -c lambda )

halves=$( { ls -d ${E}/*/runs/lambda*_half*/adapter/adapter_model.safetensors ${E}/*/*/runs/lambda*_half*/adapter/adapter_model.safetensors 2>/dev/null
  for h in "${REMOTE[@]}"; do ssh -o ConnectTimeout=8 -o BatchMode=yes "$h" \
    "ls -d ${R}/${E}/*/runs/lambda*_half*/adapter/adapter_model.safetensors ${R}/${E}/*/*/runs/lambda*_half*/adapter/adapter_model.safetensors 2>/dev/null" 2>/dev/null; done
  } | sed 's|.*lambda_lora_bits/||' | sort -u | wc -l )

echo "m500 ${m500}/18 | 재표집 ${halves}/108"
[[ "${m500}" -ge 18 && "${halves}" -ge 108 ]] && exit 9 || exit 0
