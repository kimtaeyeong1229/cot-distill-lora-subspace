#!/usr/bin/env bash
# Collect the seed-1/2 data-half adapters (HB- jobs, 2026-10-02) from server1, server3 and server5 into server2
# every 10 minutes; stop once all 216 (9 cells x 6 lambdas x 2 halves x 2 seeds) are here.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits; R=workspace/cot-distill-lora-bit; LOG=$E/queue/watch_halves.log
CELLS="sweep sweep_llama sweep_olmo2 pairB/qwen0.5b pairB/llama1b pairB/olmo1b pairD/qwen0.5b pairD/llama1b pairD/olmo1b"
while true; do
  for target in leap@100.71.184.57 leap@100.126.236.97 rtx3090; do
    for c in $CELLS; do
      rsync -a --timeout=120 --include='*_half[AB]_seed[12]/***' --exclude='*' \
        "$target:$R/$E/$c/runs/" "$E/$c/runs/" 2>/dev/null
    done
  done
  n=$(ls -d $E/*/runs/*_seed[12]/adapter/adapter_model.safetensors $E/*/*/runs/*_seed[12]/adapter/adapter_model.safetensors 2>/dev/null | wc -l)
  echo "$(date '+%F %T') collected $n/216" >> $LOG
  (( n >= 216 )) && { echo "$(date '+%F %T') ALL-DONE halves" >> $LOG; exit 0; }
  sleep 600
done
