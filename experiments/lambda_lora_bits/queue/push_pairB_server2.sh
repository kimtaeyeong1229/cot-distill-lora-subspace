#!/usr/bin/env bash
# Host-side helper on server2: copy each pair-B CoT file that server2 finished to server1,
# so server1's run_pair.sh finds it complete and skips that lambda. Exits when all are pushed.
cd "$(dirname "$0")/../../.."
EXP=experiments/lambda_lora_bits
LAMBDAS="${LAMBDAS:-0.4 0.6 0.8 1.0}"
pending="${LAMBDAS}"
while [[ -n "${pending// /}" ]]; do
  next=""
  for lam in ${pending}; do
    if [[ -f "${EXP}/pairB/cot/lambda${lam}.server2.done" ]]; then
      if rsync -a --timeout=300 "${EXP}/pairB/cot/lambda${lam}.jsonl" "server1:workspace/cot-distill-lora-bit/${EXP}/pairB/cot/" \
         && ssh server1 "wc -l < workspace/cot-distill-lora-bit/${EXP}/pairB/cot/lambda${lam}.jsonl" | grep -q '^4000$'; then
        echo "[$(date '+%F %T')] pushed lambda${lam} to server1"
      else
        next="${next} ${lam}"
      fi
    else
      next="${next} ${lam}"
    fi
  done
  pending="${next}"
  sleep 300
done
echo "[$(date '+%F %T')] all pushed"
