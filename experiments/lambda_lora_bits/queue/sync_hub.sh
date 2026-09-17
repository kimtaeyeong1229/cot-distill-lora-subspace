#!/usr/bin/env bash
# Runs on server2's host (this PC): keeps the three machines' experiment trees in step.
#
# - code and queue definitions flow out from here (server2 is the source of truth)
# - teacher splits flow to every machine, so any machine can train any pair's students
# - results (pair CoT, splits, student runs, evals, queue state) flow back here
# Teacher checkpoints and model weights are never copied.
#
#   nohup bash experiments/lambda_lora_bits/queue/sync_hub.sh > experiments/lambda_lora_bits/queue/sync_hub.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.."
EXP=experiments/lambda_lora_bits
REMOTES="server1 rtx3090"
RDIR=workspace/cot-distill-lora-bit
INTERVAL="${INTERVAL:-600}"

while true; do
  echo "[$(date '+%F %T')] sync"
  for host in ${REMOTES}; do
    # results back (no deletes; newer files win)
    for pair in pairB pairC pairD; do
      rsync -a --update --timeout=120 --exclude 'teachers/' --exclude 'pilot/teachers_*' --exclude 'cache/' \
        "${host}:${RDIR}/${EXP}/${pair}/" "${EXP}/${pair}/" 2>/dev/null || true
    done
    for tag in sweep sweep_llama sweep_olmo2 sweep_qwen1.5b sweep_qwen3b sweep_llama3b; do
      ssh -o ConnectTimeout=20 "${host}" "test -d ${RDIR}/${EXP}/${tag}" 2>/dev/null && \
        rsync -a --update --timeout=120 --exclude 'cache/' --exclude 'splits' \
          "${host}:${RDIR}/${EXP}/${tag}/" "${EXP}/${tag}/" 2>/dev/null
    done
    mkdir -p "${EXP}/queue/remote/${host}"
    rsync -a --timeout=60 --include '*.done' --include '*.failed' --include '*.log' --include 'logs/***' --exclude '*' \
      "${host}:${RDIR}/${EXP}/queue/" "${EXP}/queue/remote/${host}/" 2>/dev/null || true

    # code, queue definitions and every pair's splits out
    rsync -a --timeout=120 --include '*.py' --include '*.sh' --include 'queue/' --include 'queue/*.sh' \
      --include 'queue/*.jobs' --exclude '*' "${EXP}/" "${host}:${RDIR}/${EXP}/" 2>/dev/null || true
    rsync -a --timeout=60 experiments/PLAN.md "${host}:${RDIR}/experiments/" 2>/dev/null || true
    for pair in pairB pairC pairD; do
      [[ -f "${EXP}/${pair}/splits/split_stats.json" ]] && \
        rsync -a --update --timeout=120 --mkpath "${EXP}/${pair}/splits/" "${host}:${RDIR}/${EXP}/${pair}/splits/" 2>/dev/null
    done
    [[ -d "${EXP}/sweep/splits" ]] && \
      rsync -a --update --timeout=120 --mkpath "${EXP}/sweep/splits/" "${host}:${RDIR}/${EXP}/sweep/splits/" 2>/dev/null
  done
  sleep "${INTERVAL}"
done
