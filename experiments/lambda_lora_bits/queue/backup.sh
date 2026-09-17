#!/usr/bin/env bash
# Second independent copy of the irreplaceable data on server5, the machine the user fully owns.
#
# server2 already holds the complete union (verified: 344/344 adapters, 511/511 eval json, 26/26 CoT),
# so this pushes from there rather than from each machine -- a per-host layout would have copied the
# same adapter several times, since the seed-split and half-split scheduling left duplicates around.
#
# Excluded, with the reason each is safe to drop:
#   teachers*/   an element-wise average of two published checkpoints; reproducibility.lock.json pins
#                the exact commit of each, so interpolate.py rebuilds them byte-for-byte (228 GB saved)
#   models/      base HF checkpoints, re-downloadable at the pinned commits
#   cache/, __pycache__/, hf_upload/   derived or staging
#   *.key, .env  secrets are not copied to another machine
set -uo pipefail
cd "$(dirname "$0")/../../.."

DEST="${DEST:-tykim@100.115.125.36}"
STAMP="${STAMP:-$(date +%Y-%m-%d)}"
TARGET="workspace/backup/cot-distill-lora-bit/${STAMP}"

ssh -o ConnectTimeout=15 "${DEST}" "mkdir -p ${TARGET}"
rsync -a --info=stats2 --delete-excluded \
  --exclude='teachers/' --exclude='teachers_r1tok/' --exclude='models/' \
  --exclude='cache/' --exclude='__pycache__/' --exclude='hf_upload/' \
  --exclude='*.key' --exclude='.env' --exclude='*.pyc' \
  ./ "${DEST}:${TARGET}/" 2>&1 | tail -6
echo "[$(date '+%F %T')] 전송 완료"
ssh -o ConnectTimeout=15 "${DEST}" "du -sh ${TARGET}; df -h ~ | tail -1"
