#!/usr/bin/env bash
# Sequential job queue for one machine, run inside its container.
#
# Jobs live in queue/<host>.jobs, one per line: "<id><TAB><shell command>". The runner starts the
# first job that is neither done nor failed; a failed job pauses the queue until its id is removed
# from queue/<host>.failed. Append lines to add work; the file is re-read before every job.
#
#   docker exec -d cot-distill-lora-bit bash -c "QUEUE_HOST=server1 bash experiments/lambda_lora_bits/queue/run_queue.sh"
set -uo pipefail
cd "$(dirname "$0")/../../.."
Q=experiments/lambda_lora_bits/queue
HOST="${QUEUE_HOST:?set QUEUE_HOST}"
mkdir -p "${Q}/logs"
touch "${Q}/${HOST}.jobs" "${Q}/${HOST}.done" "${Q}/${HOST}.failed"
echo $$ > "${Q}/${HOST}.pid"
OWNER="$(stat -c '%u:%g' .)"
log() { echo "[$(date '+%F %T')] $*" >> "${Q}/${HOST}.log"; }
log "runner started (pid $$)"

while true; do
  if [[ -s "${Q}/${HOST}.failed" ]]; then sleep 60; continue; fi
  job=$(awk -F'\t' -v done_file="${Q}/${HOST}.done" '
    BEGIN { while ((getline line < done_file) > 0) finished[line] = 1 }
    NF >= 2 && $1 !~ /^#/ && !($1 in finished) { print; exit }' "${Q}/${HOST}.jobs")
  if [[ -z "${job}" ]]; then sleep 60; continue; fi
  id="${job%%$'\t'*}"
  cmd="${job#*$'\t'}"
  log "START ${id}: ${cmd}"
  if bash -c "${cmd}" > "${Q}/logs/${id}.log" 2>&1; then
    echo "${id}" >> "${Q}/${HOST}.done"
    log "DONE ${id}"
  else
    echo "${id}" >> "${Q}/${HOST}.failed"
    log "FAILED ${id} (queue paused)"
  fi
  chown -R "${OWNER}" experiments/lambda_lora_bits models 2>/dev/null || true
done
