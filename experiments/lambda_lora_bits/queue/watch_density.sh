#!/usr/bin/env bash
# Host-side watch over the student density sweep, every 5 minutes, one status line per machine.
# Lines starting with DONE / ALERT / ALL-DONE are the ones worth acting on; server5's results are
# pulled back here on every pass so server2 always holds the full set.
#   nohup bash experiments/lambda_lora_bits/queue/watch_loo.sh >/dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits; Q=$E/queue; R=workspace/cot-distill-lora-bit
S2="sweep sweep_llama sweep_olmo2 pairB/qwen0.5b pairB/olmo1b"
S5="pairD/qwen0.5b pairD/llama1b pairD/olmo1b pairB/llama1b"
LOG=$Q/watch_density.log; N2=5; N5=4
declare -A seen idle
count() { # dir list -> "tag:done/6 ..."
  for c in $1; do printf "%s:%s/6 " "${c/\//-}" "$(ls $E/$c/eval_density/*.json 2>/dev/null | wc -l)"; done; }
report() { # host done_ids failed_ids runner gpu_util gpu_mem total
  local host=$1 done_ids=$2 failed_ids=$3 runner=$4 util=$5 mem=$6 total=$7 n
  for id in $done_ids; do [[ -z "${seen[$host:$id]:-}" ]] && { seen[$host:$id]=1; echo "$(date '+%F %T') DONE $host $id" >> $LOG; }; done
  [[ -n "$failed_ids" ]] && echo "$(date '+%F %T') ALERT $host FAILED $failed_ids" >> $LOG
  n=$(wc -w <<< "$done_ids")
  if (( n < total )); then
    [[ "$runner" != up ]] && echo "$(date '+%F %T') ALERT $host runner $runner with $((total-n)) jobs left" >> $LOG
    if [[ "${util:-0}" -lt 5 && "${mem:-0}" -lt 3000 ]]; then idle[$host]=$(( ${idle[$host]:-0} + 1 )); else idle[$host]=0; fi
    (( ${idle[$host]} >= 2 )) && echo "$(date '+%F %T') ALERT $host GPU idle for 10+ min with $((total-n)) jobs left" >> $LOG
  fi
}
while true; do
  # server2 (this PC)
  g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  r=$(docker exec cot-distill-lora-bit bash -c 'pgrep -f "run_queu[e].sh" >/dev/null && echo up || echo DOWN' 2>/dev/null || echo NO-CONTAINER)
  d=$(grep '^D-' $Q/server2.done 2>/dev/null | tr '\n' ' '); f=$(grep '^D-' $Q/server2.failed 2>/dev/null | tr '\n' ' ')
  cur=$(grep -o 'START D-[^:]*' $Q/server2.log | tail -1 | cut -d' ' -f2)
  echo "$(date '+%F %T') server2 gpu=${g} runner=$r done=$(wc -w <<< "$d")/$N2 now=$cur | $(count "$S2")" >> $LOG
  report server2 "$d" "$f" "$r" "${g%%,*}" "${g##*,}" $N2
  # server5 (rtx3090): status, then pull its results back
  o=$(ssh -o BatchMode=yes -o ConnectTimeout=10 rtx3090 "cd $R && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ';
      docker exec cot-distill-lora-bit bash -c 'pgrep -f \"run_queu[e].sh\" >/dev/null && echo up || echo DOWN';
      grep '^D-' $Q/server5.done | tr '\n' ' '; echo; grep '^D-' $Q/server5.failed | tr '\n' ' '; echo;
      grep -o 'START D-[^:]*' $Q/server5.log | tail -1 | cut -d' ' -f2" 2>/dev/null)
  if [[ -z "$o" ]]; then echo "$(date '+%F %T') ALERT server5 unreachable over ssh" >> $LOG; else
    mapfile -t L <<< "$o"; g5=${L[0]:-}; r5=${L[1]:-}; d5=${L[2]:-}; f5=${L[3]:-}; c5=${L[4]:-}
    for c in $S5; do for k in eval_density; do
      rsync -a --update --mkpath --timeout=120 rtx3090:$R/$E/$c/$k/ $E/$c/$k/ 2>/dev/null; done; done
    echo "$(date '+%F %T') server5 gpu=${g5} runner=$r5 done=$(wc -w <<< "$d5")/$N5 now=$c5 | $(count "$S5")" >> $LOG
    report server5 "$d5" "$f5" "$r5" "${g5%%,*}" "${g5##*,}" $N5
    if (( $(wc -w <<< "$d") >= N2 && $(wc -w <<< "$d5") >= N5 )); then
      echo "$(date '+%F %T') ALL-DONE both machines, results pulled to server2" >> $LOG; exit 0; fi
  fi
  sleep 300
done
