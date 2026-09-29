#!/usr/bin/env bash
# Host-side watch over the integrity re-runs, every 5 minutes: base adapters re-evaluated with their
# generations (E-, dropped), the pair-A MATH-500
# evaluation at 7,680 tokens (M-) and section 5 under the main setting (X-).
# Lines starting with DONE / ALERT / ALL-DONE are the ones worth acting on; remote results are pulled
# back to server2 on every pass.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits; Q=$E/queue; R=workspace/cot-distill-lora-bit
LOG=$Q/watch_int.log
# host | ssh target ("local" for this PC) | cells | number of E- and M- jobs it owns
HOSTS=("server2|local|sweep sweep_llama sweep_olmo2|13"
       "server5|rtx3090|pairD/qwen0.5b pairD/llama1b pairD/olmo1b sweep_llama sweep|13"
       "server1|leap@100.71.184.57|pairB/qwen0.5b pairB/llama1b pairB/olmo1b sweep_llama|10")
declare -A seen idle
while read -r _ _ _ h id; do seen[$h:$id]=1; done < <(grep " DONE " $LOG)
status_cmd() { # prints: gpu util,mem / runner / done ids / failed ids / current id
  echo "cd ${1:-.} && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ';
    docker exec cot-distill-lora-bit bash -c 'pgrep -f \"run_queu[e].sh\" >/dev/null && echo up || echo DOWN';
    grep -E '^[EMXF]-' $Q/$2.done 2>/dev/null | tr '\n' ' '; echo; grep -E '^[EMXF]-' $Q/$2.failed 2>/dev/null | tr '\n' ' '; echo;
    grep -o 'START [EMXF]-[^:]*' $Q/$2.log | tail -1 | cut -d' ' -f2; echo END"; }
while true; do
  all=1
  for spec in "${HOSTS[@]}"; do
    IFS='|' read -r host target cells total <<< "$spec"
    if [[ $target == local ]]; then o=$(bash -c "$(status_cmd . $host)" 2>/dev/null)
    else o=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "$(status_cmd $R $host)" 2>/dev/null); fi
    if [[ -z "$o" ]]; then echo "$(date '+%F %T') ALERT $host unreachable" >> $LOG; all=0; continue; fi
    mapfile -t L <<< "$o"; g=${L[0]:-}; r=${L[1]:-}; d=${L[2]:-}; f=${L[3]:-}; c=${L[4]:-}
    [[ $target != local ]] && for cell in $cells; do for k in eval_math500_7680 eval_main_s0 eval_main_s1 eval_main_s2; do
      rsync -a --update --mkpath --timeout=120 "$target:$R/$E/$cell/$k/" "$E/$cell/$k/" 2>/dev/null; done; done
    counts=""; for cell in $cells; do counts+="${cell/\//-}:m$(ls $E/$cell/eval_math500_7680/*.json 2>/dev/null | wc -l),s$(cat $E/$cell/eval_main_s*/*.json 2>/dev/null | grep -c '"name"')/39 "; done
    n=$(wc -w <<< "$d")
    echo "$(date '+%F %T') $host gpu=$g runner=$r done=$n/$total now=$c | $counts" >> $LOG
    for id in $d; do [[ -z "${seen[$host:$id]:-}" ]] && { seen[$host:$id]=1; echo "$(date '+%F %T') DONE $host $id" >> $LOG; }; done
    [[ -n "${f// /}" ]] && echo "$(date '+%F %T') ALERT $host FAILED $f" >> $LOG
    if (( n < total )); then
      all=0
      [[ "$r" != up ]] && echo "$(date '+%F %T') ALERT $host runner $r with $((total-n)) jobs left" >> $LOG
      u=${g%%,*}; m=${g##*,}
      if [[ "${u:-0}" -lt 5 && "${m:-0}" -lt 3000 ]]; then idle[$host]=$(( ${idle[$host]:-0} + 1 )); else idle[$host]=0; fi
      (( ${idle[$host]} >= 3 )) && echo "$(date '+%F %T') ALERT $host GPU idle for 15+ min with $((total-n)) jobs left" >> $LOG
    fi
  done
  (( all )) && { echo "$(date '+%F %T') ALL-DONE all machines, results pulled to server2" >> $LOG; exit 0; }
  sleep 300
done
