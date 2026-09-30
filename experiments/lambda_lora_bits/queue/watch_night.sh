#!/usr/bin/env bash
# Overnight watch (2026-10-01): SAE on the remaining section-5 conditions (S-) and the data-half adapter
# evaluations (H-) on server1/2/5. Every 5 minutes: log done/failed jobs and pull results to server2.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits; Q=$E/queue; R=workspace/cot-distill-lora-bit; LOG=$Q/watch_night.log
HOSTS=("server2|local|A|sweep sweep_llama sweep_olmo2" "server1|leap@100.71.184.57|B|pairB/qwen0.5b pairB/llama1b pairB/olmo1b"
       "server5|rtx3090|C|pairD/qwen0.5b pairD/llama1b pairD/olmo1b")
declare -A seen
while true; do
  all=1
  for spec in "${HOSTS[@]}"; do
    IFS='|' read -r host target cell cells <<< "$spec"
    cmd="grep -E '^[SH]-' $Q/$host.done 2>/dev/null | tr '\n' ' '; echo; grep -E '^[SH]-' $Q/$host.failed 2>/dev/null | tr '\n' ' '"
    if [[ $target == local ]]; then o=$(bash -c "$cmd"); else o=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "cd $R && $cmd" 2>/dev/null); fi
    d=$(sed -n 1p <<< "$o"); f=$(sed -n 2p <<< "$o")
    for id in $d; do [[ -z "${seen[$host:$id]:-}" ]] && { seen[$host:$id]=1; echo "$(date '+%F %T') DONE $host $id" >> $LOG; }; done
    [[ -n "${f// /}" ]] && [[ -z "${seen[$host:fail]:-}" ]] && { seen[$host:fail]=1; echo "$(date '+%F %T') ALERT $host FAILED $f" >> $LOG; }
    if [[ $target != local ]]; then
      rsync -a --timeout=120 "$target:$R/$E/sae_llama/$cell/" "$E/sae_llama/$cell/" 2>/dev/null
      for c in $cells; do rsync -a --mkpath --timeout=120 "$target:$R/$E/$c/eval_halves/" "$E/$c/eval_halves/" 2>/dev/null; done
    fi
    (( $(wc -w <<< "$d") < 4 )) && all=0
  done
  (( all )) && { echo "$(date '+%F %T') ALL-DONE overnight jobs, results pulled to server2" >> $LOG; exit 0; }
  sleep 300
done
