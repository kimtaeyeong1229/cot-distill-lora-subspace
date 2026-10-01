#!/usr/bin/env bash
# Watch the think-off jobs (2026-10-01): Gold-Switch (G-/GS-), NoThinking + matched budget (N-), merging (M-),
# the seed-only region (SD-) and its energy-matched control (SC-) on server1/2/5. Every 5 minutes: log
# done/failed jobs, pull results into server2's eval_main_s<seed>/, and stop once no queue has a pending job.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits; Q=$E/queue; R=workspace/cot-distill-lora-bit; LOG=$Q/watch_baselines.log
CELLS="sweep sweep_llama sweep_olmo2 pairB/qwen0.5b pairB/llama1b pairB/olmo1b pairD/qwen0.5b pairD/llama1b pairD/olmo1b"
HOSTS=("server2;local" "server1;leap@100.71.184.57" "server5;rtx3090")
P='^(G|GS|M|N|SD|SC)-[ABC]-[a-z]+$'
declare -A seen
while true; do
  all=1
  for spec in "${HOSTS[@]}"; do
    IFS=';' read -r host target <<< "$spec"
    cmd="grep -E '$P' $Q/$host.done | tr '\n' ' '; echo; grep -E '$P' $Q/$host.failed | tr '\n' ' '; echo; awk -F'\t' -v d=$Q/$host.done 'BEGIN{while((getline l<d)>0)f[l]=1} \$1 ~ /$P/ && !(\$1 in f){n++} END{print n+0}' $Q/$host.jobs"
    if [[ $target == local ]]; then o=$(bash -c "$cmd" 2>/dev/null); else o=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "cd $R && $cmd" 2>/dev/null) || { all=0; continue; }; fi
    d=$(sed -n 1p <<< "$o"); f=$(sed -n 2p <<< "$o"); pending=$(sed -n 3p <<< "$o")
    for id in $d; do [[ -z "${seen[$host:$id]:-}" ]] && { seen[$host:$id]=1; echo "$(date '+%F %T') DONE $host $id" >> $LOG; }; done
    [[ -n "${f// /}" ]] && [[ -z "${seen[$host:fail:$f]:-}" ]] && { seen[$host:fail:$f]=1; echo "$(date '+%F %T') ALERT $host FAILED $f" >> $LOG; }
    if [[ $target != local ]]; then
      for c in $CELLS; do for s in 0 1 2; do
        rsync -a --timeout=120 --include='lambda0.0_gs_*' --include='lambda0.0_nothink*' --include='lambda0.0_budget*' \
          --include='lambda0.0_rmseed*' --exclude='*' "$target:$R/$E/$c/eval_main_s$s/" "$E/$c/eval_main_s$s/" 2>/dev/null
      done; done
    fi
    [[ "${pending:-1}" != 0 ]] && all=0
  done
  (( all )) && { echo "$(date '+%F %T') ALL-DONE think-off jobs, results pulled to server2" >> $LOG; exit 0; }
  sleep 300
done
