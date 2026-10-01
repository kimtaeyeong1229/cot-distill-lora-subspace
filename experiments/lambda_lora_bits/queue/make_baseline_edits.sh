set -uo pipefail
cd /home/lab/workspace/cot-distill-lora-bit
E=experiments/lambda_lora_bits
for spec in sweep:0.4 sweep_llama:0.4 sweep_olmo2:0.4 pairB/qwen0.5b:0.6 pairB/llama1b:0.6 pairB/olmo1b:0.6 pairD/qwen0.5b:0.6 pairD/llama1b:0.6 pairD/olmo1b:0.6; do
  tag=${spec%%:*}; cliff=${spec#*:}
  for seed in 0 1 2; do
    python3 $E/ablate_subspace.py --tag $tag --cliff $cliff --seed $seed --loo --sources 0.0 --out runs_edit_s$seed --variants gs_tau gs_energy || exit 1
    python3 $E/merge_adapters.py --tag $tag --seed $seed || exit 1
  done
done
