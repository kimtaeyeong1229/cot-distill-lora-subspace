# Prior work for the "think-off switch" angle (2026-10-01)

**Candidate claim.** Remove the shared region V from the λ0 adapter's weight update, with no retraining. The result:
- responses 4–5× shorter
- accuracy above the energy-matched removal controls in 9/9 cells
- accuracy above the λ1 adapter (trained on short data) in 5/9 cells, tied in 1, below in 3
- the switch works one way only: grafting V back in turns the format on but responses do not finish

**Method.** Three parallel searches:
1. activation steering
2. weight-space editing
3. efficient-reasoning distillation and baselines

Every paper listed was checked on its own abstract or proceedings page this session. Unverified leads are kept separately at the end.

**Before citing.** Each paper still needs its PDF placed in `materials/references/`, and its venue decided by the reference rules.

## 1. Closest prior work: must cite and discuss

| Paper | Venue | URL | What it does | Relation to ours |
|---|---|---|---|---|
| Gold-Switch: Training-Free Superposition of Slow- and Fast-Thinking LLMs (Lee, Kwon, Hwang) | arXiv 2025 | https://arxiv.org/abs/2510.06750 | Takes the SVD of ΔW = W_reasoning − W_base and removes a low-rank part, training-free, to switch reasoning off. QwQ-32B GSM8K 1,269→285–494 tok; always-on AIME 66.7→16.7%; soft (α) switching failed | **Closest in outcome.** Its directions come from one model-vs-base spectrum, on full 4B–32B models, not distilled LoRA students. Its failed soft switch matches our "switch, not dial" |
| S³: Spectral Null-Space Swap (Ma et al.) | arXiv 2026-09-29 | https://arxiv.org/abs/2609.37976 | Composes Thinking and Non-thinking weights by subspace, training-free; −27% tokens | Trims thinking rather than switching it off. Qwen3-4B/30B |
| ThinkEdit (Sun, Yan, Weng) | EMNLP 2025 | https://arxiv.org/abs/2503.22048 | W_o(I − vvᵀ) on about 4% of heads, training-free; direction from long-vs-short activations | Same operation, opposite goal (lengthens reasoning); reports that shortening hurts |
| CoT-Valve (Ma et al.) | ACL 2025 | https://arxiv.org/abs/2502.09601 | A LoRA direction whose scale sets CoT length; **trained** on long/short pairs | Ours is not trained for length; it appears in ordinary distillation |
| Transcoder Adapters for Reasoning-Model Diffing (Hu et al.) | arXiv 2026 | https://arxiv.org/abs/2602.20904 | Trained proxy of the distillation change; removing the hesitation features shortens responses at equal accuracy | Closest in spirit; needs a trained proxy, whereas we edit the real ΔW |
| The Halt Vector (Jayabahu, Adeleke) | COLM 2026 WS (non-archival) | https://arxiv.org/abs/2608.28859 | An activation "halt" direction trained into a LoRA; about 20–25% less thinking | Direction is trained in; ours is already present |
| Unlocking Efficient Long-to-Short Reasoning with Model Merging (Wu et al.) | arXiv 2025 | https://arxiv.org/abs/2503.20641 | Training-free merges; −55% at 7B, **but all methods break at 1.5B** | Main external baseline; small scale is the open gap |
| MixReasoning (Lu et al.) | arXiv 2025 | https://arxiv.org/abs/2510.06052 | A concise LoRA trained on short answers, switched on or off per token | A trained version of our "short-data adapter" baseline |
| Refusal is mediated by a single direction (Arditi et al.) | arXiv 2024 | https://arxiv.org/abs/2406.11717 | Projects an activation-derived direction out of the weights | Template for "project a direction out of weights" |
| Shared Parameter Subspaces … Emergent Misalignment (Arturi et al.) | arXiv 2025 (WS) | https://arxiv.org/abs/2511.02022 | Principal angles between LoRAs trained on different data: shared subspace vs a random baseline | **Closest to our "consensus region" measurement**; no removal, no reasoning |
| Convergent Linear Representations of Emergent Misalignment (Soligo et al.) | arXiv 2025 | https://arxiv.org/abs/2506.11618 | A direction shared across fine-tunes, ablated in the others | Convergence across fine-tunes, found in activations |
| Unveiling the Latent Directions of Reflection (Chang et al.) | arXiv 2025 | https://arxiv.org/abs/2508.16989 | Suppressing reflection is much easier than inducing it | Our remove/graft asymmetry is a weight-space counterpart, not a new phenomenon |
| Steering the CensorShip (Cyberey, Evans) | COLM 2025 | https://arxiv.org/abs/2504.17130 | A think-suppression vector in R1-Distill activations | A think-off vector already exists in activations |

## 2. Threats reviewers will raise

- **Universal Weight Subspace Hypothesis** (Kaushik et al., arXiv 2025, https://arxiv.org/abs/2512.05117). Many LoRAs share a few directions regardless of task. Our answer: V exists before the threshold and closes after it, and it exceeds the seed null.
- **Safety Subspaces are Not Linearly Distinct** (Ponkshe et al., ICLR 2026, https://arxiv.org/abs/2505.14185). Behaviour subspaces can be tangled with capability. Our answer is the energy-matched controls, and we should say so explicitly.
- **Name clash with "consensus".** Consensus Merging / TALL-masks (Wang et al., ICML 2024) and ACM (Yao et al., arXiv 2025) use the word. Define our term.
- **Keyword-labelled reflection is unstable.** Zhuang et al., arXiv 2026 (https://arxiv.org/abs/2604.02113) finds 93% of keyword-detected boundaries unstable. Our SAE check partly answers this.
- **Shortening usually hurts accuracy** (Sheng et al., NeurIPS 2025; Zhu et al., EMNLP-F 2026). Ours does not, but part of our gain is λ0's truncation rate (52–87%). The fair comparison is against the energy-matched controls.

## 3. Baselines a practical claim would need

- **Free, same student:** NoThinking prefill (arXiv 2504.09858), s1 budget truncation at matched tokens (2501.19393), NoWait (2506.08343).
- **Training-free weights:** Gold-Switch applied to the merged LoRA ΔW (a plain top-k / energy-threshold removal), TA/TIES merging with the base (2503.20641).
- **Training-free activations:** ASC (2507.04742), SEAL (2504.07986).
- **Trained:** answer-only student (NaturalThoughts 2507.01921), λ1 adapter (we already have it), CoT-Valve / MixReasoning.
- **Metrics:** accuracy vs tokens (Pareto at matched budgets), AES (O1-Pruner 2501.12570), truncation rate, reflective-token leakage in the off state (Demystifying Hybrid Thinking 2510.12680), results split by difficulty.

## 4. What survives as novelty

1. **How the region is found.** Agreement across adapters trained on interpolated-teacher data, with no long/short labels, no activations and no keywords. It closes exactly where the teacher stops reasoning.
2. **Where.** Distilled small students (0.5–1B), the scale where training-free merging is reported to fail.
3. **Controls.** Energy-matched removals and a trained short-data adapter. None of the prior work runs both.
4. **Weight-space evidence of one-way switching.** Removal works, grafting turns the style on but breaks completion. This is consistent with Gold-Switch's failed soft switch and with Chang et al.'s asymmetry in activations.

**Not new:** switching reasoning off training-free in weight space (Gold-Switch), projecting a direction out of weights (Arditi, ThinkEdit), a LoRA direction controlling length (CoT-Valve), LoRAs sharing a subspace across data (Arturi).

## Unverified leads (pages not opened)

- ThinkPrune 2504.01296
- DAST 2503.04472
- LightThinker 2502.15589
- AlphaOne 2505.24863
- Task Singular Vectors (CVPR 2025)
- OPLoRA 2510.13003
- PALoRA 2605.24549
- Null-Basis LoRA 2609.25618
- CoT Vectors 2510.00579
- Budget guidance 2506.13752
