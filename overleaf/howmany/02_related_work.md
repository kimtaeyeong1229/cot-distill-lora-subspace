# 2. Related Work — 문장 분해

출처: arXiv 2607.21351v1, `sections/2_related_work.tex`

## Measuring memorization.

- (선행 연구: 초기 정의)
   - Early notions of memorization are operational: a string counts as memorized if a prompt elicits it (Carlini et al. 2019, 2023).
- (선행 연구의 한계)
   - Such tests cannot tell memorization from prediction, since a model may emit a string it never saw simply because the string was predictable.
- (핵심 선행 연구)
   - Morris et al. (2026b) cut the knot with compression—memorization is the bits a model saves an arithmetic coder over a reference, measured on data of known information content—and by training on uniformly random strings, where nothing is predictable, turn those saved bits into a capacity meter that reads 3.6 bits per parameter for GPT-family models.
- (관련 측정들)
   - Related instruments measure or bound the same quantity through intrinsic dimension (Arnold 2025), counterfactual influence (Zhang et al. 2023), behavioral probes of fine-tuned models (Mireshghallah et al. 2022), and the memorization capacity of attention (Mahdavi et al. 2024).
- (갭)
   - All of it studies the full model.
- (본 연구의 위치)
   - We carry the compression meter into the frozen-base regime, where the writable object is the adapter and the base supplies structure it never updates—and where, as it turns out, protocol choices that are harmless at full-model scale come to dominate the reading.

## PEFT and privacy.

- (선행 연구)
   - A steady empirical thread reports that low-rank adaptation leaks less than full fine-tuning: under canary extraction in federated settings (Bossy et al. 2026), under extraction and membership inference in centralized ones (Wang and Li 2025), and by a differential-privacy argument that random low-rank projections sketch away gradient information (Malekmohammadi and Farnadi 2024).
   - Attacks have since reached RLVR pipelines (Liu et al. 2025), and differentially private fine-tuning (Yu et al. 2022) remains the standard defense when leakage matters.
- (갭)
   - Each of these establishes that leakage is smaller; none measures how much information the adapter actually holds, and all compare algorithms at unmatched task performance.
- (본 연구의 위치)
   - We measure the stored information itself, on the very checkpoints we then attack.

## Low-bit adaptation and what RLVR learns.

- (선행 연구: 작은 adapter)
   - Adapters have been driven from millions of parameters (Hu et al. 2022) through shared and decomposed forms (Kopiczko et al. 2024; Bałazy et al. 2025) down to thirteen trained with GRPO (Morris et al. 2026a; Shao et al. 2024); LoRA has been shown to learn and forget less than full fine-tuning (Biderman et al. 2024) and its expressive power characterized in theory (Zeng and Lee 2024).
- (선행 연구의 가설)
   - The thirteen-parameter result carries a conjecture: supervised fine-tuning must swallow the whole token stream, while reinforcement learning's usable signal is only the reward—a few bits per prompt—so a tiny update suffices.
- (선행 연구: 논쟁)
   - A parallel debate asks whether RLVR injects new ability or elicits old (Chu et al. 2025).
- (갭)
   - Both are argued from behavior.
- (본 연구의 위치)
   - We answer with the bits themselves, written into the adapter and into the policy's likelihoods, read out at matched accuracy.

## Adapters as shared artifacts.

- (문제 제기)
   - What sharpens the stakes is that adapters have become objects of exchange in their own right—published on hubs, aggregated across federated clients, and passed between teams—and every transfer moves a file whose data content no one measures.
- (두 관점의 대비)
   - The privacy literature treats a fine-tune as a model to be attacked after training; the ecosystem treats an adapter as an asset to be shared.
- (본 연구의 위치)
   - A capacity measured in bits speaks to both readings at once: it is what an attacker is ultimately bounded by, and what a sharer is unknowingly handing over.
- (동기 재확인)
   - That is the quantity we set out to measure, and the reason a number that can be computed before training is worth more here than an attack that can only be run after it.

## 구조 메모

- \paragraph 4개로 나눈 짧은 Related Work임 (약 520단어)
- 문단마다 "선행 연구 → 갭 한 문장 → We …" 3단 구조를 반복함
   - 갭 문장이 매우 짧음: "All of it studies the full model.", "Both are argued from behavior."
   - 마지막 문장은 항상 "We …"로 본 연구의 위치를 밝힘
- 마지막 문단은 선행 연구 나열이 아니라 문제의식(공유되는 artifact)을 다시 세우는 역할임
- 여러 선행 연구를 한 문장에 콜론·세미콜론으로 묶어 분량을 줄임
