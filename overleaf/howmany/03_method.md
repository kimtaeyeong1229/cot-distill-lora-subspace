# 3. Measuring What an Adapter Writes — 문장 분해

출처: arXiv 2607.21351v1, `sections/3_method.tex` (그림 캡션 제외, 수식은 텍스트로 풀어 씀)

## 3.1 Information quantities

### 문단 1 — 정의

- (정의: 모델)
   - Let θ̂ be a tuned model and θ_ref a reference that captures what is predictable without seeing the training set.
- (정의: codelength)
   - The codelength of a sequence x under a model is its negative log-likelihood in bits, H_K(x | θ) = −log₂ p(x | θ), which arithmetic coding realizes up to negligible overhead (Witten et al. 1987).
- (정의: 암기량)
   - Following Morris et al. (2026b), unintended memorization is the reference codelength the tuned model undercuts,
      - mem_U(x) = H_K(x | θ_ref) − min(H_K(x | θ_ref), H_K(x | θ̂))  … (1)
   - summed over a dataset as mem_U = Σ_i mem_U(x_i).

### 문단 2 — 직관

- (직관의 필요성)
   - The idea underneath is worth stating plainly, because everything rests on it.
- (직관)
   - A model that has genuinely memorized a string can predict it, and a good predictor is a good compressor—so the bits the tuned model shaves off the reference are, quite literally, the information it took in.
- (문제: 실제 텍스트에서의 혼동)
   - On ordinary text that figure tangles memorization together with understanding, since a model also compresses what it correctly generalized.
- (해결: 랜덤 문자열)
   - Random strings cut the tangle: there is nothing to understand, so every saved bit is a memorized bit, and a scale calibrated here can be trusted when we later turn it on real data.

### 문단 3 — 랜덤 문자열 설정과 검증

- (방법: 설정)
   - The random-string probe makes the reference exact.
   - Sequences are 64 uniform tokens over a 2,048-symbol vocabulary after a start token, so H_K(x | θ_ref) = 704 bits by construction and nothing in x can be predicted: every bit of mem_U is a bit memorized.
- (검증)
   - Fresh random sequences held out from training must then read mem_U ≈ 0, a check that passes across our whole spine (Appendix A; appendix pointers refer to the supplement).

### 문단 4 — 추가 지표 두 가지

- (정의: 가중치 쪽)
   - Two further quantities describe the artifact and the behavior.
   - The parameter-write cost CL(Δθ) is the codelength of the trainable weights themselves—per-tensor quantized and entropy-coded, minimized over precision—and upper-bounds what shipping the adapter communicates.
- (정의: 행동 쪽)
   - The behavior-write bits
      - W(D) = Σ_{(x, y*) ∈ D} [ H_K(y* | x, θ_base) − H_K(y* | x, θ̂) ]  … (2)
   - measure how far tuning moved the model's likelihoods on canonical answers.
- (성질: SFT와 RL을 비교 가능하게 함)
   - Because W reads only the final policy, it treats a supervised and a reinforcement-learned adapter alike, though only one was trained on likelihood—the property that makes the two comparable at all.
- (정의: 초과 bits)
   - On text we report the signed excess ΔW = W(D_train) − (n_train / n_held)·W(D_held): the training-set bits beyond what a size-matched held-out set explains, the memorization left once ordinary learning is subtracted off.

### 문단 5 — 왜 지표가 세 개인가

- (정당화)
   - Three readings rather than one, because they answer different questions.
- (지표별 역할)
   - The artifact cost CL(Δθ) is what the file physically transmits—the quantity a hub or a federated aggregator would reason about when it decides what a shared checkpoint carries.
   - The behavior bits W and their excess ΔW are what the model's outputs reveal, and because they read a policy rather than its weights, they compare training methods that a weight-space measure could not.
   - And mem_U, on random data, ties both to a ground truth that neither has on real text.
- (해석 방식)
   - An adapter can read high on one and low on another; the gaps between them are as much of the story as the numbers themselves.

## 3.2 Reading capacity cleanly

### 문단 1 — 측정 프로토콜

- (문제 제기)
   - A capacity number is only meaningful once optimization, not storage, has stopped being the bottleneck—and in the adapter regime that is a delicate condition.
- (어려움)
   - Learning rates that are stable for full models diverge here; early-stopping rules that behave at full scale either never fire as the loss flattens or cut off exactly the over-capacity runs that define the plateau.
- (방법: 프로토콜)
   - We therefore fix a 16,000-step budget with no early stopping, sweep the learning rate and report the envelope, and place dataset sizes relative to estimated capacity so the informative band is sampled rather than a budget-starved tail.
- (해석 범위: 하한)
   - The resulting figure is a lower bound at a declared budget: extending to 40,000 steps lifts it by 10–24% (Appendix B), so we read every value as "at least this much."
- (설정: 모델)
   - Bases are GPT-style decoders (2M and 8M non-embedding parameters) pretrained on WikiText-103 (Merity et al. 2017) then frozen; adapters are LoRA (Hu et al. 2022) and weight-tied TinyLoRA (Morris et al. 2026a).
- (설정: 상하한 기준)
   - The untouched base bounds each reading from below, and full fine-tuning of the same architecture—which reaches near-total memorization under the identical budget—bounds it from above.

## 구조 메모

- 정의 → 직관 → 설정 → 추가 지표 → "왜 세 개인가" 순서로, 수식마다 바로 다음 문장에서 의미를 풀어 씀
- "The idea underneath is worth stating plainly"처럼 독자를 위한 직관 문단을 따로 둠
- 지표를 늘어놓는 데서 끝내지 않고, 지표 간 불일치 자체가 결과의 일부라고 미리 말함
- 3.2는 결과가 하한값임을 선언해 리뷰어의 "최적화가 덜 된 것 아니냐"는 반론을 먼저 막음
- ΔW 정의(학습 이득 − 크기를 맞춘 held-out 이득)는 우리 논문의 ΔW/W와 같은 형태임
