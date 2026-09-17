# 3. Gradient Signatures of Short and Long CoT Distillation — 문장 분해

출처: arXiv 2608.29623, `acl_latex.tex` (표 캡션 제외, 수식은 텍스트로 풀어 씀)

## 도입 문단 — 질문

- (배경)
   - While recent studies indicate that smaller models benefit more from Short CoT distillation than Long CoT, the optimization behavior underlying this phenomenon remains poorly understood.
- (연구 제안)
   - To bridge this gap, we analyze the gradient signatures induced by SFT across student model sizes.

## 방법 문단 1 — 지표 정의

- (방법 출처)
   - Adopting the spectral analysis from Li et al. (2024), we characterize the gradients of the attention projection matrices (Query, Key, Value, and Output) via Singular Value Decomposition (SVD) (Carlini et al. 2024).
- (정의)
   - For a given gradient matrix G ∈ R^{m×n}, we summarize its magnitude and structural concentration using the Nuclear Norm (‖G‖_*) and Effective Rank (ERank):
      - ‖G‖_* = Σ_{k=1}^{min(m,n)} σ_k  … (1)
      - ERank(G) = exp( −Σ_{k=1}^{min(m,n)} p_k log p_k )  … (2)
   - where σ_k represents the k-th singular value of G, and p_k = σ_k / ‖G‖_* defines the normalized singular-value distribution.

## 방법 문단 2 — 지표 해석

- (해석: nuclear norm)
   - From an optimization perspective, the nuclear norm captures the total spectral energy of the gradients; a higher value signifies a more substantial optimization signal imparted by the distillation process.
- (해석: effective rank)
   - Conversely, the effective rank acts as a continuous measure of dimensionality by computing the Shannon entropy of the gradient spectrum.
   - A lower effective rank indicates that the parameter updates are structurally concentrated along a few dominant principal directions, leading to lower-dimensional optimization trajectories.
- (부록 안내)
   - Detailed layer-wise formulations are provided in Appendix B.

## 3.1 Results and Analysis

### 문단 1 — 설정

- (설정)
   - We generate Short and Long CoT trajectories on MATH-500 using Qwen2.5-32B-Instruct (Yang et al. 2024) and QwQ-32B (Qwen Team 2025), respectively, and analyze gradients from Qwen2.5 students at 1.5B, 3B, and 7B scales.
   - Dataset details are provided in Appendix C, with results reported in Table 1.

### 문단 2 — 결과

- (결과 1: nuclear norm)
   - Across all model scales, Long CoT consistently induces larger nuclear norms than Short CoT, indicating stronger optimization signals during distillation.
- (해석)
   - This suggests that Long CoT supervision carries richer gradient information, but also imposes a more demanding learning target for the student.
- (결과 2: 규모 효과)
   - Notably, the nuclear-norm gap becomes more pronounced as the student scale increases, especially for the 3B and 7B models.
- (해석)
   - This trend indicates that larger students are better able to absorb the additional supervision provided by long reasoning trajectories and translate it into stronger spectral gradient updates.
- (결과 3: effective rank)
   - Effective rank reveals a complementary trend.
   - From the 3B scale onward, Long CoT produces lower effective rank than Short CoT, showing that its gradients become more concentrated along dominant spectral directions.
- (종합)
   - Thus, for sufficiently capable students, Long CoT induces updates that are not only larger in magnitude but also more structurally coherent.

### 문단 3 — 원리 도출

- (원리)
   - This motivates a central principle of our method: Effective Long CoT distillation should balance the information density of teacher reasoning trajectories with the distributional and optimization capacity of the student model.

## 구조 메모

- 짧은 분석 절 하나로 방법의 동기를 만듦: 질문 → 지표 정의 → 지표 해석 → 설정 → 결과 3개 → 원리
- 결과 문장마다 바로 뒤에 "This suggests / This trend indicates" 해석 문장을 붙임
- 절 마지막 문장을 굵게 쓴 원리 한 문장으로 끝내고, 이후 방법 절 전체가 이 원리를 따름
- 우리 논문과의 관계
   - 이 절은 학습 중 그래디언트를 봄. 우리 논문은 학습이 끝난 뒤 adapter에 써진 정보를 봄
   - 원리 문장의 "information density … capacity of the student"를 우리 결과가 bits로 뒷받침하거나 구체화하는 관계로 서술할 수 있음
