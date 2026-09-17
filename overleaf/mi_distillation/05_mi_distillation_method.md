# 5. MI-Distillation: SeqLSS-Guided CoT Selection from Interpolated Teachers — 문장 분해

출처: arXiv 2608.29623, `acl_latex.tex` (그림 캡션 제외, 수식은 텍스트로 풀어 씀)

## 도입 문단 — 프레임워크 개요

- (방법 제안)
   - Building on the interpolated teacher spectrum, we propose Model-Interpolation Distillation (MI-Distillation), a student-aware framework for small model CoT distillation.
   - MI-Distillation leverages interpolated teachers to construct a fine-grained spectrum of CoT trajectories, rather than relying solely on instruction-oriented or reasoning-oriented endpoints.
- (필요성)
   - Since a fixed interpolation coefficient may not suit students with different capacities or model families, we further introduce Sequential Learnable Surprisal Score (SeqLSS) to select trajectories that balance reasoning informativeness with student learnability.
- (기대 효과)
   - This enables MI-Distillation to provide supervision that is both informative and aligned with the student's learning capacity, improving the efficiency and effectiveness of CoT distillation for small models.

## 5.1 Model Interpolation and Generate Diverse CoTs

### 문단 1 — teacher 집합

- (방법)
   - Following the model interpolation defined in Eq. (3), the first stage instantiates a set of interpolated teachers from a reasoning-oriented model and an instruction-oriented model.
   - Specifically, we vary the interpolation coefficient over Λ = {0.2, 0.4, 0.6, 0.8, 1.0}, yielding a family of teacher models {M_λ} with parameters {Θ_λ}.
- (성질)
   - These teachers exhibit different trade-offs between reasoning depth, response compactness, and instruction-oriented behavior.

### 문단 2 — 후보 생성

- (방법)
   - Given a training set S = {(q_i, a_i)}_{i=1}^N, each interpolated teacher M_λ is prompted to generate a CoT trajectory for question q_i:
      - R_{i,λ} ~ M_λ(· | q_i),  λ ∈ Λ  … (5)
- (성질)
   - The generated trajectories naturally differ in correctness, complexity, and reasoning granularity.
- (정의)
   - We collect them into an Instruct-Reasoning spectrum:
      - D = {(q_i, a_i, λ, R_{i,λ}) | (q_i, a_i) ∈ S, λ ∈ Λ}  … (6)
- (설계 이유)
   - Rather than committing to a single teacher or a fixed Short/Long CoT style, D exposes a structured set of candidate trajectories along the instruct-reasoning spectrum.
   - This design allows MI-Distillation to perform student-aware CoT selection from a richer supervision space, instead of directly distilling from either endpoint teacher.

## 5.2 Accuracy and SeqLSS Selection

### Accuracy Selection.

- (방법과 이유)
   - We first filter candidate trajectories by answer correctness, since erroneous CoTs can introduce misleading supervision and bias the student toward invalid reasoning patterns (Xiang and Tang 2025).
   - For each trajectory R_{i,λ} generated along the Instruct-Reasoning spectrum, we retain it only when its final prediction matches the ground-truth answer a_i.

### SeqLSS Selection. 문단 1 — 문제 설정과 surprisal

- (문제 제기)
   - After correctness filtering, multiple valid rationales may still differ substantially in how useful they are for a specific student model.
- (이상적 조건)
   - An ideal CoT trajectory should be informative enough to provide non-trivial reasoning signals, while remaining aligned with the student's current predictive distribution.
- (방법 제안)
   - To capture this trade-off, we introduce a student-conditioned criterion, Learnable Surprisal Score (LSS), defined at the token level.
- (정의: surprisal)
   - Given an input problem x and a candidate rationale R = (r_1, …, r_T), the surprisal of token r_i under the student model θ_s is
      - S_i = −log p_{θ_s}(r_i | x, r_{<i})  … (7)
   - which quantifies how much new information r_i conveys to the student.

### SeqLSS Selection. 문단 2 — 학습 가능성 항

- (surprisal의 한계)
   - However, high-surprisal tokens are not always beneficial: they may lie far outside the student's learnable region.
- (정의: U_i)
   - We measure this distributional misalignment by the cumulative probability mass that the student assigns to tokens ranked above r_i:
      - U_i = Σ_{v ∈ V, p_{θ_s}(v | x, r_{<i}) > p_{θ_s}(r_i | x, r_{<i})} p_{θ_s}(v | x, r_{<i})  … (8)
- (직관)
   - Intuitively, U_i reflects how much probability mass the student assigns to tokens ranked above the target token; a smaller U_i indicates that r_i lies closer to the student's high-probability region and is therefore easier to learn.

### SeqLSS Selection. 문단 3 — LSS와 SeqLSS

- (정의: LSS)
   - The token-level LSS combines these two quantities:
      - LSS_i = S_i · (1 − U_i)^α  … (9)
   - where S_i rewards informative tokens and (1 − U_i)^α down-weights tokens that fall outside the student's high-probability region.
- (하이퍼파라미터)
   - The exponent α ≥ 0 controls the strength of this learnability penalty, so that LSS_i favors tokens that are simultaneously informative and learnable.
- (단순 평균의 문제)
   - Naively averaging LSS_i over a rationale is unstable, since a few tokens with extreme surprisal may dominate the sequence-level score (Yang et al. 2026).
- (정의: SeqLSS)
   - We therefore aggregate token-level scores using surprisal as a normalization signal, yielding the Sequential Learnable Surprisal Score:
      - SeqLSS(R) = Σ_i S_i (1 − U_i)^α / Σ_i S_i  … (10)
- (직관)
   - Intuitively, SeqLSS(R) estimates the proportion of informative reasoning signal in R that lies within the student's learnable region.
- (선택 규칙)
   - For each problem, we select the correct candidate rationale with the highest SeqLSS, thereby balancing reasoning-information density with student-side distributional alignment for CoT distillation.

### Format and Distillation.

- (데이터 정의)
   - After accuracy and SeqLSS-based selection, the retained trajectories form the final distillation set D_select = {(q_i, R_i)}, where q_i denotes the input question and R_i is the selected response trajectory, including both the reasoning process and the final answer.
- (학습 목적 함수)
   - The student model θ_S is trained on D_select with the standard next-token prediction objective:
      - L(θ_S) = −Σ_{(q_i, R_i) ∈ D_select} log p_{θ_S}(R_i | q_i)  … (11)
- (평가)
   - The resulting student model is then evaluated on downstream reasoning benchmarks.

## 구조 메모

- 개요 → 후보 생성 → 정답 필터 → SeqLSS → 학습 순서로 파이프라인을 그대로 따라감
- 수식마다 앞에 "무엇을 재려는가", 뒤에 "Intuitively, …" 직관 문장을 붙임
- 지표를 한 번에 제시하지 않고 문제를 하나씩 해결하며 쌓아 올림
   - surprisal만으로는 부족함 → U_i 추가 → 단순 평균은 불안정함 → 정규화한 SeqLSS
- 우리 논문과의 관계
   - S_i(student 기준 surprisal)의 평균이 우리 논문의 밀도 H와 같은 양임
   - 이 논문은 surprisal에 학습 가능성 벌점을 곱해 데이터를 고르고, 우리는 surprisal 자체가 암기 비율을 예측함을 보이는 관계로 서술할 수 있음
