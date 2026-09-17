# 섹션 구성

- 3 Measuring What CoT Distillation Memorizes (방법만, 결과 없음)
   - 3.1 Teacher Spectrum (+ Table 1: teacher 쌍)
   - 3.2 Separating Transfer from Memorization (기호: 밀도 ρ, 암기 비율 m)
   - 3.3 Experimental Setup
- 수치 출처: `experiments/PLAN.md`, `experiments/lambda_lora_bits/{prepare_data,build_splits,train_lora,measure_bits,common}.py`, `pair*/splits/split_stats.json`
- AMC23·AIME24는 MI-Distillation처럼 인용함 (maa2023amc, zhang2024aime24). Llama-3.2는 Herd 보고서와 Meta 블로그를 함께 인용함
- `[TBD]`: 묶음 D의 공통 문제 수

# Table 1 (초안)

| Pair | Reasoning model (λ = 0) | Instruction model (λ = 1) | Template | Shared problems |
|---|---|---|---|---|
| A | DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B-Instruct | reasoning | 2,413 |
| B | Qwen3-4B-Thinking-2507 | Qwen3-4B-Instruct-2507 | reasoning | 2,561 |
| C | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-Math-7B-Instruct | instruction | 1,940 |
| D | DeepSeek-R1-Distill-Llama-8B | Llama-3.1-8B-Instruct | reasoning | [TBD] |

- 캡션: Teacher pairs. Each pair shares a base model and is interpolated at six values of λ. Template is the chat template used by every teacher of the pair, and shared problems are those answered correctly by all six teachers.

# 문장별 초안

## 3.1 Teacher Spectrum

- (보간)
   - To vary the style of teacher rationales, we interpolate the weights of a reasoning model θ_R and an instruction model θ_I derived from the same base, following MI-Distillation (lan2026midistillation), θ(λ) = λθ_I + (1 − λ)θ_R. (1)
   - teacher 풀이의 스타일을 바꾸기 위해, MI-Distillation을 따라 같은 base에서 나온 추론 모델 θ_R과 지시 모델 θ_I의 가중치를 보간함
   - The coefficient λ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1} moves the teacher from long reasoning at λ = 0 to short solutions at λ = 1.
   - 보간 계수 λ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1}은 teacher를 λ = 0의 긴 추론에서 λ = 1의 짧은 풀이로 옮김
- (teacher 쌍)
   - We build four spectra from the pairs in Table 1, which pair R1-distilled reasoning models (guo2025deepseekr1) and Qwen3 thinking models (yang2025qwen3) with instruction models from Qwen2.5-Math (yang2024qwen25math), Qwen3, and Llama-3.1 (grattafiori2024llama3).
   - Table 1의 네 쌍으로 스펙트럼을 만들며, R1 증류 추론 모델과 Qwen3 thinking 모델을 Qwen2.5-Math, Qwen3, Llama-3.1의 지시 모델과 짝지음
   - All teachers of a pair share one chat template, which we choose with a small pilot that keeps long reasoning at λ = 0 and the most shared problems (Appendix A).
   - 한 쌍의 모든 teacher는 같은 chat template을 쓰며, λ = 0에서 긴 추론이 유지되고 공통 문제가 가장 많은 쪽을 작은 파일럿으로 고름
- (생성과 공통 문제)
   - Each teacher answers the same 4,000 problems sampled from the MATH training set (hendrycks2021math) after removing problems that appear in MATH-500 (lightman2023verify), with temperature 0.6, top-p 0.95, and at most 8,192 new tokens.
   - 모든 teacher는 MATH-500과 겹치는 문제를 뺀 MATH 학습 세트에서 뽑은 같은 4,000문제를 temperature 0.6, top-p 0.95, 최대 8,192토큰으로 풂
   - We keep only problems that every teacher in the pair answers correctly without truncation and split them into 300 held-out problems and a training set (Table 1).
   - 한 쌍의 모든 teacher가 잘리지 않고 맞힌 문제만 남겨 held-out 300문제와 학습 세트로 나눔
   - Every condition therefore trains and is measured on the same problems, and conditions differ only in the text of the rationales.
   - 따라서 모든 조건은 같은 문제로 학습하고 측정되며, 조건 사이에는 풀이 텍스트만 다름

## 3.2 Separating Transfer from Memorization

- (부호 길이와 흡수량)
   - Following compression-based memorization measurement (morris2025memorize, tan2026bits), the codelength of a rationale y for problem x under a model θ is H(y | x, θ) = −log₂ p(y | x, θ), summed over the rationale tokens and the end-of-sequence token with the prompt as context.
   - 압축 기반 암기 측정을 따라, 모델 θ에서 문제 x에 대한 풀이 y의 부호 길이는 프롬프트를 조건으로 풀이 토큰과 종료 토큰에 대해 합한 H(y | x, θ) = −log₂ p(y | x, θ)임
   - Let θ_0 be the base student and θ_A the student with a trained LoRA adapter. The bits the adapter writes on a set of rationales D are the codelength it saves relative to the base student, W(D) = Σ_{(x,y)∈D} [H(y | x, θ_0) − H(y | x, θ_A)]. (2)
   - θ_0을 base student, θ_A를 LoRA adapter를 학습한 student라 할 때, 풀이 집합 D에서 adapter가 쓴 bits는 base student 대비 절약한 부호 길이 W(D)임
- (전이와 암기)
   - We measure W on the training rationales D_tr and on the held-out rationales D_ho written by the same teacher, and read W(D_ho) as information that transfers to unseen problems.
   - 같은 teacher가 쓴 학습 풀이 D_tr와 held-out 풀이 D_ho에서 W를 재고, W(D_ho)를 처음 보는 문제로 넘어가는 정보로 봄
   - The memorized bits are the training savings that a token-matched held-out set does not explain, ΔW = W(D_tr) − (T_tr / T_ho) W(D_ho), (3) where T counts rationale tokens.
   - 암기된 bits는 토큰 수를 맞춘 held-out 집합으로 설명되지 않는 학습 절약분 ΔW이며, T는 풀이 토큰 수임
   - Tan et al. (2026) scale by the number of samples, and we scale by tokens because rationale length changes by several times across λ (Appendix B).
   - Tan et al. (2026)은 샘플 수로 환산하지만, 풀이 길이가 λ에 따라 몇 배씩 달라지므로 우리는 토큰 수로 환산함
   - We report the memorized share m = ΔW / W(D_tr), which is near 0 when almost everything the adapter writes transfers, reaches 1 when nothing transfers, and exceeds 1 when training makes held-out rationales harder to predict.
   - 암기 비율 m = ΔW / W(D_tr)을 보고하며, adapter가 쓴 것이 거의 모두 전이되면 0에 가깝고, 전혀 전이되지 않으면 1이 되며, 학습 후 held-out 풀이를 오히려 더 못 예측하면 1을 넘음
- (밀도)
   - Before training, we also compute the information density of the training rationales under the base student, ρ = H(D_tr | θ_0) / T_tr in bits per token.
   - 학습 전에 base student 기준 학습 풀이의 정보 밀도 ρ = H(D_tr | θ_0) / T_tr (bits/token)도 계산함

## 3.3 Experimental Setup

- (student)
   - We distill into six students from three families, Qwen2.5-0.5B, 1.5B, and 3B-Instruct (qwen2025qwen25), Llama-3.2-1B and 3B-Instruct (grattafiori2024llama3, meta2024llama32), and OLMo-2-1B-Instruct (olmo2025olmo2).
   - 세 계열의 student 6개로 증류함 (Qwen2.5-0.5B·1.5B·3B-Instruct, Llama-3.2-1B·3B-Instruct, OLMo-2-1B-Instruct)
- (학습)
   - Each student–teacher pair gets its own LoRA adapter (hu2021lora) of rank 16 and scale 32 on all attention and MLP projections, trained with AdamW at a learning rate of 2×10⁻⁴ under a cosine schedule with 5% warmup, for 3 epochs with batches of 32 rationales and a maximum length of 8,192 tokens (4,096 for OLMo-2).
   - 각 student와 teacher 조합마다 모든 attention·MLP projection에 rank 16, scale 32의 LoRA adapter를 두고, AdamW(learning rate 2×10⁻⁴, 5% warmup 뒤 cosine), 3 epoch, 배치 32, 최대 길이 8,192토큰(OLMo-2는 4,096)으로 학습함
   - Adapters for the same student start from the same initialization, so conditions differ only in their training rationales.
   - 같은 student의 adapter는 같은 초기값에서 시작하므로, 조건 사이에는 학습 풀이만 다름
- (평가)
   - We measure accuracy on the 300 held-out problems and on the benchmarks of MI-Distillation, namely MATH-500, GSM8K (cobbe2021training), AMC23 (maa2023amc), AIME24 (zhang2024aime24), Minerva (lewkowycz2022solving), OlympiadBench (he2024olympiadbench), and GPQA-Diamond (rein2023gpqa), with its decoding settings and grader.
   - 정확도는 held-out 300문제와 MI-Distillation의 벤치마크(MATH-500, GSM8K, AMC23, AIME24, Minerva, OlympiadBench, GPQA-Diamond)에서 같은 디코딩 설정과 채점기로 잼
   - Accuracy is averaged over 16 samples for AIME24 and AMC23 and over 4 samples otherwise, and further details are in Appendix C.
   - 정확도는 AIME24와 AMC23은 16개, 나머지는 4개 샘플의 평균이며, 자세한 설정은 Appendix C에 있음

# 영문

## 3.1 Teacher Spectrum

To vary the style of teacher rationales, we interpolate the weights of a reasoning model θ_R and an instruction model θ_I derived from the same base, following MI-Distillation (lan2026midistillation), θ(λ) = λθ_I + (1 − λ)θ_R. (1) The coefficient λ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1} moves the teacher from long reasoning at λ = 0 to short solutions at λ = 1. We build four spectra from the pairs in Table 1, which pair R1-distilled reasoning models (guo2025deepseekr1) and Qwen3 thinking models (yang2025qwen3) with instruction models from Qwen2.5-Math (yang2024qwen25math), Qwen3, and Llama-3.1 (grattafiori2024llama3). All teachers of a pair share one chat template, which we choose with a small pilot that keeps long reasoning at λ = 0 and the most shared problems (Appendix A).

Each teacher answers the same 4,000 problems sampled from the MATH training set (hendrycks2021math) after removing problems that appear in MATH-500 (lightman2023verify), with temperature 0.6, top-p 0.95, and at most 8,192 new tokens. We keep only problems that every teacher in the pair answers correctly without truncation and split them into 300 held-out problems and a training set (Table 1). Every condition therefore trains and is measured on the same problems, and conditions differ only in the text of the rationales.

## 3.2 Separating Transfer from Memorization

Following compression-based memorization measurement (morris2025memorize, tan2026bits), the codelength of a rationale y for problem x under a model θ is H(y | x, θ) = −log₂ p(y | x, θ), summed over the rationale tokens and the end-of-sequence token with the prompt as context. Let θ_0 be the base student and θ_A the student with a trained LoRA adapter. The bits the adapter writes on a set of rationales D are the codelength it saves relative to the base student, W(D) = Σ_{(x,y)∈D} [H(y | x, θ_0) − H(y | x, θ_A)]. (2) We measure W on the training rationales D_tr and on the held-out rationales D_ho written by the same teacher, and read W(D_ho) as information that transfers to unseen problems. The memorized bits are the training savings that a token-matched held-out set does not explain, ΔW = W(D_tr) − (T_tr / T_ho) W(D_ho), (3) where T counts rationale tokens. Tan et al. (2026) scale by the number of samples, and we scale by tokens because rationale length changes by several times across λ (Appendix B). We report the memorized share m = ΔW / W(D_tr), which is near 0 when almost everything the adapter writes transfers, reaches 1 when nothing transfers, and exceeds 1 when training makes held-out rationales harder to predict. Before training, we also compute the information density of the training rationales under the base student, ρ = H(D_tr | θ_0) / T_tr in bits per token.

## 3.3 Experimental Setup

We distill into six students from three families, Qwen2.5-0.5B, 1.5B, and 3B-Instruct (qwen2025qwen25), Llama-3.2-1B and 3B-Instruct (grattafiori2024llama3, meta2024llama32), and OLMo-2-1B-Instruct (olmo2025olmo2). Each student–teacher pair gets its own LoRA adapter (hu2021lora) of rank 16 and scale 32 on all attention and MLP projections, trained with AdamW at a learning rate of 2×10⁻⁴ under a cosine schedule with 5% warmup, for 3 epochs with batches of 32 rationales and a maximum length of 8,192 tokens (4,096 for OLMo-2). Adapters for the same student start from the same initialization, so conditions differ only in their training rationales. We measure accuracy on the 300 held-out problems and on the benchmarks of MI-Distillation, namely MATH-500, GSM8K (cobbe2021training), AMC23 (maa2023amc), AIME24 (zhang2024aime24), Minerva (lewkowycz2022solving), OlympiadBench (he2024olympiadbench), and GPQA-Diamond (rein2023gpqa), with its decoding settings and grader. Accuracy is averaged over 16 samples for AIME24 and AMC23 and over 4 samples otherwise, and further details are in Appendix C.

# 한문

## 3.1 Teacher Spectrum

teacher 풀이의 스타일을 바꾸기 위해, MI-Distillation을 따라 같은 base에서 나온 추론 모델 θ_R과 지시 모델 θ_I의 가중치를 θ(λ) = λθ_I + (1 − λ)θ_R로 보간한다(식 1). 보간 계수 λ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1}은 teacher를 λ = 0의 긴 추론에서 λ = 1의 짧은 풀이로 옮긴다. Table 1의 네 쌍으로 스펙트럼을 만들며, R1 증류 추론 모델과 Qwen3 thinking 모델을 Qwen2.5-Math, Qwen3, Llama-3.1의 지시 모델과 짝짓는다. 한 쌍의 모든 teacher는 같은 chat template을 쓰며, λ = 0에서 긴 추론이 유지되고 공통 문제가 가장 많은 쪽을 작은 파일럿으로 고른다(Appendix A).

모든 teacher는 MATH-500과 겹치는 문제를 뺀 MATH 학습 세트에서 뽑은 같은 4,000문제를 temperature 0.6, top-p 0.95, 최대 8,192토큰으로 푼다. 한 쌍의 모든 teacher가 잘리지 않고 맞힌 문제만 남겨 held-out 300문제와 학습 세트로 나눈다(Table 1). 따라서 모든 조건은 같은 문제로 학습하고 측정되며, 조건 사이에는 풀이 텍스트만 다르다.

## 3.2 Separating Transfer from Memorization

압축 기반 암기 측정을 따라, 모델 θ에서 문제 x에 대한 풀이 y의 부호 길이는 프롬프트를 조건으로 풀이 토큰과 종료 토큰에 대해 합한 H(y | x, θ) = −log₂ p(y | x, θ)이다. θ_0을 base student, θ_A를 LoRA adapter를 학습한 student라 할 때, 풀이 집합 D에서 adapter가 쓴 bits는 base student 대비 절약한 부호 길이 W(D)이다(식 2). 같은 teacher가 쓴 학습 풀이 D_tr와 held-out 풀이 D_ho에서 W를 재고, W(D_ho)를 처음 보는 문제로 넘어가는 정보로 본다. 암기된 bits는 토큰 수를 맞춘 held-out 집합으로 설명되지 않는 학습 절약분 ΔW이며(식 3), T는 풀이 토큰 수이다. Tan et al. (2026)은 샘플 수로 환산하지만, 풀이 길이가 λ에 따라 몇 배씩 달라지므로 우리는 토큰 수로 환산한다(Appendix B). 암기 비율 m = ΔW / W(D_tr)을 보고하며, adapter가 쓴 것이 거의 모두 전이되면 0에 가깝고, 전혀 전이되지 않으면 1이 되며, 학습 후 held-out 풀이를 오히려 더 못 예측하면 1을 넘는다. 학습 전에 base student 기준 학습 풀이의 정보 밀도 ρ = H(D_tr | θ_0) / T_tr (bits/token)도 계산한다.

## 3.3 Experimental Setup

세 계열의 student 6개로 증류한다(Qwen2.5-0.5B·1.5B·3B-Instruct, Llama-3.2-1B·3B-Instruct, OLMo-2-1B-Instruct). 각 student와 teacher 조합마다 모든 attention·MLP projection에 rank 16, scale 32의 LoRA adapter를 두고, AdamW(learning rate 2×10⁻⁴, 5% warmup 뒤 cosine), 3 epoch, 배치 32, 최대 길이 8,192토큰(OLMo-2는 4,096)으로 학습한다. 같은 student의 adapter는 같은 초기값에서 시작하므로, 조건 사이에는 학습 풀이만 다르다. 정확도는 held-out 300문제와 MI-Distillation의 벤치마크(MATH-500, GSM8K, AMC23, AIME24, Minerva, OlympiadBench, GPQA-Diamond)에서 같은 디코딩 설정과 채점기로 잰다. 정확도는 AIME24와 AMC23은 16개, 나머지는 4개 샘플의 평균이며, 자세한 설정은 Appendix C에 있다.
