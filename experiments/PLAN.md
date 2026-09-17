# 실험 설계 (2026-09-13 확정)

진행 현황은 `python3 experiments/lambda_lora_bits/status.py`로 결과 파일을 직접 세어 확인한다.
확정된 항목을 바꿀 때는 반드시 먼저 합의한다.

## Teacher 쌍 — MI-Distillation 방식 가중치 보간

λ = 0, 0.2, 0.4, 0.6, 0.8, 1.0 (λ는 짧은 풀이 쪽 가중치)

| 묶음 | λ=0 (긴 추론) | λ=1 (짧은 풀이) | 규모 | 비고 |
|---|---|---|---|---|
| A | DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B-Instruct | 1.5B | |
| B | Qwen3-4B-Thinking-2507 | Qwen3-4B-Instruct-2507 | 4B | |
| C | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-Math-7B-Instruct | 7B | RTX 3090(24GB)에서 bf16 생성 |
| D | DeepSeek-R1-Distill-Llama-8B | Llama-3.1-8B-Instruct | 8B | RTX 3090(24GB)에서 bf16 생성 (라이선스 확인됨) |

모든 teacher는 양자화 없이 bf16으로 생성한다 (MI-Distillation과 동일, 2026-09-13 결정).

Chat template: 보간 모델은 λ마다 같은 template을 써야 하므로, 묶음마다 A와 같은 파일럿으로 정한다 (2026-09-14 결정).
λ ∈ {0, 0.4, 1.0} × template 2종(reasoning 쪽, instruct 쪽) × MATH 풀 200문제로 teacher 정답률·길이·잘림·공통 문제 수를 비교한다.
묶음 A는 파일럿 결과 reasoning(R1) template을 썼다. 묶음 B도 파일럿 결과 reasoning(Qwen3-4B-Thinking) template으로 정했다 (2026-09-14, 세 λ 공통 문제 186 vs 181, λ0.4 teacher 정답률 97% vs 94%).
묶음 C·D는 같은 기준(세 λ 공통 문제 수, 중간 λ teacher 정답률, 비슷하면 A·B와 같은 reasoning template)으로 Claude가 정하고 결과를 보고한다 (2026-09-14 사용자 위임).
묶음 C는 instruct(Qwen2.5-Math-7B-Instruct) template으로 정했다: 세 λ 공통 문제 183 vs 175, λ=1 teacher 정답률 95% vs 90% (reasoning template에서는 짧은 풀이 끝점이 제대로 작동하지 않음).
묶음 D는 reasoning(R1-Distill-Llama-8B) template으로 정했다: instruct template에서는 λ=0이 생각 모드로 들어가지 않아(평균 476토큰) 긴 추론 끝점이 사라지고, reasoning template은 세 λ 공통 문제 172 vs 160이다.
주의: transformers 5.17은 R1-Distill-Llama-8B의 tokenizer_config("LlamaTokenizerFast")를 잘못된 토크나이저로 불러 띄어쓰기가 사라진다. 3090의 모델 사본은 tokenizer_class를 "PreTrainedTokenizerFast"로 고쳐 썼다(원본은 tokenizer_config.orig.json). 묶음 A·B·C 출력은 영향이 없음을 확인했다. 생성 설정은 A와 같다 (최대 8,192토큰, temperature 0.6, top-p 0.95, 풀 4,000문제).

## Student — 모든 묶음 공통, LoRA r=16 (모든 linear 층), lr 2e-4, 3 epoch, 배치 32, seed 0

| student | 규모 | 비고 |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.5B | |
| Llama-3.2-1B-Instruct | 1B | |
| OLMo-2-0425-1B-Instruct | 1B | 문맥 4,096 → 학습·평가 길이 4,096, λ=0 데이터 일부 잘림 (결과에 명시) |
| Qwen2.5-1.5B-Instruct | 1.5B | |
| Qwen2.5-3B-Instruct | 3B | MI-Distillation student |
| Llama-3.2-3B-Instruct | 3B | MI-Distillation student |

## 학습 데이터

MATH train (MATH-500과 겹치는 문제 제거) → 묶음마다 모든 λ teacher가 맞힌 공통 문제 → 학습 / held-out 300 분할.

## 측정

| 측정 | 대상 | 데이터 |
|---|---|---|
| 밀도 H, bits (W_tr, W_ho, ΔW/W), 교차 λ | 모든 λ | MATH 학습·held-out 풀이 |
| held-out 정답률 (최적 λ 선택용) | 기본 + 모든 λ | MATH held-out 300 |

## 벤치마크 — MI-Distillation과 동일

Pass@1, temperature 0.6, top-p 0.95, 최대 8,192토큰 (OLMo는 4,096 문맥 한도), MI-Distillation의 `evaluation/evaluate.py` 사용

| 벤치마크 | 반복 | 평가 대상 조건 |
|---|---|---|
| MATH-500 | 4회 | 기본 + 모든 λ |
| GSM8K | 4회 | 기본 + 모든 λ |
| AMC23 | 16회 | 기본 + 모든 λ |
| AIME24 | 16회 | 기본 + 모든 λ |
| Minerva | 4회 | 기본 + 모든 λ |
| OlympiadBench | 4회 | 기본 + 모든 λ |
| GPQA-Diamond | 4회 | 기본 + 모든 λ |

7개 벤치마크 × 모든 λ는 MI-Distillation 부록 Table 8(고정 λ별 전체 결과)과 같은 구성이다 (2026-09-13 확정).

GPQA-Diamond는 MI-Distillation이 "난이도가 다양한 추론 벤치마크 모음"의 하나로 넣은 과학 추론 벤치마크이며, 논문 스스로 평가 범위를 "수학 추론 중심"이라고 밝힌다.

## 결과 디렉토리 규칙

- 묶음 A: `sweep`(Qwen2.5-0.5B), `sweep_llama`, `sweep_olmo2`, `sweep_qwen1.5b`, `sweep_qwen3b`, `sweep_llama3b`
- 묶음 B·C·D: `pair{B,C,D}/teachers`, `pair{X}/cot`, `pair{X}/splits`, `pair{X}/<student>/`
- 벤치마크 결과: `<student 디렉토리>/eval_<벤치마크>/<조건>.json` (묶음 A의 Qwen2.5-0.5B MATH-500은 `eval_test/`)

## 제외 결정

- SmolLM2-1.7B: 선행 연구에서 쓰이지 않았고 승인되지 않은 교체였음. 결과는 본문에서 제외.
- ARC, BBH 등 MI-Distillation에 없는 비수학 벤치마크: 제외.

## 2026-09-16 변경: 본문 쌍을 A·B·D로 확정 (사용자 승인)

본문 결과는 쌍 A(Qwen2.5 1.5B), B(Qwen3 4B), D(Llama3.1 8B) 세 쌍으로 한다. 셋 다 reasoning template이라
`</think>` 비율을 공통 지표로 쓸 수 있고, 서로 다른 세 계열이라 문턱 현상의 일반성을 주장할 수 있다.

쌍 C(Qwen2.5 7B)는 본문에서 빼고 **대조군**으로만 쓴다. 파일럿에서 instruct template을 골랐기 때문에
`</think>`가 0%인데 반성 어휘는 13.7 → 1.98 → 0.05로 λ0.2~0.4에서 동일하게 급감한다. 즉 "태그가 없어도
문턱은 있다"를 보이는 자연 대조군이다. 이미 학습·bits·held-out이 끝나 있으므로 추가 GPU를 쓰지 않는다.
쌍 C 벤치마크는 돌리지 않는다.

규모와 계열이 완전히 붙어 있으므로(1.5B=Qwen2.5, 4B=Qwen3, 8B=Llama3.1) **규모 효과는 주장하지 않는다.**
쓸 수 있는 것은 "세 계열 모두에서 문턱이 나타난다"와 "문턱 위치는 쌍마다 다르다"까지다.

측정 지표는 `</think>` 비율과 반성 어휘 밀도(wait/hmm/alternatively 등) 두 가지를 함께 보고한다.

## 2026-09-17 변경: λ 격자를 0.2 간격으로 통일 (사용자 승인)

λ = 0, 0.2, 0.4, 0.6, 0.8, 1.0 여섯 개만 쓴다. 쌍 A에만 있던 λ=0.1과 λ=0.3은 분석과 본문에서 **제외**한다.
설계가 0.2 간격이므로 한 쌍에만 더 촘촘한 표본이 있으면 쌍 간 비교가 불공정해지기 때문이다.

결과적으로 teacher의 추론 형식 전환은 점이 아니라 **구간**으로 보고한다.
쌍 A는 (0.2, 0.4), 쌍 B는 (0.4, 0.6), 쌍 D는 (0.4, 0.6)에서 `</think>` 비율이 무너지고,
0이 되는 시점은 각각 0.4 / 0.6 / 0.8이다. adapter 부분공간 겹침도 같은 구간에서 seed 기준선 아래로 내려간다.

λ0.1·0.3의 생성물과 adapter는 디스크에 남겨두되 어떤 표·그림에도 쓰지 않는다.

## 2026-09-18 — seed baseline retracted, data-resample baseline added

The decision line "phi(lambda0, lambda) crosses the seed-noise floor at the teacher's transition lambda"
is withdrawn. phi was computed seed-matched, so both adapters shared one LoRA initialisation, and a
cross-seed recomputation showed that shared initialisation carried most of the pre-threshold height
(lambda 0.2: same-seed 0.310-0.682 vs cross-seed 0.111-0.205). Against that floor every one of the nine
cells already sits below at lambda 0.2, so the crossing lambda no longer distinguishes the pairs.

What survives: the cliff itself, with the initialisation removed (cross-seed phi falls 0.111 -> 0.036 for
pair A, 0.150 -> 0.076 for B, 0.098 -> 0.032 for D), its pair-specific location, and two new results --
a consensus region of 5.4-7.7 of the 16 dimensions that every pre-threshold adapter shares and every
post-threshold one drops (leave-one-out 0.414-0.580 vs 0.050-0.179, chance 0.002-0.010), and the loss of
seed reproducibility after the cliff in exactly the cells that fail to form a second region.

Replacement baseline: 108 half-data runs, 9 cells x 6 lambdas x 2 disjoint halves with the seed fixed at 0
(run_halves.sh). Treatment phi(half at lambda0, same half at lambda) is judged against baseline
phi(halfA at lambda, halfB at lambda), so both sides share the initialisation and the data budget and the
difference is attributable to the teacher alone (analyze_resample.py).

Scripts: analyze_phi.py (seed-matched, kept for the record), analyze_phi_xseed.py (initialisation removed),
analyze_consensus.py --side pre|post (consensus region), analyze_resample.py (the new decision line).
