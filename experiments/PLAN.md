# 실험 설계 (2026-09-13 확정)

진행 현황은 `python3 experiments/lambda_lora_bits/status.py`로 결과 파일을 직접 세어 확인한다.
확정된 항목을 바꿀 때는 반드시 먼저 합의한다.

## Teacher 쌍 — MI-Distillation 방식 가중치 보간

λ = 0, 0.2, 0.4, 0.6, 0.8, 1.0 (λ는 짧은 풀이 쪽 가중치)

| 묶음 | λ=0 (긴 추론) | λ=1 (짧은 풀이) | 규모 | 비고 |
|---|---|---|---|---|
| A | DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B-Instruct | 1.5B | |
| B | Qwen3-4B-Thinking-2507 | Qwen3-4B-Instruct-2507 | 4B | |
| C | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-Math-7B-Instruct | 7B | 16GB라 FP8로 생성 |
| D | DeepSeek-R1-Distill-Llama-8B | Llama-3.1-8B-Instruct | 8B | 16GB라 FP8로 생성, Llama-3.1 라이선스 필요 |

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
