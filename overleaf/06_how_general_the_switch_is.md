<!-- Notion에서 동기화: 2026-10-04 -->
# How General the Switch Is

| Condition | GSM8K: tokens | GSM8K: acc. | GPQA-Diamond: tokens | GPQA-Diamond: acc. |
|---|---|---|---|---|
| λ = 0 | 3,596 | 34.2 | 3,319 | 13.7 |
| Shared removed | 353 | 47.6 | 938 | 20.1 |
| Other removed | 3,268 | 39.3 | 2,687 | 17.6 |
| λ = 1 | – | 48.5 | – | 22.0 |

| MATH-500 | Markers | Tokens | Acc. |
|---|---|---|---|
| λ = 0 | 22.0 | 4,820 | 13.9 |
| Shared removed | 0.6 | 1,075 | 24.7 |
| Two-seed region removed | 1.7 | 1,403 | 21.7 |
| Top removed (τ = 0.6) | 5.2 | 2,585 | 19.1 |
| Length-matched cut | 15.0 | 1,076 | 16.0 |
| Think-skip prompt | 16.2 | 4,570 | 13.9 |
| Other removed (two-seed region) | 18.2 | 4,616 | 16.1 |
| Merged with λ = 1 | 3.5 | 1,429 | 25.5 |
| λ = 1 | 1.6 | 1,218 | 24.3 |
| Untrained | 0.4 | 813 | 26.9 |

**표 1.** 토큰 수, 정답률(%), 1천 단어당 반성 표현의 아홉 칸 평균임. 위 표는 MATH로 학습한 $\lambda=0$ adapter를 GSM8K와 GPQA-Diamond에 적용한 것이고, 아래 표는 MATH-500에서 생각을 끄는 방법들을 비교함. 아래 표의 회색 행(Merged with λ = 1, λ = 1, Untrained)은 짧은 풀이로 학습한 adapter가 필요하거나 adapter를 쓰지 않으며, –는 보고하지 않은 값임. 칸별 값은 부록 D와 E에 있음. 공유 구역을 빼면 MATH 밖에서도 생각이 꺼지고, seed 2개로 찾은 구역을 빼면 추가 학습 없이 생각이 꺼지지만, 그때의 MATH-500 정답률은 학습 전 student 수준이거나 그보다 낮음
## Other Benchmarks
**MATH에서 찾은 구역은 GSM8K와 GPQA-Diamond에서도 생각을 끔**
- **(결과)**
    - MATH로 학습한 adapter에서 구역을 빼면 반성 표현이 더 쉬운 수학 서술형인 GSM8K에서 원래의 0.2–9%로, 수학 밖 과학 객관식인 GPQA-Diamond에서 0.1–10%로 떨어지지만, 다른 곳에서 같은 에너지를 빼면 각각 50–197%와 27–140%가 남음 (표 13, 부록 E)
    - GSM8K에서는 표 1(위)처럼 응답이 5–19배 짧아지고 정답률이 34.2%에서 47.6%로 올라, $\lambda=1$ adapter(48.5%)에 가깝고 아홉 칸 모두에서 다른 곳을 뺀 조건(39.3%)보다 높으며, 잘림은 11–84%에서 2% 이하로 줄어듦 (표 14). B/Llama만 정답률이 그대로임
    - GPQA-Diamond에서는 $\lambda=1$ adapter(22.0%)를 포함한 모든 조건이 찍기 수준(25%)보다 낮아 반성 표현과 길이만 읽으며, OLMo는 문맥이 4,096토큰이라 긴 문제에서 생성 한도가 1,293토큰으로 줄어 1.5–1.6배만 짧아짐(다른 student는 3.8–4.9배)
- **(해석)**
    - 따라서 이 구역은 MATH의 내용이 아니라 생각하며 푸는 것에 걸린 것으로 보임
    - 다만 수학 밖 근거는 정답률 없이 본 GPQA-Diamond 하나뿐이고, MATH 밖에서는 같은 길이로 끊는 대조가 없어 GSM8K 정답률 이득 중 잘림이 풀린 몫이 얼마인지는 말할 수 없음
## Without Interpolated Teachers
**보간 없이 학습한 adapter 두 개가 구역의 절반 넘게를 되찾고, 그 부분을 빼면 생각이 꺼짐**
- **(목적)**
    - 구역이 보간의 산물인지 보기 위해, seed만 다른 $\lambda=0$ adapter들로 만든 구역(seed 2개로 찾은 구역)을 그 구역을 만드는 데 쓰지 않은 adapter에서 빼서 순환을 피함. 이 구역은 공유 구역 차원의 54–60%를 덮음
- **(결과)**
    - 표 1(아래)에서 이 구역을 빼면 반성 표현이 1천 단어당 22번에서 1.7번으로 줄고 응답이 3.4배 짧아지지만, 다른 곳에서 같은 에너지를 빼면 18.2번이 남고 길이도 원래의 79–113%임 (표 15)
    - Gold-Switch ([Lee et al., 2025](https://arxiv.org/abs/2510.06750))는 5.2번, 생각을 끝냈다는 문장 ([Ma et al., 2025b](https://arxiv.org/abs/2504.09858))은 16.2번, 같은 길이로 끊기 ([Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393))는 15.0번이 남고, 짧은 풀이 adapter와 병합하기 ([Wu et al., 2025](https://arxiv.org/abs/2503.20641))는 3.5번이지만 짧은 풀이로 학습한 adapter가 필요함
    - 정답률은 13.9%에서 21.7%로 올라 추가 학습 없는 방법 중 가장 높지만, 학습 전 student(Qwen 35.1%, Llama 23.3%, OLMo 22.2%)에 가깝거나 그보다 낮은 정도임. $\lambda=0$ adapter가 학습 전 student보다 낮은 것은 주로 응답의 51–87%가 생성 한도에 걸리기 때문이므로 ([Li et al., 2025](https://arxiv.org/abs/2502.12143); [Luo et al., 2025](https://arxiv.org/abs/2506.07712)), 이 결과는 더 좋은 모델을 만드는 방법이 아님
- **(해석)**
    - 따라서 이 구역은 보간 없는 추론 증류에서도 생기지만, seed 둘로는 절반 남짓만 잡히고 그 구역을 빼는 효과도 원래 구역을 뺄 때(반성 표현 0.07–1.4번, 정답률 24.7%)보다 약함
