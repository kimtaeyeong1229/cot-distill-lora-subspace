# 선행 연구 분류 조사 (2026-09-16 기준)

네 갈래로 나눠 각각 독립 조사했고, 모든 서지는 arXiv abstract 페이지를 직접 열어 확인했다.

## A. 작은 모델로의 CoT 증류 — 방법과 한계

| 논문 | 시기 | 핵심 | 우리와의 차이 |
|---|---|---|---|
| [Kajitsuka, Revisiting the Capacity Gap](https://arxiv.org/abs/2604.08880) | 2026.4 | 대부분의 설정에서 CoT 증류가 **증류 전 baseline보다 나쁨** | teacher를 이산적으로 바꿈, 정답률만 봄 |
| [Li, Small Models Struggle to Learn from Strong Reasoners](https://arxiv.org/abs/2502.12143) | 2025.2 | "Small Model Learnability Gap" — ≤3B는 짧은 CoT가 나음 | 긴/짧은을 데이터 혼합 비율로 다룸 |
| [Luo, Through the Valley](https://arxiv.org/abs/2506.07712) | 2025.6 | "Long CoT Degradation" — 8k 예제만으로 **최대 75% 성능 손실** | 데이터 양을 쓸고, teacher는 고정 |
| [Boizard, Scale or Reason?](https://arxiv.org/abs/2509.22193) | 2025.9 | 한 teacher의 **추론 모드만 토글**해 같은 프롬프트로 두 데이터 생성 | 토글이 이진. 전환 지점을 묻지 않음 |
| [Lyu, When Compression Helps and When It Hurts](https://arxiv.org/abs/2606.21704) | 2026.6 (EMNLP 2026) | 압축 예산에 **sharp non-linear degradation** | 전환이 후처리 예산에 있음, teacher가 아님 |
| [Hu, When the Strongest Teacher Is Not the Best](https://arxiv.org/abs/2605.26872) | 2026.5 | teacher 30개 × student 6개, 강한 teacher가 최선이 아님 | teacher가 무관한 기성 모델들 |
| [Griot, Compress-Distill](https://arxiv.org/abs/2606.05988) | 2026.6 | 트레이스 압축 48-run 격자, **원본이 항상 최고 정확도** | 고정 teacher의 트레이스를 사후 수정 |
| [Chen, Skip-Thinking](https://arxiv.org/abs/2505.18642) | 2025.5 | 긴 근거가 **핵심 추론 토큰의 gradient를 과평활**한다는 기제 제시 | 토큰/gradient 수준, 파라미터 미측정 |
| [Luo, DLCoT](https://arxiv.org/abs/2503.16385) | 2025.3 | 긴 CoT 증류가 **비동계 student에서 크게 저하** | 계보를 이진 교란으로만 관찰 |
| [Saini, ThinkSwitch](https://arxiv.org/abs/2606.01080) | 2026.5 | LoRA 증류 + **구면 가중치 보간** 조합 | 보간 계수를 쓸지 않고 보고도 안 함 |

**이 갈래에서 아무도 안 한 것**: teacher를 **연속적으로 매개변수화해 쓴 사람이 없음**. 전부 teacher를 이산적 객체로 두고 그 하류(트레이스 선별·재작성·gradient 혼합·손실 변경)를 조작함. 따라서 **문턱을 관찰할 위치에 있던 연구가 없음**.

## B. 가중치 보간·병합과 경로 위의 행동 전이

| 논문 | 시기 | 문턱 보고 | 우리와의 차이 |
|---|---|---|---|
| [Wu, Revisiting Model Interpolation](https://arxiv.org/abs/2510.10977) | 2025.10 v1 / 2026.01 v2 (**프리프린트, 미게재**) | **예.** Think #R이 λ∈[0.4,0.6]에서 0→1 (Qwen3-4B), [0.5,0.8] (30B) | **같은 지표(Think #R)에 더해 모델 쌍 2개가 우리 쌍 B·D와 동일** (Qwen3-4B Ins/Thinking-2507, Llama-3.1-8B-Instruct + R1-Distill-Llama-8B). 다만 student·LoRA·증류·가중치 기하는 전혀 다루지 않음 |
| [Lan, The Thinking Spectrum](https://arxiv.org/abs/2509.22034) | 2025.9 | **예.** λ≈0.6~0.7 임계 구간, 0.01 간격 | 정답률·토큰으로만 전이를 추론 |
| [Yang, DAMI](https://arxiv.org/abs/2601.21414) | 2026.1 | **아니오.** "convex, monotonic Pareto frontier" | **정면 반대 주장.** 반드시 대응해야 함 |
| [Huang, RAIN-Merging](https://arxiv.org/abs/2602.22538) | 2026.2 (ICLR 2026 Oral) | 아니오 | 형식 붕괴를 **막는 공학**, 우리는 통과시켜 측정 |
| [Xia, Data-Free Layer-Adaptive Merging](https://arxiv.org/abs/2603.21705) | 2026.3 | 값 없음 | **"출력이 병합 계수에 선형이라는 가정이 체계적으로 위배된다"** — 우리 효과의 유일한 이론적 근거 |
| [Li, Scaling LMC to Billion Parameter Transformers](https://arxiv.org/abs/2606.23607) | 2026.6 | 아니오 | **손실 장벽만** 봄. 평평한 손실 경로가 계단형 모드 전환을 가릴 수 있음 |
| [Shen, Merge-of-Thought Distillation](https://arxiv.org/abs/2509.08814) | 2025.9 | 아니오 | **student 쪽 병합.** 우리 설계의 정확한 거울상 |

**이 갈래에서 아무도 안 한 것**: ~~보간·병합된 체크포인트를 teacher로 써서 학습 데이터를 생성한 논문이 0편~~ — **2026-09-21 정정: MI-Distillation ([2608.29623](https://arxiv.org/abs/2608.29623))이 정확히 그것을 했다.** 공백은 "0편"이 아니라 **"있지만 학습 중 gradient까지만 보고 학습이 끝난 가중치는 열지 않았다"**로 써야 한다. 또 `</think>` 방출률을 order parameter로 잰 것은 Wu 하나뿐이고, 그조차 하류로 전파하지 않음. 그리고 임계 구간 보고값이 [0.4,0.6] / [0.6,0.7] / [0.5,0.8]로 **엇갈리고, DAMI는 전이 자체를 부정**함 — 존재·위치·실재성이 모두 미결. 계열 간 재현도 없음.

## C. LoRA 가중치가 담는 것과 읽는 방법

| 논문 | 시기 | 잰 것 | 우리와의 차이 |
|---|---|---|---|
| [Dieudonné, When One Adapter Speaks for Many](https://arxiv.org/abs/2606.28117) | 2026.6 (ICML 2026 WS) | **다른 과제 adapter 간 부분공간 겹침** | 재사용 게이트로 씀. **우연 수준 기준선 없음** |
| [Paul, Spectral Geometry of LoRA Adapters](https://arxiv.org/abs/2604.08844) | 2026.4 | 스펙트럼 특징 → 목적함수 분류 AUC 1.00, 행동과 ρ=0.72 | 지도학습 분류기. seed는 복제일 뿐 기준선 아님 |
| [Shuttleworth, Illusion of Equivalence](https://arxiv.org/abs/2410.21228) | 2024.10 | 침입 차원 (W vs W+ΔW) | adapter를 **사전학습 가중치**와 비교. seed는 강건성 점검 |
| [Tariq, SAD-LoRA](https://arxiv.org/abs/2607.04306) | 2026.7 (ICML 2026 WS) | colspan(B) 주각을 **학습 손실로** 사용 | 정렬을 **최적화**, 우리는 **측정** |
| [Lion, PoLAR](https://arxiv.org/abs/2506.03133) | 2025.6 | **stable rank가 r=16에서 중앙값 1.30~1.40** | ⚠️ **우리 φ의 분모에 직접 타격** |
| [Zhu, Rank-Efficient LoRA](https://arxiv.org/abs/2609.12123) | 2026.9 | 유효 rank가 **optimizer 산물** (AdamW vs Muon) | ⚠️ optimizer 통제 필요 |
| [Zhang, Rethinking Inter-LoRA Orthogonality](https://arxiv.org/abs/2510.03262) | 2025.10 | 직교성을 강제해도 **의미 분리가 안 됨** | ⚠️ 낮은 겹침 ≠ 기능적 독립 |
| [Zhu, When Do Task Vectors Interfere?](https://arxiv.org/abs/2608.09490) | 2026.8 | 가중치 공간 구조가 **프롬프트 형식에 따라 무너짐** | ⚠️ 외적 타당도 경고 |

**이 갈래에서 아무도 안 한 것**: **다른 데이터로 학습한 adapter들을, 같은 데이터·다른 seed 귀무분포에 대고 비교한 논문이 없음.** 조각은 다 있으나 결합된 적이 없다. 또 데이터 분포를 **연속적으로** 바꾼 사례가 없고(전부 범주적으로 다른 과제), 겹침 지표를 행동과 검증한 사람도 없다.

## D. 증류 추론 모델의 평가와 정답률 비판

| 논문 | 시기 | 핵심 | 우리에게 주는 함의 |
|---|---|---|---|
| [Kaiser, Beyond Accuracy](https://arxiv.org/abs/2602.09805) | 2026.2 | 토큰 효율을 **완료율 × 조건부 정확도 × 길이**로 정확 분해 | **우리 3종 지표와 동일한 분해.** 인용해 프로토콜로 채택할 것 |
| [Goyal, Mind the Cap](https://arxiv.org/abs/2608.04160) | 2026.8 | 출력 한도에 따라 측정 격차가 **최대 57점** 변동 | ⚠️ 문맥 한도가 **통제되지 않은 독립변수** |
| [Guedes de Souza, Budget-Dependent Rankings](https://arxiv.org/abs/2608.12150) | 2026.8 | 예산에 따라 **순위가 뒤집힘**. 잘림을 통제해도 3~19% 비단조 | ⚠️ 비잘림 필터링만으로 예산 교란이 안 사라짐 |
| [Carlon, Shorter Reasoning, Earlier Answers?](https://arxiv.org/abs/2608.03401) | 2026.8 | 짧은 추론의 이점 대부분이 **일찍 끝나서**임 | ⚠️ 완료 응답만 보는 것은 **사후 변수 조건화** |
| [Sumit, Below the Noise Floor](https://arxiv.org/abs/2608.27729) | 2026.8 | 1.5B student KD에서 **seed별 표준편차 2.8~48.7점**, **출력 잘림 붕괴 모드** 문서화 | ⚠️ **정답률도 다중 seed 필수** |
| [Bowyer, Don't Use the CLT](https://arxiv.org/abs/2503.01747) | 2025.3 (ICML 2025 Spotlight) | 수백 미만 벤치마크에서 CLT 오차막대가 심하게 과소평가 | ⚠️ 300제·AMC23·AIME24에 정규근사 금지 |
| [Bulut, It's the Problem, Not the Path](https://arxiv.org/abs/2609.03436) | 2026.9 | 178개 셀 중 **1개만** 진짜 prefix-limited | ⚠️ 잘림을 원인으로 부르기 전 **동일 예산 재시작 대조** 필요 |
| [Qian, Benchmark²](https://arxiv.org/abs/2601.03986) | 2026.1 | 벤치마크 변별력 지표 | ⚠️ 우리 300제는 **저변별 영역**. 파지 점검으로 규정할 것 |
| [Wu, Reasoning or Memorization?](https://arxiv.org/abs/2507.10532) | 2025.7 (AAAI 2026) | Qwen2.5 오염으로 **무작위 보상에서도** 수학 성능 향상 | ⚠️ 학습 측 제거로는 **사전학습 오염을 못 막음** |
| [He, LogitTrace](https://arxiv.org/abs/2509.20909) | 2025.9 | **LoRA 주입 실험**으로 반복 노출이 기억 흔적을 남김을 확인 | ⚠️ n-gram 제거 외에 **행동 점검** 필요 |

**우리 평가 설계의 취약점 네 가지**
1. 비잘림 응답만의 정답률은 **사후 변수 조건화**임. 완료율이 다른 조건끼리는 같은 모집단이 아님
2. **문맥 한도가 통제되지 않은 변수**임. λ가 길이를 바꾸면 pair 효과와 한도 효과가 섞임
3. **단일 seed 정답률은 해석 불가**. 우리 정확한 설정(1.5B student, 추론 KD)에서 seed 표준편차 48.7점 사례가 보고됨
4. 두 시험지가 **반대 방향 문제**를 가짐 — 300제는 저변별, MATH-500은 사전학습 오염 가능

## 우리 위치 (네 갈래 종합)

**확인된 공백**
- teacher를 **연속 매개변수로 쓸고 문턱을 찾은** 연구 없음 (A)
- **보간 모델을 teacher로 써서 데이터를 생성한** 연구 0편 (B)
- **다른 데이터 adapter를 seed 귀무분포에 대고 판정한** 연구 없음 (C)
- teacher 쪽 불연속과 student 파라미터 쪽 불연속을 **짝지은** 연구 없음 (A+B+C)

**반드시 인용할 선행**
- 지표: Hu et al. 2021 §7.2 (Grassmann), Tariq 2026 (colspan(B) 근거)
- teacher 쪽 문턱: Wu 2510.10977 (프리프린트), Klagges 2025, Lan 2025
- 반대 주장: Yang DAMI 2026 (매끄럽다), Li LMC 2026 (손실 장벽 없음)
- 이론적 여지: Xia 2026 (병합 계수 선형성 위배)
- 평가 프로토콜: Kaiser 2026 (3종 분해), Bowyer ICML 2025 (오차막대)
