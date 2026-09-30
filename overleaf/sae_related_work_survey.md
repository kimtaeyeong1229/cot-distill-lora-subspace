# SAE 관련 선행연구 조사 (2026-09-30)

지도교수 피드백(SAE로 추론 형식을 재고, SAE editing으로 증류 성능을 올려 볼 것)에 대한 선행연구 조사.
arXiv 검색 + arXiv API로 **제목·저자·게재일·comment를 전부 대조**한 44편. venue는 arXiv comment/journal_ref에 적힌 것만 적음.
**PDF는 아직 받지 않았으므로, 논문에 인용하기 전에 CLAUDE.md 규칙대로 PDF를 받아 1쪽을 대조해야 함.**
"전수"라고 하기에는 한계가 있음: 웹 검색 기반이라 OpenReview 전용 논문, 블로그, 비영어권 문헌은 빠졌을 수 있음.

---

## 0. 한 줄 결론

- **SAE로 추론 모델의 반성·백트래킹 feature를 찾고 steering하는 연구는 이미 많음** (A). 우리 teacher인 R1-Distill-Qwen-1.5B, R1-Distill-Llama-8B도 이미 대상이 됨.
- **SAE/crosscoder로 "증류·미세조정이 무엇을 바꾸는가"를 보는 연구도 있음** (B). 가장 가까운 것은 Baek & Tegmark (2025)와 OPD crosscoder (2026-09).
- **SAE로 추론을 다른 모델에 옮기는 방법은 이미 있음 (C, Resa의 SAE-Tuning, 1.5B).** 다만 teacher 풀이로 학습하는 CoT 증류에 SAE를 쓰거나, teacher 스타일에 따라 무엇이 옮겨지는지 본 연구는 찾지 못함. 교수님 3번 제안은 Resa와의 차별화가 관건임.
- **반론이 강함** (E): SAE의 "추론 feature"가 사실은 "wait" 같은 단서 토큰에 반응하는 경우가 45–90%라는 ICML 2026 논문이 있음. SAE로 바꾼다고 키워드 지표의 약점이 자동으로 해결되지는 않음.

---

## A. SAE로 추론 모델의 추론 feature를 찾고 조절한 연구

| arXiv | 논문 | 저자 | venue (확인된 것) | 내용 | 우리와의 관계 |
|---|---|---|---|---|---|
| 2503.18878 | I Have Covered All the Bases Here: Interpreting Reasoning Features in LLMs via SAEs | Galichin, Dontsov, Druzhinina 외 (7) | (AAAI 페이지 검색됨, 미확인) | R1 계열에서 불확실성·탐색·반성 feature를 찾고, 키우면 추론 벤치 성능과 풀이 길이가 함께 늘어남. 추론 fine-tuning 뒤에만 생기는 feature 확인 | 반성 feature 정의의 1차 참고. "키우면 길이도 는다"는 우리 5.3 증폭과 대비됨 |
| 2601.03595 | Controllable LLM Reasoning via SAE-Based Steering (SAE-Steering) | Fang, Wang, Xue 외 (7) | ACL 2026 Main | **R1-Distill-Llama-8B**, Qwen3-8B 마지막 층 SAE. 이해·계획·백트래킹·다관점 검증·가설 추론 5전략을 feature로 제어, 오답 경로를 바로잡아 정확도 +7%p | 우리 쌍 C teacher와 동일 모델. 반성 feature를 teacher 쪽에서 정의할 때 바로 쓸 수 있음 |
| 2510.07364 | Base Models Know How to Reason, Thinking Models Learn When | Venhoff, Arcuschin, Torr 외 (5) | ICML (Spotlight) | 문장 단위 소형 SAE로 추론 행동 분류 체계를 만들고, base 모델에 그 방향을 "적절한 때" 켜서 thinking 모델과의 격차 대부분을 회복. SFT-증류 모델 포함 9쌍 | "추론 능력은 이미 있고, 학습은 언제 켤지를 배운다"는 주장. 우리 "스위치" 해석과 직접 연결됨 |
| 2506.18167 | Understanding Reasoning in Thinking LMs via Steering Vectors | Venhoff, Arcuschin, Torr 외 (5) | 워크숍 (Reasoning and Planning for LLMs) | 백트래킹·불확실성 표현 등이 활성 공간의 선형 방향으로 조절됨 (R1-Distill) | SAE 없이 방향만으로도 가능하다는 비교 기준 |
| 2510.04128 | Internal states before wait modulate reasoning patterns | Troitskii, Pal, Wendler 외 (5) | EMNLP 2025 Findings | **R1-Distill-Llama-8B**와 base에 crosscoder. "wait" 직전 상태에서 wait 확률을 올리고 내리는 feature를 찾고, 재시작·재확인 등 추론 패턴을 인과적으로 바꿈 | 우리 지표("wait" 빈도)가 내부에서 무엇에 대응하는지 보여 주는 가장 직접적인 근거 후보 |
| 2512.23988 | Fantastic Reasoning Behaviors and Where to Find Them (RISE) | Zhang, Zhang, Lambert 외 (9) | – | 단계 단위 활성에 SAE를 학습해 반성·백트래킹 feature를 비지도로 찾고, 개입으로 키우거나 줄임 | 단계 단위 SAE는 토큰 단위 키워드 문제를 줄이는 방법 |
| 2603.03031 | Step-Level Sparse Autoencoder for Reasoning Process Interpretation (SSAE) | Yang, Liu, Lai 외 (6) | – | 추론 단계 단위 SAE. R1-Distill-Qwen-32B AIME 개선 보고 | 위와 같은 계열 |
| 2608.08168 | Thinking vs. NoThinking: Interpreting Reasoning Mechanisms via SAEs | Cheng, Lu, Chang 외 (4) | – | R1-Distill-Qwen-7B. Thinking 모드는 희소하고 강한 feature, NoThinking 모드는 분산된 패턴. 상위 3개 feature 억제 시 인과 효과 | 우리 "teacher가 풀이를 버림(λ)" 축의 활성 쪽 대응물 |
| 2505.15634 | Feature Extraction and Steering for Enhanced CoT Reasoning | Li, Wang, Yang 외 (6) | (EMNLP 2025 Main 검색됨, 미확인) | 일반 CoT에서 SAE feature를 뽑아 R1식 깊은 사고를 유도. SAE가 없는 모델용 SAE-free steering도 제안 | 우리 student(SAE 없음)에 적용 가능한 우회로 |
| 2601.08058 | Triggering Chain-of-Thought via Latent Feature Interventions | He, Xiong, Liu 외 (5) | EMNLP 2026 Main | 잠재 feature 개입으로 CoT를 유발 | β=2 이식("형식을 켬")과 같은 방향의 활성 공간 버전 |
| 2510.01528 | Towards Interpretable and Inference-Optimal CoT with SAE-Guided Generation | Zhao, Shankarampeta, Hu 외 (5) | – | SAE 안내로 CoT 생성을 효율화 | 길이·효율 제어 |
| 2506.09967 | Resa: Transparent Reasoning Models via SAEs | Wang, Asilis, Akgül 외 (7) | – | SAE-Tuning: source 모델에서 추론 능력을 담도록 SAE를 학습한 뒤, 그 SAE로 target 1.5B 모델의 SFT를 안내함. 풀이 없이 정답만 있는 데이터로 RL 모델 추론 성능의 97% 이상을 약 2000배 싼 비용으로 재현 | **교수님 3번 제안의 직접 선행.** C에도 다시 적음 |
| 2602.10437 | Control Reinforcement Learning: Token-Level Steering via SAE Features | Cho, Wu, Koshiyama (3) | – | RL로 토큰 단위 SAE feature steering 정책 학습 | SAE editing을 학습에 넣는 예 |
| – | Under the Hood of a Reasoning Model (Goodfire 블로그) | Goodfire | 블로그 (2025-04) | DeepSeek-R1 671B에 SAE를 학습해 공개 | 참고용, 학술 인용 대상 아님 |

## B. SAE/crosscoder로 미세조정·증류가 무엇을 바꾸는지 본 연구 (model diffing)

| arXiv | 논문 | 저자 | venue | 내용 | 우리와의 관계 |
|---|---|---|---|---|---|
| 2503.03730 | Towards Understanding Distilled Reasoning Models: A Representational Approach | **Baek, Tegmark** (2) | ICLR 2025 Workshop (BuildingTrust) | Qwen 계열 base와 R1-Distill 변형에 crosscoder. 증류 모델에만 있는 반성·검산 feature 방향을 찾고, 그 방향으로 과잉 사고/간결 사고를 조절 | **가장 가까운 선행.** 차이: 그들은 활성·teacher 하나·"증류 전후"; 우리는 가중치(LoRA)·teacher를 λ로 연속 이동·전환 시점과의 정렬·가중치 공간 제거/이식 |
| 2609.35210 | Understanding On-Policy Distillation via Sparse Crosscoders | Yu, Ye, Wang 외 (4) | – (2026-09-28) | OPD는 feature를 새로 만들지도 teacher 것을 넘기지도 않고, 공유 feature의 사용만 재조정함. swap readout 제안 | OPD와 SFT(우리)의 대비. 우리 결과(긴 CoT SFT에서 전용 구역이 생김)와 합치면 "SFT는 새 구역을, OPD는 재가중을" 이라는 그림 |
| 2604.25011 | Why Does RL Generalize? A Feature-Level Mechanistic Study of Post-Training | **Shi**, Han, Ostermann 외 (6) | ACL 2026 Main | 같은 데이터·같은 base에서 SFT는 특화 feature를 빨리 많이 만들고, RL은 기존 feature를 보존하며 천천히 바꿈 | 우리 "SFT 증류가 5–8차원 공통 구역을 만든다"와 일치하는 활성 쪽 근거 |
| 2504.02922 | Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning | Minder, Dumas, Juang 외 (5) | NeurIPS (39th) | crosscoder model diffing의 인공물(latent scaling 등) 보정 | crosscoder를 쓸 경우 필수 방법론 |
| 2603.04426 | Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning | Kassem, Jiralerspong, Rostamzadeh 외 (4) | – | 좁은 미세조정에서 변한 방향을 잘 잡는 crosscoder | 같은 문제의 개선판 |
| 2602.11729 | Cross-Architecture Model Diffing with Crosscoders | Jiralerspong, Bricken (2) | – | 서로 다른 구조 모델 간 diffing | teacher(1.5–8B)와 student(0.5–1B)가 구조가 다른 우리 설정에 해당 |
| 2410.12391 | Tracking Universal Features Through Fine-Tuning and Model Merging | Horn, Elliott (2) | – | 1층 소형 Transformer를 두 도메인으로 미세조정한 뒤 slerp로 병합하며 feature를 추적 | **보간 teacher의 feature 변화를 SAE로 볼 때의 방법론 선행** |
| 2605.11426 | A Mechanistic Investigation of Supervised Fine Tuning | Chopra (1) | – | SFT 동안 SAE latent 수준의 표현 이동 측정 | 학습 동역학 |
| 2605.28896 | Feature Geometry of LoRA Adapters: An SAE Analysis | Prasanth K K (1) | – | LoRA 기여만 떼어 낸 activation에 SAE 학습. **LoRA가 만든 feature는 사전학습 SAE feature와 정렬이 약함** | 우리 LoRA 공유 구역을 기성 SAE feature로 설명하려 할 때의 경고 |
| 2606.00995 | Subliminal Learning Is Steering Vector Distillation | Blank, Bhatia, Rajamanoharan 외 (5) | – | teacher 출력으로 학습한 student가 teacher의 steering vector와 정렬된 방향 하나를 배움 | **"증류가 방향 하나를 옮긴다"는 활성 쪽 결과.** 우리 가중치 쪽 공유 구역과 짝을 이룸 |
| 2606.11270 | Quantifying Subliminal Behavioral Transfer Ratios in LM Distillation | König, Kazmi, Li 외 (4) | – | 위 현상의 전이 비율 정량화 | 참고 |
| 2507.15977 | On the transferability of SAEs for interpreting compressed models | Gupte, Chhabra, Khalili (3) | – | 원 모델 SAE를 압축 모델 해석에 재사용 가능한지 | teacher SAE를 student에 쓸 수 있는지의 근거 |
| 2410.06981 | Quantifying Feature Space Universality Across LLMs via SAEs | Lan, Torr, Meek 외 (6) | – | 모델 간 SAE feature 공간의 보편성 | teacher·student feature 대응의 근거 |
| 2603.02908 | SAE as a Crystal Ball: Features Predict Cross-domain Transferability | Zhang, Wang, Wang 외 (7) | – | SAE 표현 이동으로 SFT 전이성을 학습 전에 예측 | SAE를 "예측 도구"로 쓴 예 |

## C. SAE를 학습·데이터에 쓴 연구 (방법 쪽; 교수님 제안 3번과 관련)

| arXiv | 논문 | 저자 | venue | 내용 | CoT 증류에 적용? |
|---|---|---|---|---|---|
| 2507.14894 | SASFT: SAE-guided SFT to Mitigate Unexpected Code-Switching | Deng, Wan, Yang 외 (6) | ICLR 2026 | 특정 언어 feature의 pre-activation을 SFT 중에 적정 수준으로 묶는 손실 | 아니요. **"SAE feature를 SFT 손실에 거는" 가장 직접적인 선례** |
| 2506.09967 | Resa (SAE-Tuning) | Wang, Asilis, Akgül 외 (7) | – | source 모델의 SAE로 target 모델 SFT를 안내해 추론을 이끌어 냄 (1.5B, 풀이 없는 QA 데이터) | **예. "SAE로 증류"의 가장 가까운 선례.** 차이: teacher 풀이를 쓰는 CoT 증류가 아니고, teacher 스타일(λ) 변화나 student 가중치 분석이 없음 |
| 2605.25525 | SAE-FD: SAE Feature Distillation for Continual Learning | Zhang, Li, Li 외 (6) | – | 사전학습 SAE의 희소 feature 공간에서 표현을 고정해 망각 방지 | 아니요 (지속 학습) |
| 2512.23260 | SAILS: Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace | Wang, Ma, Shang 외 (8) | – | SAE 디코더 방향으로 안전 부분공간을 만들어 **LoRA를 초기화** | 아니요 (안전 정렬). **우리 LoRA 공유 구역과 가장 비슷한 구조의 방법** |
| 2505.24428 | Model Unlearning via SAE Subspace Guided Projections | Wang, Li, Wang 외 (5) | – | SAE로 관련·무관 부분공간을 만들어 가중치 갱신을 사영 | 아니요 (unlearning). 우리 "제거"의 SAE 버전 |
| 2605.27354 | SaeRL: Guiding Post-training Data Engineering with SAE Internals | Jing, Dai, Hu 외 (7) | – | SAE 활성으로 다양성·난이도·품질을 모델링해 RLVR 배치·커리큘럼·필터링 | 아니요 (RL 데이터) |
| 2605.28247 | IRDS: Interpretable RLVR Data Selection via SAE Coverage | Li, Zhang, Shen 외 (4) | – | SAE 클러스터 커버리지로 RLVR 데이터 선택 | 아니요 |
| 2502.14133 | Self-Regularization with SAEs for Controllable LLM Classification | Wu, Yu, Zhai 외 (4) | SIGKDD 2025 | 원치 않는 feature를 SAE로 찾아 정규화 | 아니요 (분류) |
| 2605.15961 | SAEs enable Robust and Interpretable Fine-tuning of CLIP | Morelli, Uselis, Sonthalia 외 (4) | – | feature 추가·제거에 페널티를 주는 미세조정 | 아니요 (비전) |
| 2509.23799 | Enhancing LLM Steering through SAE-Based Vector Refinement | Wang, Wu, Shu 외 (5) | – | steering 벡터를 SAE로 정제 | editing 기법 |
| 2510.01246 | Comparative Analysis of SAE and Activation Difference in Steering | Xie (1) | – | SAE steering과 평균 차이 steering 비교 | editing 기법 비교 |
| 2608.25354 | Multi-Task Model Merging via High-Dimensional Sparse Disentanglement | Zhang, Sun, Wen 외 (4) | – | task vector를 SAE 공간에 사영해 병합 간섭 감소 | 보간·병합을 SAE 공간에서 하는 예 |
| 2501.19406 | Low-Rank Adapting Models for SAEs | Chen, Engels, Tegmark (3) | – | SAE를 끼운 모델을 LoRA로 적응 | 도구 |
| (ACL 2026 long 1685) | LangSAE Editing (교수님 추천) | Kim, Yoon, Park, Lim | ACL 2026 Long | 임베딩에 사후 SAE를 학습해 언어 정체성 latent만 끄고 원 공간으로 복원 | 아니요 (검색). **"사후 SAE로 불필요 신호를 끄고 성능 향상"의 틀** |

## D. SAE 인프라 (우리 모델에 쓸 수 있나)

검색 결과 기준이며 HF 저장소를 직접 열어 확인하지는 않음.

| 모델 | 우리 역할 | 공개 SAE (검색 결과) |
|---|---|---|
| R1-Distill-Qwen-1.5B | 쌍 A teacher (λ=0) | `DGurgurov/DeepSeek-R1-Distill-Qwen-1.5B-sae`, EleutherAI skip-transcoder, Galichin 외 (2503.18878) 코드 |
| R1-Distill-Llama-8B | 쌍 C teacher (λ=0) | SAE-Steering (2601.03595), Troitskii 외 crosscoder (2510.04128) 학습분 |
| Llama-3.1-8B(-Instruct) | 쌍 C teacher (λ=1) | Llama Scope (2410.20526, base 모델용), EleutherAI `sae-llama-3.1-8b-32x/64x` |
| Qwen3-4B 계열 | 쌍 B teacher | Qwen-Scope (2605.11887: Qwen3·Qwen3.5 7개 변형) — 4B 포함 여부 미확인 |
| Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct | student | "Breaking the Block" (2506.07691, instruct 모델용 SAE 학습법 FAST): 초록은 Qwen2.5-7B-Instruct, Llama-3.2-3B-it만 명시. 0.5B·1B 학습 여부와 공개 여부는 검색 요약에만 있어 미확인 |
| OLMo-2-1B-Instruct | student | 찾지 못함 |
| 보간 teacher (λ=0.2–0.8) | – | 없음. 직접 학습하거나 양 끝 SAE 재사용 필요 (2410.12391이 병합 모델 feature 추적) |

도구: SAELens, EleutherAI `sparsify` (TopK SAE·transcoder 학습).

## E. 반드시 다뤄야 할 반론

| arXiv | 논문 | venue | 요지 |
|---|---|---|---|
| 2601.05679 | Do Sparse Autoencoders Identify Reasoning Features in Language Models? (Ma, Liang, Chen 외 4) | ICML 2026 | 대조적으로 고른 "추론 feature" 후보의 45–90%가 추론과 무관한 텍스트에 관련 토큰 몇 개만 넣어도 켜짐. 희소성 정규화가 안정적인 단서 토큰 쪽으로 치우친다는 이론 분석 |
| 2605.28896 | Feature Geometry of LoRA Adapters | – | LoRA가 만든 표현은 사전학습 SAE 사전과 정렬이 약함 |

**의미:** SAE "반성 feature"도 결국 "wait" 토큰 탐지기일 수 있어서, 키워드 빈도를 SAE로 바꾸는 것만으로 타당성 문제가 해결되지는 않음. SAE를 쓰려면 Ma 외의 반증 절차(단서 토큰 주입)를 통과한 feature만 써야 함.

---

## F. 우리 연구와의 위치 (빈칸)

1. **가중치 쪽 분석은 비어 있음.** A·B의 거의 전부가 활성(SAE/crosscoder)을 봄. 증류가 LoRA 가중치에 남긴 구역을 찾고 가중치 공간에서 제거·이식한 연구는 찾지 못함. 2605.28896(LoRA feature geometry)만 가까운데 방향이 다름(adapter 활성에 SAE를 새로 학습).
2. **teacher를 연속으로 움직인 설정은 비어 있음.** 선행은 "증류 전후", "SFT vs RL", "Thinking vs NoThinking" 같은 이산 비교. 보간 teacher의 전환 시점과 student 표현 전환의 정렬은 없음.
3. **"SAE로 추론을 옮긴다"는 빈칸이 아님.** Resa(2506.09967)의 SAE-Tuning이 source 모델 SAE로 target 1.5B 모델의 SFT를 안내해 추론을 이끌어 냄. 남은 빈칸은 더 좁음: teacher 풀이로 학습하는 CoT 증류에서, teacher 스타일(λ)에 따라 어떤 feature/구역이 옮겨지는지와 그것을 골라 옮기는 방법. SAE를 손실(SASFT, SAE-FD), 초기화(SAILS), 데이터 선택(SaeRL, IRDS)에 쓴 선례도 모두 다른 과제임.
4. **가장 먼저 차별화해야 할 논문:** Baek & Tegmark (2503.03730), OPD crosscoder (2609.35210), Venhoff 외 (2510.07364), Resa (2506.09967), 반론 Ma 외 (2601.05679).

## G. 다음에 할 일 (제안, 결정 필요)

- 위 5편 PDF를 받아 `papers/`에 두고 Related Work 2.2를 보강.
- 교수님 2번(측정)을 받는다면: 모델을 바꾸지 않고 **teacher 쪽(R1-Distill-Qwen-1.5B, R1-Distill-Llama-8B)은 공개 SAE**로, student 쪽은 SAE-free 방향(2505.15634)이나 직접 학습한 SAE로 반성 feature를 재고, Ma 외의 토큰 주입 반증을 통과시키는 방식이 가능함.
- 교수님 3번(방법)을 받는다면: SASFT식 "SAE feature 손실"이나 SAILS식 "SAE 부분공간 LoRA 초기화"를 CoT 증류에 적용하는 것이 가장 직접적인 빈칸임. 다만 이는 분석 논문에서 방법 논문으로 방향을 바꾸는 결정임.
