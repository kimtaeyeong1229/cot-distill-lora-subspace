# EDIT: Capture the Key in Reasoning to Enhance CoT Distillation Generalization

| 항목 | 내용 |
|---|---|
| ACL Anthology | [2025.acl-long.21](https://aclanthology.org/2025.acl-long.21/) |
| 저자 | Chengwei Dai, Kun Li\*, Wei Zhou, Songlin Hu |
| 소속 | Institute of Information Engineering, CAS / School of Cyber Security, UCAS |
| 학회 | ACL 2025 (Volume 1: Long Papers), pages 441–465 |
| PDF | [materials/2025.acl-long.21_EDIT.pdf](../materials/2025.acl-long.21_EDIT.pdf) (25쪽) |
| 코드 | https://github.com/C-W-D/EDIT |
| 키워드 | CoT distillation, dual CoTs, minimum edit distance, key reasoning steps, token-level weighted loss, learning from mistakes, IND/OOD generalization, LoRA |

---

## TL;DR

- CoT는 대부분 **단순한 추론 형태**로 이루어져 있고, 결론을 실제로 좌우하는 **key reasoning step은 약 4.7%**뿐이라고 주장한다.
- 기존 CoT distillation은 teacher의 **맞은 CoT만** SFT하므로, student가 key step을 배우지 못하고 **teacher의 추론 "형식"만 흉내 내다가** 정작 중요한 곳에서 틀리거나 빠뜨린다 (Figure 1의 "unthinking student").
- EDIT(mistak**E**-**D**riven key reason**I**ng step distilla**T**ion)는 teacher에게 **비슷한 추론 경로를 따라가되 결론만 다른 dual CoTs**(정답 CoT ↔ 오답 CoT)를 생성시킨다.
- dual CoTs 쌍에 **minimum edit distance**를 적용해 insert/replace/delete된 토큰만 골라내고, 이 토큰에만 가중치를 줘서 **정답 쪽은 likelihood를 올리고 오답 쪽은 내리는** token-level loss로 학습한다.
- LLaMA2-7B 기준 Std-CoT 대비 5개 벤치 평균 **41.8 → 46.5 (+4.7점)**. IND(BBH-test) 54.2 → 60.9.
- **평가는 전부 정확도(accuracy)와 GPT-4 품질 점수뿐**이다. memorization을 정량화하는 지표는 하나도 없다 (→ §9 참조).

---

## 1. 문제의식

- LLM의 CoT 추론 능력을 SLM으로 옮기는 표준 방식(standard CoT distillation)은 teacher가 **맞힌** CoT 데이터로 student를 SFT하는 것이다 (Magister et al. 2023; Ho et al. 2023).
- 저자들의 관찰: CoT 시퀀스의 대부분은 문제를 다시 서술하거나 형식을 갖추는 "단순 추론 형태"이고, 결론을 뒤집는 **분기점(key reasoning step)은 극히 일부**다.
- 따라서 전체 시퀀스에 균일하게 NLL을 거는 SFT는 손실의 대부분을 "형식" 토큰이 차지하게 만들고, student는 **teacher의 문체·구조를 모사**하면서 정작 key step에서 오류/누락을 낸다.
- **Figure 1 예시**: teacher는 "If we remove all the magenta items from the floor, we are left with 2 silver ... Therefore, the answer is (C)"라고 답하고, student는 앞부분 문장을 거의 그대로 따라 쓰다가 "we are left with only the silver ... the number ... is three. Therefore, the answer is (D)"로 틀린다. 논문은 모사된 부분을 빨강, key step을 상자로 표시한다.
- 인간 학습 유추: **정답 풀이에 비추어 오답을 분석**할 때 성패를 가른 결정적 단계가 드러난다. 이걸 distillation에 옮긴 것이 EDIT.

### 4.7%라는 숫자의 출처

본문(§1)과 abstract에 나오는 "≈ 4.7% of key reasoning steps"의 근거는 **각주 2 한 줄이 전부**다:

> "We calculated the edit distance and its average proportion in the overall sequence on the dual CoT dataset mentioned in our subsequent methods §3."

즉 §3에서 만든 **자기들의 dual CoT 데이터셋(5,207쌍)** 위에서 edit distance가 전체 시퀀스 길이에서 차지하는 평균 비율을 계산한 값이다. 별도의 표/그림/분포/태스크별 분해는 **없다**. 자세한 유보 사항은 §8에 정리했다.

---

## 2. 방법 (Section 3)

전체 흐름은 3단계다 (Figure 2).

### Step 1. Teacher가 CoT 주석 (§3.1)

CoT Extraction Prompt $CEP$(few-shot, human-curated 예시 포함)로 원 데이터셋 $D=\{(q,a)\}$에서 CoT를 뽑는다.

$$\mathrm{CoT} \sim \mathrm{LLM}(CEP \oplus q)$$

STaR(Zelikman et al. 2022)를 따라 **최종 답의 정오로 두 갈래로 나눈다**. 기존 연구와 달리 **틀린 CoT를 버리지 않는 것**이 핵심.

- 정답 CoT 집합: $D^{+}=\{(q,\mathrm{CoT}^{+}) \mid \hat a = a\}$
- 오답 CoT 집합: $D^{-}=\{(q,\mathrm{CoT}^{-}) \mid \hat a \neq a\}$

### Step 2. Dual CoTs 생성 (§3.2)

**dual CoTs의 정의**: 원래 CoT와 **중간 추론 단계는 비슷하지만 결론이 갈리는** 대조 CoT.

**(a) 틀린 CoT를 고치기 (Rectify).** Rationalization(Zelikman et al. 2022)에서 영감을 받아 Answer Hint Prompt $AHP$를 쓴다. $CEP$와 **같은 in-context 예시**를 쓰되 정답을 먼저 알려주는 구조라, teacher가 원래의 틀린 추론 경로와 비슷하게 가되 정답에 도달한다.

$$\mathrm{CoT}^{-+} \sim \mathrm{LLM}(AHP \oplus q \oplus a) \;\Rightarrow\; D^{-+}=\{(q,\mathrm{CoT}^{-+})\}$$

**(b) 맞은 CoT를 망가뜨리기 (Corrupt).** 단순히 $AHP$에 틀린 답을 힌트로 주면 **LLM이 힌트를 무시하고 계속 정답을 낸다**(문제가 쉽고, RLHF를 거쳐 unhelpful한 답을 거부하기 때문). 그래서 Contrastive CoTs Prompt $CCP$를 설계한다. $D^{-}$의 오답 예시와 $D^{-+}$의 정답 예시를 **쌍으로 묶어** in-context 예시로 넣어, teacher가 "그럴듯한 오답 + 그 근거"를 생성하도록 유도한다.

$$\mathrm{CoT}^{+-} \sim \mathrm{LLM}(CCP \oplus q \oplus \mathrm{CoT}^{+}) \;\Rightarrow\; D^{+-}=\{(q,\mathrm{CoT}^{+-})\}$$

**품질 검증**: 무작위 100쌍을 사람이 직접 확인해 **93쌍**이 rationale과 결론이 논리적으로 일관됨을 확인했다 (각주 3).

### Step 3. Student 학습 (§3.3)

**(3-1) 정답 CoT로 SFT.** 원래 정답 CoT와 교정된 CoT를 합친 $D^{+}_{merge}=D^{+}\cup D^{-+}$로 먼저 fine-tuning:

$$\pi_{sft} = \arg\max_{\pi}\ \mathbb{E}_{q,\mathrm{CoT}\sim D^{+}_{merge}} \log \pi(\mathrm{CoT}\mid q)$$

**(3-2) Key Reasoning Steps Learning (KRSL).** Guo et al. (2024)의 fine-grained quality signal 아이디어를 차용.

*Step1 — 쌍 구성.*
- 원래 맞았던 쪽: $D^{+}_{dual}=\{(q,\mathrm{CoT}^{+},\mathrm{CoT}^{+-})\}$
- 원래 틀렸던 쪽: $D^{-}_{dual}=\{(q,\mathrm{CoT}^{-+},\mathrm{CoT}^{-})\}$
- 최종: $D_{dual}=D^{+}_{dual}\cup D^{-}_{dual}$

*Step2 — minimum edit distance로 key step 식별 + 토큰 가중치.* 정답 CoT와 오답 CoT를 정렬해, **오답 쪽에서 삽입/치환된 구간**과 **정답 쪽에서 삽입/치환된 구간**을 key reasoning step으로 본다. 동일한 토큰은 가중치 0으로 **무시**한다.

$$\omega^{+}_{t}=\begin{cases}\alpha, & \mathrm{CoT}^{+}_{t}\ \text{가 inserted 또는 replaced}\\ 0, & \text{otherwise}\end{cases}
\qquad
\omega^{-}_{t}=\begin{cases}\beta, & \mathrm{CoT}^{-}_{t}\ \text{가 deleted 또는 replaced}\\ 0, & \text{otherwise}\end{cases}$$

($\alpha\ge 0$, $\beta\ge 0$)

*Step3 — 가중 손실.* 정답 key step은 likelihood를 올리고, 오답 key step은 내린다.

$$L(\pi,q,\mathrm{CoT},\omega) = -\sum_{\mathrm{CoT}_t \in \mathrm{CoT}} \omega_t \log \pi(\mathrm{CoT}_t \mid q, \mathrm{CoT}_{<t})$$

$$\max_{\pi_{sft}}\ \mathbb{E}_{q,\mathrm{CoT}^{+},\mathrm{CoT}^{-}\sim D_{dual}}\ L(\pi_{sft},q,\mathrm{CoT}^{+},\omega^{+}) - L(\pi_{sft},q,\mathrm{CoT}^{-},\omega^{-})$$

> **주의**: 식 (7)의 $L$은 이미 음의 로그우도(NLL)인데 식 (6)은 이를 $\max$한다고 적혀 있다. 반면 본문은 "정답 쪽은 weighted negative log-likelihood로, 오답 쪽은 weighted positive log-likelihood로 최적화하고 **둘의 합을 최종 loss로 삼는다**"고 서술한다. 즉 식 (6)의 $\max$는 최소화해야 할 loss 표기와 부호가 맞지 않는다 (§8).

### KRSL vs DPO (Appendix A.8)

positive/negative를 같이 쓴다는 점에서 DPO와 닮았지만 저자들은 두 가지 차이를 든다.
1. KRSL은 **극도로 비슷한** 쌍(dual CoTs)을 쓰는 반면 DPO는 완전히 다른 선호 쌍을 쓴다.
2. DPO의 loss는 **모든 토큰 위치**의 NLL을 합하므로, 시퀀스의 대부분을 차지하는 동일 토큰이 보상을 지배해 유사한 두 텍스트를 구별하지 못한다.

실제로 EDIT에 DPO를 붙이면(TRL `dpo_trainer`, lr 1e-5, cosine, warmup 0.3, beta 0.1, prompt 512 / max 1024, 10 epochs, batch 16) **평균 8.1점으로 붕괴**했고(Table 9), 생성이 추론 없이 답만 뱉거나 질문을 무한 반복하는 패턴으로 무너졌다 (Table 10).

---

## 3. 실험 설정 (Section 4.1, Appendix A.9)

| 항목 | 설정 |
|---|---|
| Teacher | **ChatGPT `gpt-3.5-turbo-0613`** (black-box). CoT 추출 시 Suzgun et al. (2023)의 manual prompt 사용 |
| Student (main) | **LLaMA2-7B** |
| Student (ablation) | TinyLLaMA-1.1B, LLaMA2-13B / CodeLLaMA-7B, LLaMA3-8B, Mistral-7B-v0.2 |
| IND 데이터 | **BBH (BIG-Bench Hard)** 27개 subtask, 총 6,511문항 → subtask별 **4:1**로 BBH-train(distillation) / BBH-test(IND 평가) 분할 |
| OOD 데이터 | **BB-sub** (BIG-Bench에서 "multiple-choice"·"reasoning" 키워드로 추린 61 subtask, subtask당 최대 100개, 총 5,384) / **AGIEval** (영어 객관식 8 task) / **ARC-E** (2,376) / **ARC-C** (1,172) |
| Dual CoTs 규모 | $D^{+}_{dual}$ 3,805 + $D^{-}_{dual}$ 1,402 = $D_{dual}$ **5,207쌍** (총 학습 데이터 10,414) |
| Fine-tuning | **LoRA** (전체 fine-tuning 아님): rank **64**, $\alpha$ **32**, dropout 0.05, target modules **q_proj, v_proj** |
| 학습 | SFT lr **2e-4** (1e-4/2e-4/3e-4 중 탐색), KRSL lr **5e-6**, AdamW (0.9, 0.999), $\epsilon$ 1e-8, scheduler $\gamma$ 0.95, weight decay 0, warmup 0, max length 1024 |
| Epochs | 7B/8B: 10, 13B: 15, 1.1B: 20 (Table 16) |
| KRSL 하이퍼파라미터 | $\alpha=1.0$, $\beta=0.025$ |
| 체크포인트 선택 | **IND 태스크에서 가장 성능 좋은 epoch**을 고른 뒤 그 체크포인트로 OOD 평가 |
| 하드웨어 | 4×A100 80GB, mixed precision, FSDP, Meta `llama-recipes` 기반. 7B SFT ≈ 40분/epoch, KRSL ≈ 90분/epoch |
| 추론 | vLLM, student는 **greedy** (do_sample False, top-p 1.0, max new tokens 1024), teacher는 temperature 0.2, max 2048 |

**Baselines**: Teacher/Vanilla student의 Zero/Few-shot (+CoT), **Std-CoT** (Magister et al. 2023), **MT-CoT** (Li et al. 2022), **SCOTT** (Wang et al. 2023a), **SBS** (Hsieh et al. 2023), **SBS-MI** (Chen et al. 2024b), 그리고 EDIT의 내부 변형들(w/o RWC, w/o KRSL 등).

---

## 4. 주요 결과

### 4.1 Main Results (Table 1, LLaMA2-7B, Accuracy %)

| Method | Distill? | BBH-test (IND) | BB-sub | AGIEval | ARC-E | ARC-C | AVG |
|---|---|---|---|---|---|---|---|
| Teacher: ChatGPT Zero-shot-CoT | ✕ | 42.7 | 44.1 | 49.5 | 91.9 | 81.1 | 61.9 |
| Teacher: ChatGPT Few-shot-CoT | ✕ | 73.1 | - | - | - | - | - |
| Student Zero-shot | ✕ | 14.8 | 15.5 | 6.9 | 18.2 | 13.9 | 13.9 |
| Student Few-shot | ✕ | 15.1 | 28.5 | 25.5 | 25.5 | 25.4 | 24.0 |
| Student Zero-shot-CoT | ✕ | 10.6 | 7.7 | 7.1 | 18.4 | 14.8 | 11.7 |
| Student Few-shot-CoT | ✕ | 16.3 | 25.3 | 9.9 | 17.2 | 17.2 | 17.2 |
| MT-CoT | ✓ | 56.8 | 30.3 | 22.0 | 49.4 | 38.2 | 39.3 |
| SCOTT | ✓ | 42.4 | 18.8 | 13.0 | 45.7 | 34.1 | 30.8 |
| Std-CoT | ✓ | 54.2 | 28.7 | 21.6 | 59.6 | 45.1 | 41.8 |
| SBS | ✓ | 42.4 | 27.7 | 28.8 | **68.5** | 48.6 | 43.2 |
| SBS-MI | ✓ | 42.9 | 24.3 | 29.2 | 68.4 | 49.3 | 42.8 |
| w/o RWC + KRSL on $D^{+}_{dual}$ | ✓ | 55.1 | 30.1 | 24.1 | 60.3 | 44.1 | 42.7 |
| w/o RWC + KRSL on $D_{dual}$ | ✓ | 55.4 | 30.1 | 24.2 | 63.6 | 48.3 | 44.3 |
| w/ RWC + w/o KRSL | ✓ | 59.7 | 30.0 | 24.5 | 61.9 | 45.5 | 44.3 |
| **EDIT (w/ RWC + w/ KRSL on $D_{dual}$)** | ✓ | **60.9** | **31.1** | 25.9 | 64.1 | **50.5** | **46.5** |

- Std-CoT 대비 평균 **+4.7점** (41.8 → 46.5). IND뿐 아니라 OOD 4개 중 3개에서도 개선.
- **AGIEval과 ARC-E에서는 SBS가 더 낫다.** 저자 설명: 이 데이터셋들은 질문-답 상관이 강해 rationale 없이 답을 직접 예측하는 SBS가 유리하다는 것. 다만 SBS는 생성한 rationale이 답의 논리와 불일치한다는 단점이 있다고 반박한다 (Dai et al. 2024).
- Ablation: RWC(교정된 오답 CoT) 제거와 KRSL 제거 모두 거의 모든 벤치에서 성능을 떨어뜨린다.
- IND에서 EDIT(60.9)는 teacher의 **Zero-shot-CoT(42.7)는 넘지만 Few-shot-CoT(73.1)에는 못 미친다.**

### 4.2 모델 크기 (Figure 4, Figure 8)

TinyLLaMA-1.1B / LLaMA2-7B / LLaMA2-13B 모두에서 EDIT가 baseline을 앞선다. 다만 **1.1B는 ARC-E/ARC-C 같은 쉬운 벤치에서 개선폭이 작고**, BB-sub·AGIEval 같은 어려운 벤치에서는 전 크기대에서 뚜렷한 개선이 있다. 저자 해석: (1) 작은 모델은 용량 제약으로 복잡한 추론 습득이 어렵고, (2) 쉬운 벤치는 개선 여지가 적다. → **"어려운 과제일수록 모사가 아닌 진짜 추론을 요구하므로 EDIT의 이득이 커진다."** (수치는 그래프로만 제시)

### 4.3 아키텍처 (Figure 5)

| Student | IND AVG: Std-CoT / EDIT w/o KRSL / **EDIT** | OOD AVG: Std-CoT / EDIT w/o KRSL / **EDIT** |
|---|---|---|
| CodeLLaMA-7B | 56.2 / 61.2 / **61.7** | 30.8 / 30.8 / **31.9** |
| LLaMA3-8B | 66.9 / 68.2 / **69.5** | 49.1 / 50.5 / **51.2** |
| Mistral-7B-v0.2 | 72.2 / 75.5 / **77.9** | 49.1 / 50.1 / **53.5** |

- 모든 아키텍처에서 EDIT > w/o KRSL > Std-CoT.
- **강한 base model일수록 격차가 크다** (Mistral: IND +5.7, OOD +4.4).

### 4.4 Key reasoning step의 정오 기여 (Figure 6)

`w/o Correct` = 오답 CoT의 key step만 학습 ($\alpha=0$), `w/o Wrong` = 정답 CoT의 key step만 학습 ($\beta=0$).

| | BBH-test | BB-sub | AGIEval | ARC-E | ARC-C |
|---|---|---|---|---|---|
| w/o Correct | 59.0 | 30.2 | 24.1 | 62.1 | 45.9 |
| w/o Wrong | 60.2 | 30.5 | 23.4 | 62.7 | 48.0 |
| **EDIT** | **60.9** | **31.1** | **25.9** | **64.1** | **50.5** |

한쪽만 학습하면 모두 떨어진다. **정답 쪽 key step을 뺐을 때(w/o Correct) 하락폭이 더 크다** → 정답 CoT의 key step이 더 중요하다.

### 4.5 어떤 dual CoTs가 중요한가 (Table 2)

| Dataset | $D^{+}_{dual}$ (n=3,805) | $D^{-}_{dual}$ (n=1,402) | $D_{dual}$ (n=5,207) |
|---|---|---|---|
| BBH-test | **61.3** | 60.9 | 60.9 |
| BB-sub | **31.2** | 30.8 | 31.1 |
| AGIEval | 24.4 | **26.0** | 25.9 |
| ARC-E | **64.6** | 63.8 | 64.1 |
| ARC-C | 48.9 | **50.5** | **50.5** |
| AVG | 46.1 | 46.4 | **46.5** |

**데이터가 2.7배 적은데도** $D^{-}_{dual}$(원래 teacher가 틀렸던 문제에서 만든 쌍)이 $D^{+}_{dual}$보다 평균이 높다 → 어려운 dual CoTs가 key step을 더 효과적으로 드러낸다.

### 4.6 하이퍼파라미터 (Table 4)

| Group | $\alpha$ | $\beta$ | BBH-test | BB-sub | AGIEval | ARC-E | ARC-C | AVG |
|---|---|---|---|---|---|---|---|---|
| A | 0 | 0 | 59.7 | 30.0 | 24.5 | 61.9 | 45.5 | 44.32 |
| B | 0 | 0.025 | 59.0 | 30.2 | 24.1 | 62.1 | 45.9 | 44.26 |
| C | 1 | 0 | 60.2 | 30.5 | 23.4 | 62.7 | 48.0 | 44.96 |
| **D** | **1** | **0.025** | **60.9** | **31.1** | **25.9** | **64.1** | **50.5** | **46.50** |
| E | 1 | 0.05 | 59.7 | 30.0 | 24.7 | 61.9 | 45.5 | 44.36 |

$\alpha$를 0→1로 올리면 크게 개선되지만, $\beta$를 0.025 이상으로 키우면 오히려 떨어진다. 두 항이 **협력적이면서 동시에 적대적**인 관계라 균형이 필요하다.

### 4.7 데이터 양 통제 (Cost Analysis, Table 5)

dual CoTs 때문에 학습 데이터가 2배(10,414)라는 반론에 대한 통제 실험.

| Method | Training Data Size | BBH-test | BB-sub | AGIEval | ARC-E | ARC-C | AVG |
|---|---|---|---|---|---|---|---|
| Std-CoT w/ Repeat Sampling | 10,414 | 59.4 | 30.3 | 24.0 | 58.0 | 42.1 | 42.8 |
| Std-CoT w/ Dual CoTs | 10,414 | 54.8 | **32.9** | 25.1 | 62.2 | 44.1 | 43.8 |
| **EDIT** | 10,414 | **60.9** | 31.1 | **25.9** | **64.1** | **50.5** | **46.5** |

데이터 양을 맞춰도 EDIT가 앞선다 → 이득은 데이터 증량이 아니라 **key step 학습 방식**에서 온다. 부수적으로, dual CoTs를 (counterfactual 마커를 붙여) 그냥 넣기만 해도 Repeat Sampling보다 **OOD**가 좋아진다. 저자는 이를 "overfitting을 줄이고 추론을 더 잘 일반화한 것"으로 해석한다.

### 4.8 기타

- **추가 subtask (Table 3, AVG)**: EDIT 42.1 > Std-CoT w/ Dual CoTs 38.8 > Std-CoT 38.3 > Std-CoT w/ Repeat 38.1 > MT-CoT 34.9 > SCOTT 27.7. GSM8K 17.5, MATH **5.6**, AQuA-RAT 24.4, SAT-Math 24.5. 학습 데이터 5,207개 중 수학 문제는 **200개뿐**인데도 수학 벤치에서 개선.
- **Self-Consistency 결합 (Table 8, maj@8, temp 0.7/top-p 0.95)**: EDIT 49.1로 최고 (MT-CoT 46.4, Std-CoT 45.8, Std-CoT w/ Dual CoTs 46.3, SCOTT 35.5). EDIT 단독 46.5 → 49.1.
- **Mistake pattern (Table 6·7)**: 오답을 Logical(LEs)/Knowledge(KEs)/Mathematical Calculation(MCEs)/Other로 분류. 데이터 규모를 412개로 맞춘 비교에서 $D_{LEs}$ 44.9 > $D_{KEs}$ 44.6 > $D_{MCEs}$ 44.5. **논리 오류가 가장 유용**하다 — LEs는 넓은 범위의 추론 패턴을 주는 반면 KEs/MCEs는 너무 국소적이라 일반 해법을 배우기 어렵다.
- **CoT 품질 (Figure 7)**: GPT-4로 Std-CoT / EDIT / teacher의 CoT가 "key reasoning step을 얼마나 잘 드러내는가"를 1–10으로 채점. EDIT의 점수 분포가 Std-CoT보다 **teacher 분포에 더 가깝다** (KDE로 시각화, 수치표는 없음).
- **Case study (Table 20–24)**: teacher와 Std-CoT가 **같은 위치**에서 (오류 성격은 달라도) 틀리는 반면 EDIT는 그 지점을 올바르게 처리한다. Table 24에서는 teacher/Std-CoT가 선택지를 나열·분석하는 논리를 쓰는데 EDIT는 **각 선택지에 질문을 제기하는 다른 방식**으로 푼다 → "teacher CoT의 추론 단계에 overfit하지 않고 자기 논리를 적응시킨 것"이라고 해석한다.

---

## 5. 저자가 밝힌 한계 (Limitations)

논문의 Limitations 절은 **한 문단**뿐이다.

> 현재 CoT distillation 평가의 대부분이 **정확도(accuracy)에 치중**하는데, 안전한 LLM은 신뢰할 수 있는 CoT에 크게 의존하므로 이는 불충분하다. GPT-4의 자동 평가에만 의존하지 말고 **커뮤니티가 CoT 품질 평가 표준을 개발하기를 희망**한다.

즉 저자들도 **자신들의 평가가 정확도 + GPT-4 자동채점에 머문다는 점을 스스로 한계로 인정**한다. 여기서 언급되지 않은 한계는 §8에 따로 정리했다.

---

## 6. 재현 관련 세부 사항

- 코드는 https://github.com/C-W-D/EDIT (본문 각주 1). 다만 Appendix A.9.3에는 "We will release our code in the future"라고 적혀 있어 표기가 엇갈린다.
- 학습 코드는 Meta `llama-recipes` 수정본, FSDP 병렬.
- BBH 4:1 분할의 구체적 시드/인덱스는 명시되지 않았다.
- teacher가 black-box API(`gpt-3.5-turbo-0613`, 현재 deprecated)라 dual CoTs 데이터 재생성은 사실상 불가능. 공개 데이터에 의존해야 한다.

---

## 7. 핵심 시사점

- **손실을 어디에 거는가**가 CoT distillation에서 중요한 설계 축이다. 전 토큰 균일 NLL은 "형식" 토큰에 예산을 낭비한다.
- **teacher의 오답은 버릴 데이터가 아니다.** 정답 쌍과 대조시키면 key step의 위치를 공짜로 라벨링해 준다 (minimum edit distance만 있으면 된다).
- **dual CoTs라는 "거의 같은 두 시퀀스"에는 DPO가 맞지 않는다.** 동일 토큰이 보상을 지배하기 때문. 차이 토큰만 골라내는 것이 필수다 (Table 9: 46.5 vs 8.1).
- 어려운 문제에서 만든 대조쌍이 쉬운 문제에서 만든 것보다 **적은 데이터로 더 큰 효과**를 낸다.

---

## 8. 읽으면서 확인한 세부 사항 / 주의점

- **"4.7%"의 근거가 각주 한 줄뿐이다.** (a) 별도 표·분포·태스크별 분해가 없고, (b) 토큰 단위인지 단어/문자 단위인지, 어떤 tokenizer로 정렬했는지 명시가 없으며, (c) **자기들이 생성한 합성 dual CoTs 위에서 측정한 값**이다. dual CoTs는 "중간 단계는 비슷하게, 결론만 다르게"라는 프롬프트 제약으로 만들어졌으므로 edit distance가 작게 나오는 것은 **구성 방식의 산물에 가깝다**. 즉 "자연스러운 CoT에서 key step이 4.7%"라는 독립적 측정이 아니라 **"이 데이터 구성에서 두 CoT가 4.7%만 다르다"**는 기술 통계다. 이 수치를 인용할 때 주의가 필요하다.
- **4.7이 두 번 등장한다.** key step 비율 ≈4.7%(§1)와 Std-CoT 대비 평균 개선폭 +4.7점(§4.2). 서로 무관한 우연이다.
- **식 (6)의 부호.** $L$(식 7)은 이미 NLL인데 식 (6)은 $\max$로 적혀 있다. 본문 서술("둘의 합을 최종 loss로")과 맞추려면 최소화여야 한다. 표기 오류로 보인다.
- **epoch 수 불일치.** 본문 A.9.2는 "7B 모델은 15 epoch"이라 하지만 Table 16은 7B/8B에 **10 epoch**, 13B에 15 epoch으로 적혀 있다.
- **체크포인트를 IND 성능으로 고른 뒤 OOD를 평가**한다. OOD 숫자는 IND-selected 모델의 것이므로 완전히 독립적인 일반화 측정이 아니다.
- **평가가 거의 전부 객관식**이다. BBH는 대부분, BB-sub·AGIEval·ARC는 전부 multiple-choice로 필터링했다. 답이 유한한 선택지에서 나오므로 정확도가 추론 품질을 얼마나 반영하는지 제한적이다.
- **teacher가 하나뿐이다.** `gpt-3.5-turbo-0613` 고정. teacher 스타일/길이를 변주한 실험은 없다.
- EDIT의 MATH 점수는 **5.6**으로, 수학 추론에서는 절대 성능이 매우 낮다.
- GPT-4 품질 평가(Figure 7)는 밀도 곡선으로만 제시되고 평균/분산 등 수치가 없다.
- CodeLLaMA-7B의 OOD에서 Std-CoT와 EDIT w/o KRSL이 **동률(30.8)**이라, w/o KRSL 단계의 이득이 항상 있는 것은 아니다.

---

## 9. 우리 실험과의 관계

### 9.1 결론부터: 이 논문은 memorization을 **전혀 측정하지 않는다**

EDIT의 동기 서술은 우리 실험의 문제의식과 **문장 수준에서는 거의 같다**. "student가 key를 배우지 못하고 teacher의 추론 형식을 모사한다(imitating the teacher's reasoning forms)", "mere imitation이 아닌 진짜 추론", "teacher CoT의 추론 단계에 overfit하지 않는다" 같은 표현이 반복된다. 그러나 **이 주장을 뒷받침하는 정량 지표는 IND/OOD 정확도와 GPT-4 자동채점, 그리고 몇 개의 case study가 전부다.** 아래 항목별로 확인했다 (전문에 대해 `memoriz|verbatim|bits|codelength|compress|description length|train accuracy|held-out|n-gram` 등을 grep한 결과 포함).

| 항목 | 이 논문에 있는가 | 근거 |
|---|---|---|
| **(a) bits / codelength 계량** | **없음 (No)** | "bit", "codelength", "description length", "compression", "MDL"이 논문 전체에 **단 한 번도 등장하지 않는다**. 정보이론적 양을 다루지 않는다. mutual information은 baseline SBS-MI(Chen et al. 2024b)를 **인용할 때만** 나오고 저자들이 계산하지 않는다. |
| **(b) train vs held-out likelihood/accuracy gap** | **없음 (No)** | **학습 데이터에 대한 정확도나 likelihood를 단 한 번도 보고하지 않는다.** BBH를 4:1로 나눈 BBH-test는 문항 수준 held-out이지만, 대응하는 BBH-train 성능이 없어 gap을 만들 수 없다. IND↔OOD 비교는 "같은 분포 내 held-out vs 다른 벤치마크"일 뿐 train-vs-heldout이 아니다. 게다가 **체크포인트를 IND 성능으로 선택**하므로 (A.9.2) 애초에 train/heldout gap을 관측할 설계가 아니다. training loss 곡선은 "15 epoch쯤 평탄해졌다"는 서술만 있고 그림조차 없다. |
| **(c) teacher rationale의 verbatim 재생산** | **없음 (No)** | verbatim/exact-match/n-gram overlap/ROUGE/copy rate 같은 측정이 없다. 가장 가까운 것은 **정성적 관찰 두 가지**다: (i) Figure 1에서 student가 teacher 문장을 따라 쓴 부분을 빨강으로 **손으로 표시**한 것, (ii) Appendix A.5에서 "teacher와 Std-CoT가 **같은 위치에서** 틀린다"는 case study(Table 20·21). 둘 다 **사례 제시이지 측정이 아니다.** 재생산 비율을 숫자로 적은 곳은 없다. |
| **(d) 연속적인 teacher-style 축 (λ 스펙트럼 등)** | **없음 (No)** | teacher는 `gpt-3.5-turbo-0613` **하나로 고정**이다. teacher의 길이/스타일을 연속적으로 변화시킨 축이 없다. 논문에 존재하는 "축"은 전부 이산적이거나 다른 성격이다: 오류 패턴 3분류(LEs/KEs/MCEs), dual CoTs 구성($D^{+}_{dual}$/$D^{-}_{dual}$), 그리고 하이퍼파라미터 $\alpha\in\{0,1\}$·$\beta\in\{0,0.025,0.05\}$ grid(Table 4). $\alpha,\beta$는 **loss 가중치**이지 teacher 스타일이 아니다. |
| **(e) LoRA/PEFT adapter** | **사용은 함 (Yes), 측정 도구로는 아님 (No)** | 모든 실험이 LoRA다: **rank 64, $\alpha$ 32, dropout 0.05, target = q_proj·v_proj** (Table 16). 하지만 LoRA는 순수하게 **저비용 학습 수단**으로만 쓰인다. adapter의 용량·codelength·rank별 흡수량 같은 분석이 전혀 없고, rank를 변주한 ablation조차 없다 (ablation은 모델 크기와 아키텍처에 대해서만 한다). |

**요약**: (a) 없음, (b) 없음, (c) 없음, (d) 없음, (e) LoRA를 쓰지만 분석 도구로는 쓰지 않음. → **이 논문은 IND 정확도와 OOD 정확도만 보고한다** (+ GPT-4 품질 점수 분포와 case study).

### 9.2 저자들이 "일반화"를 측정하는 방식

- **OOD 정확도 4종** (BB-sub, AGIEval, ARC-E, ARC-C)의 평균. 논문에서 "generalization"은 사실상 이 평균과 동의어다.
- 보조 증거 세 가지: (i) **모델 크기·아키텍처 전반의 일관성** (Figure 4·5), (ii) **데이터량 통제 baseline** (Table 5) — 이득이 데이터 증량 탓이 아님을 보이고, "Std-CoT w/ Dual CoTs가 Repeat Sampling보다 OOD가 좋다 → overfitting을 줄인다"는 **유일하게 overfitting을 언급한 정량 논거**인데, 이 역시 **5개 벤치 평균 1점 차(42.8 → 43.8)** 가 근거의 전부이며 memorization 자체는 재지 않는다, (iii) **GPT-4 품질 점수 분포가 teacher에 가까워짐** (Figure 7).
- 학습 데이터 5,207개 중 수학이 200개뿐인데 GSM8K/MATH/AQuA에서 개선되었다는 점(Table 3)을 **도메인 전이의 증거**로 든다.

### 9.3 우리 실험이 선점당하는가 — 아니다

우리는 λ∈[0,1] 보간 teacher 스펙트럼에서 뽑은 CoT를 고정된 MATH 2,113문제 위에서 0.5B student에 LoRA r=16으로 distill하고, adapter가 흡수한 양 $W$, held-out으로 전이되지 않는 부분 $\Delta W$(= memorization), adapter codelength $CL$을 **bit 단위로** 측정했다. memorization share가 λ=0의 9%에서 λ=0.8–1.0의 약 120%까지 오르고, held-out 이득이 λ≈0.5에서 부호를 바꾸며, train-vs-held 정확도 격차와 verbatim prefix 재생산으로 교차 검증된다.

EDIT와의 관계는 다음과 같이 정리된다.

1. **주장은 겹치지만 증거의 종류가 다르다.** EDIT는 "SFT student가 teacher의 표면적 추론 형식을 모사한다"는 **명제를 세우고 정성적으로 예시한 뒤, 그것을 고치는 방법의 효과를 정확도로 입증**한다. 우리는 **그 명제 자체를 bit 단위로 계량**한다. EDIT는 "얼마나 모사하는가"에 숫자를 붙인 적이 없으므로, 우리의 정량화를 선점하지 않는다. 논문의 Limitations가 "정확도 중심 평가는 불충분하다"고 스스로 적은 바로 그 빈틈이 우리 기여 지점이다.
2. **λ 축이 없다.** EDIT의 teacher는 고정이고 long/short CoT, reasoning-style 변주를 다루지 않는다. teacher 스타일이 연속적으로 변할 때 memorization share가 단조 증가하고 held-out 이득이 부호를 바꾼다는 현상은 EDIT의 실험 공간에 아예 존재하지 않는다.
3. **LoRA를 둘 다 쓰지만 역할이 정반대다.** EDIT에게 LoRA(r=64, q/v)는 4×A100에서 7B를 돌리기 위한 수단이다. 우리에게 adapter는 **측정 장치**($CL$, $\Delta W$)다. rank를 바꾼 실험이 EDIT에 없다는 점도 확인했다.
4. **선행 연구로서의 정확한 위치**: EDIT는 우리 논지의 **동기(motivation) 문헌**이자 **비교 대상 개입(intervention)**이다. "표면 모사 문제"를 ACL long paper 수준에서 명시적으로 제기한 선행이므로 문제 설정의 정당성을 빌려올 수 있고(인용 가치 높음), 동시에 "그 주장이 지금까지 정량화된 적이 없다"는 gap 서술의 근거로도 쓸 수 있다. 우리 논문에서는 **"관련 연구가 이 실패 양상을 지적했지만 정확도 대리지표로만 다뤘다"**는 문장의 대표 인용으로 배치하는 것이 적절하다.
5. **방법론적으로 빌려올 만한 것**: minimum edit distance로 dual CoT 쌍에서 차이 토큰을 뽑아내는 절차는, 우리 실험에서 **verbatim 재생산을 토큰 수준으로 세분화**하는 데 재활용할 수 있다. 예컨대 student 출력과 teacher rationale 사이의 edit alignment를 구해, 재생산된 구간이 "형식 토큰"인지 "결론을 바꾸는 토큰"인지 나눠 $\Delta W$를 분해하면, "무엇을 암기했는가"를 위치별로 말할 수 있다. EDIT가 라벨링 도구를 제공하되 계량은 하지 않았으므로 조합 여지가 있다.
6. **반대 방향의 주의점**: EDIT의 결과는 λ 축과 무관하게 **"손실 가중치를 key step으로 옮기면 OOD가 오른다"**를 보여준다. 우리가 λ에서 관측한 memorization 증가가 **teacher 스타일의 문제인지, 균일 NLL이라는 손실 설계의 문제인지** 구분하려면 이 논문이 자연스러운 대조군이 된다. 우리 세팅에서 KRSL류의 토큰 가중을 적용했을 때 $\Delta W$가 실제로 줄어드는지는 아직 아무도 측정하지 않았고, 이는 후속 실험으로 바로 세울 수 있다.
