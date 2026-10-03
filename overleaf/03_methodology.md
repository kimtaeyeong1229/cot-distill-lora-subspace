<!-- Notion에서 동기화: 2026-10-03 -->
# Methodology
## Distillation Setup
- **(teacher)**
    - student가 달라진 원인을 풀이 스타일로 좁히기 위해, teacher만 바꾸고 나머지 학습 조건은 전부 고정함
    - 추론 모델($\lambda=0$)과 지시 모델($\lambda=1$)을 가중치에서 섞어(보간) $\lambda \in \{0, 0.2, \dots, 1.0\}$의 여섯 teacher를 만듦 ([Wortsman et al., 2022](https://arxiv.org/abs/2203.05482); [Ilharco et al., 2023](https://arxiv.org/abs/2212.04089))
    - 쌍 A는 R1-Distill-Qwen-1.5B와 Qwen2.5-Math-1.5B-Instruct, 쌍 B는 Qwen3-4B-Thinking-2507과 Qwen3-4B-Instruct-2507, 쌍 C는 R1-Distill-Llama-8B와 Llama-3.1-8B-Instruct이며, 원소별로 섞는 것이 뜻을 갖도록 같은 base에서 갈라져 나온 모델끼리 짝지음
    - 쌍 B·C는 [Wu et al. (2026a)](https://arxiv.org/abs/2510.10977)와 같은 구성이라 teacher 쪽 현상을 재현으로 확인할 수 있음. 계열과 규모가 붙어 있어 규모 효과는 주장하지 않음
- **(student)**
    - Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct, OLMo-2-1B-Instruct를 씀. 모두 긴 CoT가 오히려 해로워지는 규모 대역임 ([Li et al., 2025](https://arxiv.org/abs/2502.12143))
    - teacher 쌍 셋과 student 셋을 모두 짝지은 아홉 조합을 이하 칸이라 부름
    - 계열을 셋으로 나눠 한 계열의 버릇이나 오염이 아님을 확인함 ([Wu et al., 2026b](https://arxiv.org/abs/2507.10532))
        - OLMo-2는 문맥이 4,096이라 가장 긴 $\lambda=0$ 풀이 일부는 뒷부분이 잘린 채 학습됨
- **(데이터)**
    - 모든 $\lambda$가 같은 문제를 같은 개수만큼 보도록 아래 순서로 만듦
        1. MATH train split에서 MATH-500과 겹치는 문제를 뺀 뒤 4,000문제를 뽑음
        2. 여섯 $\lambda$의 teacher가 각각 그 4,000문제를 풂
        3. 각 $\lambda$에서 끝까지 쓰고, 답이 맞고, student 학습 길이 안에 통째로 들어가는 풀이만 남김
        4. 여섯 $\lambda$가 전부 통과한 문제만 취해, 300문제를 떼어 둔 나머지(쌍 A·B·C 각 2,113·2,261·1,015문제)를 학습에 씀
- **(학습)**
    - 모든 student는 선형 층 7종에 $r=16$, $\alpha=32$의 LoRA를 붙이고, batch 크기 32와 학습률 $2\times10^{-4}$로 3 epoch 학습함. 최대 길이는 8,192(OLMo 4,096)임
        - LoRA는 효율이 아니라 측정을 위해 씀. 업데이트가 작은 행렬 한 쌍에 모여 학습끼리 같은 자리에서 비교할 수 있음
    - 각 칸과 $\lambda$마다 학습 집합 전체로 seed를 셋 바꿔 학습하고, 데이터만 달라졌을 때의 차이를 재기 위해 같은 seed 셋 각각에서 학습 집합을 겹치지 않는 두 반쪽으로 나눠 따로 학습한 adapter도 만듦
- **(평가)**
    - 생성은 teacher와 student 모두 $T=0.6$, top-p 0.95로 하며, 최대 길이는 teacher 8,192토큰, student 7,680토큰(OLMo 3,840)임
    - student는 모든 실험에서 MATH-500의 앞 200문제(난이도와 과목 비율이 전체와 같음)를 seed 셋의 adapter가 각각 2회씩 풀며, 표의 값은 seed별 값의 평균 ± 표준편차임
    - 정답률은 pass@1로 재고, 잘림이 오답으로 섞이지 않도록 완주 응답만 채점한 값을 함께 씀 ([Wang et al., 2026](https://arxiv.org/abs/2601.21684))
    - 생각하며 푸는지는 teacher에서는 `</think>` 비율로 ([Wu et al., 2026a](https://arxiv.org/abs/2510.10977)), student에서는 첫 응답의 반성 표현(wait · recheck · alternatively 등) 빈도로 셈 ([Yeo et al., 2025](https://arxiv.org/abs/2502.03373); [Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393)). 반성 표현은 대리 지표이므로 토큰 수와 생성 한도 도달 비율을 함께 적고 ([Chen et al., 2026](https://arxiv.org/abs/2601.06002); [Goyal et al., 2026](https://arxiv.org/abs/2608.04160)), 같은 문장을 되풀이하는 조건은 앞 300단어에서 셈
    - 단어만의 착시인지는 공개 SAE(sparse autoencoder)가 있는 Llama student의 내부 feature로 다시 보며 ([EleutherAI SAE](https://huggingface.co/EleutherAI/sae-Llama-3.2-1B-131k)), 반성 단어를 끼우거나 지우는 것만으로 켜지고 꺼지는 feature는 버림 ([Ma et al., 2026](https://arxiv.org/abs/2601.05679))
    - MATH 밖에서도 성립하는지는 MATH로 학습한 adapter를 그대로 GSM8K ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168))와 GPQA-Diamond ([Rein et al., 2024](https://openreview.net/forum?id=Ti67584b98))의 앞 200문제(GPQA-Diamond는 전체 198문제)에서 같은 설정으로 평가함
    - 증류가 student를 얼마나 바꿨는지 가늠하기 위해, 학습하기 전의 student도 같은 설정으로 평가함 (adapter가 없으므로 생성 seed를 셋 바꿈)
## Subspace Analysis
- **(분해)**
    $$
    \Delta W = \tfrac{\alpha}{r} BA = U\Sigma Q^\top, \qquad U = [\,u_1, \dots, u_r\,] \tag{1}
    $$
    $$
    \phi(\Delta W_i, \Delta W_j) = \lVert U_i^\top U_j \rVert_F^2 / r \tag{2}
    $$
    - 각 $\lambda$의 adapter가 가중치를 어느 방향으로 고치는지는, 층마다 업데이트를 분해한 앞 $r$개의 좌특이벡터 $U$로 봄
    - 두 adapter가 같은 방향을 쓰는지는 두 $U$가 포개지는 정도 $\phi$로 봄. 1이면 같은 방향이고 무작위 부분공간끼리는 $r/d$임($d$는 층의 출력 차원) ([Hu et al., 2022](https://arxiv.org/abs/2106.09685))
        - 층과 모듈별 값은 $\lVert \Delta W \rVert_F$로 가중 평균함
- **(비교)**
    $$
    R(\lambda) = \dfrac{\tfrac{1}{2}\left[\,\phi\!\left(\Delta W_a^{0},\, \Delta W_b^{\lambda}\right) + \phi\!\left(\Delta W_b^{0},\, \Delta W_a^{\lambda}\right)\right]}{\phi\!\left(\Delta W_a^{\lambda},\, \Delta W_b^{\lambda}\right)} \tag{3}
    $$
    - 방향이 달라진 것이 teacher 때문인지는, teacher가 다른 adapter끼리의 겹침을 데이터만 다른 adapter끼리의 겹침으로 나눈 $R$로 봄. 겹침을 바닥값 없이 읽어 해석이 흔들린 선행과 달리, 판정은 이 비율로만 함 ([Dieudonné et al., 2026](https://arxiv.org/abs/2606.28117))
        - 데이터 반쪽 $a$, $b$와 seed가 분자와 분모에 똑같이 들어가므로 teacher 효과만 남음. 1이면 teacher 효과가 데이터 차이 수준이고, 0에 가까우면 방향을 통째로 갈아탄 것임
        - seed만 다른 adapter끼리의 겹침은 기준선으로 쓰지 않음. 같은 seed로 학습한 adapter끼리는 LoRA 초기화를 공유해 겹침이 부풀기 때문이며, 분자와 분모도 따로 보고함 (부록 A)
    - adapter가 teacher가 생각을 그만두는 $\lambda$에서 방향을 갈아타는지는, $\lambda$ 구간마다 $R$의 낙차와 teacher `</think>` 비율의 낙차가 이루는 상관을 순열 검정으로 봄
        - 전환 $\lambda$를 한 점으로 미리 고르면 기준에 따라 답이 갈리고 후행 선택이 되기 때문임
        - teacher 쪽은 `</think>` 비율 대신 풀이 속 반성 표현의 낙차로도 같은 검정을 함
- **(추출)**
    $$
    P = \frac{1}{n}\sum_{t=1}^{n} U_t U_t^\top,\qquad V \leftarrow \{\,v : Pv = \mu v,\ \mu > 0.5\,\} \tag{4}
    $$
    - 전환 전 adapter들이 다 같이 쓰는 방향은, teacher의 `</think>` 비율이 처음으로 절반 넘게 떨어지는 구간을 전환 구간으로 삼아 그 이전 adapter들의 $U$를 겹치고 절반 넘게 함께 쓰는 방향 $V$만 남겨 봄 ([Gerasimov et al., 2026](https://arxiv.org/abs/2606.12138)). 이하 $V$를 공유 구역이라 부름
        - 각 adapter가 $V$를 쓰는 정도는 $\lVert U_t^\top V \rVert_F^2 / k$($k$는 $V$의 차원)로 재어 무작위 수준 $r/d$와 비교하며, 채점과 편집 모두 그 adapter를 빼고 만든 $V$를 씀
        - 실제로 쓸 수 있는지 보기 위해, 같은 데이터로 학습한 다른 seed의 $\lambda=0$ adapter 두 개만으로도 공유 구역을 만듦. adapter가 둘이면 짝지은 방향의 고윻값이 $(1+\cos\theta)/2$라 0.5 기준은 거의 모든 방향을 남기므로, 60° 안으로 겹치는 방향($\mu > 0.75$)만 남김
## Subspace Interventions
- **(제거)**
    $$
    \Delta W' = (I - VV^\top)\,\Delta W \tag{5}
    $$
    - 공유 구역이 생각하며 푸는 데 필요한지는, $\lambda=0$ adapter에서 $V$ 성분만 빼 봄 ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717))
        - $B \leftarrow B - V(V^\top B)$로 구현하므로 $A$와 rank는 그대로임
- **(이식)**
    $$
    \Delta W'(\lambda{=}1) = \Delta W(\lambda{=}1) + \beta\, VV^\top \Delta W(\lambda{=}0) \tag{6}
    $$
    - 공유 구역만으로 생각하며 풀게 되는지는, 짧게만 답하는 $\lambda=1$ adapter에 $\lambda=0$ adapter의 $V$ 성분만 더해 봄
- **(증폭)**
    $$
    \Delta W' = \lVert \Delta W \rVert_F \cdot \dfrac{\Delta W + \gamma\, VV^\top \Delta W}{\lVert \Delta W + \gamma\, VV^\top \Delta W \rVert_F} \tag{7}
    $$
    - 공유 구역이 생각의 양까지 조절하는지는, 노름은 그대로 묶은 채 $V$ 쪽 비중만 키워 봄
- **(대조)**
    - 효과가 $V$라는 자리 때문인지는, 같은 에너지를 $V$ 밖에서 지우거나 ([Lion et al., 2025](https://arxiv.org/abs/2506.03133); [Fu et al., 2026](https://arxiv.org/abs/2601.07411)), 업데이트 전체를 줄이거나, 가장 큰 특잇값 성분부터 지운 조건([Lee et al., 2025](https://arxiv.org/abs/2510.06750))과 비교함
    - 정답률이 오른 것이 짧아진 덕분일 뿐인지는 원래 adapter를 같은 길이에서 끊은 조건으로 ([Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393)), 편집 자체의 부작용인지는 공유 구역을 거의 쓰지 않는 $\lambda=1$ adapter에 같은 편집을 한 조건으로 봄
    - teacher를 섞지 않고 seed 2개로 찾은 구역이 생각을 얼마나 깔끔하게 끄는지는, "생각을 끝냈다"는 문장을 답 앞에 넣는 방법(태그를 배우지 않은 student라 태그 없이 문장만 넣음) ([Ma et al., 2025b](https://arxiv.org/abs/2504.09858)), 같은 길이에서 끊고 답을 쓰게 하는 방법 ([Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393)), 짧은 풀이 adapter와 반반 섞는 방법 ([Wu et al., 2025](https://arxiv.org/abs/2503.20641)), 학습 전 student와 비교함 (부록 D)
