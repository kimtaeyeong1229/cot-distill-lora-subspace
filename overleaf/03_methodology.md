<!-- Notion에서 동기화: 2026-10-04 -->
# Methodology
## Distillation Setup
- **(teacher)**
    - 원인을 풀이 스타일로 좁히기 위해 teacher만 바꾸며, 같은 base에서 갈라져 나온 추론 모델($\lambda=0$)과 지시 모델($\lambda=1$)의 가중치를 섞어(보간) $\lambda \in \{0, 0.2, \dots, 1.0\}$의 여섯 teacher를 만듦 ([Wortsman et al., 2022](https://arxiv.org/abs/2203.05482); [Ilharco et al., 2023](https://arxiv.org/abs/2212.04089))
    - 쌍 A는 R1-Distill-Qwen-1.5B와 Qwen2.5-Math-1.5B-Instruct, 쌍 B는 Qwen3-4B-Thinking-2507과 Qwen3-4B-Instruct-2507, 쌍 C는 R1-Distill-Llama-8B와 Llama-3.1-8B-Instruct를 섞음 ([Guo et al., 2025](https://arxiv.org/abs/2501.12948); [Yang et al., 2024](https://arxiv.org/abs/2409.12122); [Yang et al., 2025](https://arxiv.org/abs/2505.09388); [Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783))
    - 쌍 B와 C는 [Wu et al. (2026a)](https://arxiv.org/abs/2510.10977)를 따르므로 teacher 쪽은 재현임. 쌍마다 계열과 규모가 함께 바뀌므로 규모에 대해서는 주장하지 않음
- **(student, 데이터, 학습)**
    - student는 Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct, OLMo-2-1B-Instruct이며 ([Qwen et al., 2025](https://arxiv.org/abs/2412.15115); [Meta AI, 2024](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/); [Team OLMo et al., 2025](https://arxiv.org/abs/2501.00656)), 긴 풀이가 해로워지는 규모이고 ([Li et al., 2025](https://arxiv.org/abs/2502.12143)), 결과가 한 계열의 버릇이나 오염이 아니도록 세 계열에서 고름 ([Wu et al., 2026b](https://arxiv.org/abs/2507.10532)). 쌍과 student의 각 조합을 칸이라 부름
    - 모든 teacher가 같은 MATH 학습 문제 4,000개를 풀며 ([Hendrycks et al., 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html)), 모든 $\lambda$의 teacher가 맞게, 끝까지 푼 문제로만 학습함
    - 각 student에는 LoRA adapter($r=16$, $\alpha=32$)를 붙이며, LoRA는 효율이 아니라 측정을 위해 씀. 업데이트가 작은 행렬 한 쌍에 모여 학습끼리 비교할 수 있기 때문임
    - 각 칸과 $\lambda$마다 학습 집합 전체로 seed 셋을 학습하고, 데이터만의 효과를 재기 위해 겹치지 않는 두 반쪽 각각으로도 학습함 (부록 A)
- **(평가)**
    - 각 seed의 adapter가 MATH-500의 앞 200문제를 2회씩 풀며 ([Lightman et al., 2024](https://openreview.net/forum?id=v8L0pN6EOi)), seed 간 평균 ± 표준편차를 보고함. 잘림이 오답으로 섞이지 않도록 pass@1 정답률을 전체 응답과 끝까지 쓴 응답만으로 각각 잼
    - 생각은 teacher에서는 think rate(`</think>`를 포함한 응답의 비율)로 ([Wu et al., 2026a](https://arxiv.org/abs/2510.10977)), student와 teacher 풀이에서는 반성 표현(wait, recheck, alternatively 등)의 빈도로 잼 ([Yeo et al., 2025](https://arxiv.org/abs/2502.03373); [Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393))
    - 반성 표현은 대리 지표이므로 토큰 수와 생성 한도에 걸린 비율을 함께 적고 ([Chen et al., 2026](https://arxiv.org/abs/2601.06002); [Goyal et al., 2026](https://arxiv.org/abs/2608.04160)), Llama student를 공개 SAE로 읽음. 이때 $\lambda=1$보다 $\lambda=0$ teacher 풀이에서 훨씬 강하게 켜지고, 반성 표현을 끼우거나 지우는 두 반증 검사를 통과한 생각 feature만 남김 ([Ma et al., 2026a](https://arxiv.org/abs/2601.05679))
    - GSM8K ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168))와 GPQA-Diamond ([Rein et al., 2024](https://openreview.net/forum?id=Ti67584b98))에서도 평가하고, 학습 전 student도 평가함 (부록 A)
## Subspace Analysis
- **(겹침)**
    - adapter의 업데이트 방향은 각 층 업데이트의 앞 $r$개 좌특이벡터 $U$이고(식 1), 두 adapter가 같은 방향을 쓰는 정도는 겹침 $\phi$로 봄(식 2)
    - $\phi$는 같은 방향이면 1, 출력 차원이 $d$인 무작위 부분공간끼리는 $r/d$이며 ([Hu et al., 2022](https://arxiv.org/abs/2106.09685)), 층과 모듈에 걸쳐 $\lVert \Delta W \rVert_F$로 가중 평균함
    $$
    \Delta W = \tfrac{\alpha}{r} BA = U\Sigma Q^\top, \qquad U = [\,u_1, \dots, u_r\,] \tag{1}
    $$
    $$
    \phi(\Delta W_i, \Delta W_j) = \lVert U_i^\top U_j \rVert_F^2 / r \tag{2}
    $$
- **(비교)**
    - $R$은 teacher가 다른 adapter끼리의 겹침을 데이터만 다른 adapter끼리의 겹침으로 나눈 값임(식 3). 데이터 반쪽 $a$, $b$와 seed가 분자와 분모에 똑같이 들어가 teacher 효과만 남으며, 선행 연구는 겹침을 우연 수준이나 데이터 차이와 비교하지 않았음 ([Dieudonné et al., 2026](https://arxiv.org/abs/2606.28117))
    $$
    R(\lambda) = \dfrac{\phi(\Delta W_a^{0}, \Delta W_b^{\lambda}) + \phi(\Delta W_b^{0}, \Delta W_a^{\lambda})}{2\,\phi(\Delta W_a^{\lambda}, \Delta W_b^{\lambda})} \tag{3}
    $$
    - $R=1$이면 teacher 효과가 데이터 차이 수준이고, 0에 가까우면 업데이트 방향이 크게 바뀐 것임
    - seed만 다른 adapter끼리의 겹침은 기준선으로 쓰지 않음. 같은 seed로 학습한 adapter끼리는 LoRA 초기화를 공유하기 때문임 (부록 B)
    - 전환 $\lambda$를 미리 한 점으로 고르면 기준에 따라 답이 갈리고 후행 선택이 되므로, 대신 $\lambda$ 구간마다 $R$의 낙차와 teacher의 think rate 또는 반성 표현의 낙차가 상관하는지를 순열 검정으로 봄 (부록 A)
- **(추출)**
    - teacher의 think rate가 처음으로 절반 넘게 떨어지는 구간을 전환 구간으로 삼아, 그 전의 $n$개 adapter 중 절반 넘게 함께 쓰는 방향 $V$만 남기고(식 4) ([Gerasimov et al., 2026](https://arxiv.org/abs/2606.12138)), $V$를 공유 구역이라 부름
    $$
    P = \tfrac{1}{n}\textstyle\sum_{t} U_t U_t^\top,\; V = \{v : Pv = \mu v,\, \mu > 0.5\} \tag{4}
    $$
    - adapter가 $V$를 덮는 비율은 $\lVert U_t^\top V \rVert_F^2 / k$($k = \dim V$)로 재어 우연 수준 $r/d$와 비교하며, 채점과 편집에는 늘 그 adapter를 빼고 만든 $V$를 씀
    - $V$가 보간의 산물이 아닌지 보기 위해, seed가 다른 $\lambda=0$ adapter 두 개로도 구역을 만듦(이하 seed 2개로 찾은 구역) (부록 A)
## Subspace Interventions
- **(제거)**
    $$
    \Delta W' = (I - VV^\top)\,\Delta W \tag{5}
    $$
    - 공유 구역이 생각하며 푸는 데 필요한지 보기 위해, $\lambda=0$ adapter에서 $V$를 뺌(식 5) ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717))
- **(이식)**
    $$
    \Delta W'(\lambda{=}1) = \Delta W(\lambda{=}1) + \beta\, VV^\top \Delta W(\lambda{=}0) \tag{6}
    $$
    - 공유 구역으로 충분한지 보기 위해, 짧게 답하는 $\lambda=1$ adapter에 $V$의 $\lambda=0$ 성분만 $\beta=1$과 2로 더함(식 6)
- **(증폭)**
    $$
    \Delta W' = \lVert \Delta W \rVert_F \, \dfrac{\Delta W + \gamma\, VV^\top \Delta W}{\lVert \Delta W + \gamma\, VV^\top \Delta W \rVert_F} \tag{7}
    $$
    - 공유 구역이 다이얼인지 보기 위해, 노름을 고정한 채 $V$의 비중만 $\gamma=1$과 4로 키우거나 업데이트 전체를 $V$로 옮김(식 7) ([Ma et al., 2025a](https://aclanthology.org/2025.acl-long.300/); [Galichin et al., 2026](https://arxiv.org/abs/2503.18878); [Sheng et al., 2025](https://arxiv.org/abs/2506.08390))
- **(대조)**
    - 제거의 대조는 같은 에너지를 $V$와 직교하는 쪽의 에너지가 가장 큰 방향에서 빼거나 ([Lion et al., 2025](https://arxiv.org/abs/2506.03133); [Fu et al., 2026](https://arxiv.org/abs/2601.07411)) 가장 큰 특잇값 성분에서 빼거나 ([Lee et al., 2025](https://arxiv.org/abs/2510.06750)), 업데이트 전체를 줄이는 것임. 이식의 대조는 같은 에너지를 $V$ 밖에서 가져오거나 $\lambda=0$ 업데이트 전체에서 가져와 더하는 것임
    - 원래 adapter를 같은 길이에서 끊는 조건 ([Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393))과 같은 편집을 $\lambda=1$ adapter에 하는 조건도 봄
    - seed 2개로 찾은 구역은 생각을 끝냈다는 문장을 넣는 방법 ([Ma et al., 2025b](https://arxiv.org/abs/2504.09858)), 짧은 풀이 adapter와 병합하는 방법 ([Wu et al., 2025](https://arxiv.org/abs/2503.20641))과도 비교함 (부록 A)
