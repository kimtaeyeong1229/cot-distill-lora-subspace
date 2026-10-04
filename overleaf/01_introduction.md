<!-- Notion에서 동기화: 2026-10-04 -->
# Introduction
[그림: figures/fig1_overview.pdf]
**그림 1.** 분석의 개요임. (a) teacher는 추론 모델($\lambda=0$)과 지시 모델($\lambda=1$)을 섞은 것으로, 어느 $\lambda$를 넘으면 생각을 그만둠(파선). 막대는 응답이고, 빗금 친 부분이 생각임. (b) 그 지점 전에 학습한 adapter들은 16개 업데이트 방향 중 5–8개(보라)를 함께 쓰고, 그 뒤에 학습한 adapter들은 이 방향을 대부분 떠남. (c) 이 공유 구역을 $\lambda=0$ adapter에서 빼면 답만 남고, $\lambda=1$ adapter에 넣으면 생각이 시작되지만 끝맺는 일은 드물며, 키우면 응답이 전과 같은 길이로 유지됨. 옅은 막대는 각 편집 전의 응답이며, 길이는 예시임. 이 구역은 생각의 다이얼보다 스위치에 가까움
- **(배경)**
    - 큰 추론 모델의 긴 풀이를 작은 모델로 증류하는 것이 널리 쓰이지만 ([Guo et al., 2025](https://arxiv.org/abs/2501.12948); [Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393)), 작은 student는 긴 풀이를 배우면 오히려 성능이 떨어짐 ([Li et al., 2025](https://arxiv.org/abs/2502.12143); [Luo et al., 2025](https://arxiv.org/abs/2506.07712))
    - 그래서 최근에는 추론 모델과 짧게 답하는 지시 모델을 비율 $\lambda$로 섞은 teacher로 풀이 길이를 조절함 ([Lan et al., 2026](https://arxiv.org/abs/2608.29623))
- **(문제)**
    - 이렇게 섞은 teacher는 어느 $\lambda$에서 생각하며 푸는 것을 통째로 그만두지만 ([Wu et al., 2026a](https://arxiv.org/abs/2510.10977); [Lan et al., 2025](https://arxiv.org/abs/2509.22034)), 그때 student의 가중치에 무엇이 남는지는 알려져 있지 않음
- **(선행 접근)**
    - 보간 연구들은 teacher의 출력만 봤고, 그런 전환이 없다는 보고도 있음 ([Yang et al., 2026](https://arxiv.org/abs/2601.21414)). 증류 연구는 $\lambda$를 student의 정답률로 고르며, 긴 풀이가 student를 더 집중된 방향으로 학습시킨다는 것만 보였음 ([Lan et al., 2026](https://arxiv.org/abs/2608.29623))
    - 증류된 모델의 가중치나 내부를 읽는 연구는 teacher를 바꿔 가며 student를 비교하지 않고, 증류된 모델 하나를 원래 모델과 비교함 ([Ward et al., 2025](https://arxiv.org/abs/2511.06739); [Hu et al., 2026b](https://arxiv.org/abs/2602.20904); [Venhoff et al., 2026](https://arxiv.org/abs/2510.07364))
    - 게다가 긴 풀이에서는 생각하는 방식과 길이가 함께 움직여, 출력만으로는 둘 중 무엇이 바뀐 것인지 가릴 수 없음
- **(물음)**
    - 그래서 teacher가 생각을 그만둘 때 student의 가중치에서 무엇이 달라지는가, 그리고 달라진 그 부분이 생각하는 방식을 담고 있는가를 물음
- **(설계)**
    - 전체 미세조정은 바뀐 곳이 넓게 흩어지므로, 학습을 LoRA(low-rank adaptation)로 묶어 adapter를 측정 도구로 씀
    - teacher 3쌍과 student 3종에서 $\lambda$만 바꿔 학습한 adapter들을 데이터만 다른 adapter들과 비교해, teacher 효과만 남김
    - 생각하며 푸는지는 teacher에서는 think rate(`</think>` 태그가 있는 응답의 비율)로, student에서는 반성 표현으로 잼
    - 그다음 찾은 구역을 빼거나 넣거나 키워서 출력과 SAE(sparse autoencoder)의 내부 신호를 읽음 (그림 1)
- **(기여)**
    - teacher가 생각을 그만두는 $\lambda$에서(think rate로 재든 반성 표현으로 재든) student의 adapter는 세 쌍 모두 가중치를 고치는 방향을 크게 바꾸며, 쌍마다 자기 teacher의 $\lambda$를 따름. 그 전까지의 adapter들은 아홉 칸 모두에서 16개 방향 중 5–8개를 함께 씀 (4장)
    - 이 공유 구역은 생각을 켜고 끄는 스위치처럼 작동하지만, 풀이를 끝까지 나르지는 않음 (5장)
        - 빼면 반성 표현이 원래 수준의 0.2–10%로, 답 길이가 3.7–5.4배 줄어, 다른 곳에서 같은 에너지를 뺄 때보다 더 꺼지며, 같은 길이로 끊은 것보다 정답률이 높음
        - 노름을 고정하고 키워도 생각이 늘지 않고, 짧게 답하던 student에 넣으면 생각이 시작되지만 끝맺는 일은 드물며, SAE 신호도 대부분 같은 결론을 냄
    - 이 구역은 학습한 문제 밖과 섞은 teacher가 없는 증류에서도 나타남 (6장)
        - 빼면 GSM8K와 GPQA-Diamond에서도 생각이 꺼지며, 보간 없이 학습한 adapter들이 이 구역의 상당 부분을 되찾음
