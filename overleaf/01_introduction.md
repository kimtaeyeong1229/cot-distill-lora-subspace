<!-- Notion에서 동기화: 2026-09-28 -->
# Introduction

- **(배경)**
    - 큰 추론 모델이 긴 풀이로 어려운 문제를 풀게 되면서, 그 풀이로 작은 모델을 학습시키는 증류가 널리 쓰임 ([Guo et al., 2025](https://arxiv.org/abs/2501.12948); [Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393)). 그런데 작은 student는 긴 풀이에서 오히려 손해를 보아 ([Li et al., 2025](https://arxiv.org/abs/2502.12143); [Luo et al., 2025](https://arxiv.org/abs/2506.07712)), 최근에는 추론 모델과 지시 모델을 $\lambda$로 섞은 teacher로 풀이 스타일을 연속으로 바꿔 가며 student에 맞는 지점을 고름 ([Lan et al., 2026](https://arxiv.org/abs/2608.29623))
- **(문제)**
    - 하지만 $\lambda$를 움직일 때 teacher의 무엇이 바뀌고 그것이 student에 무엇을 남기는지는 알려져 있지 않음. 그래서 최적 $\lambda$가 벤치마크마다 왜 갈리는지 설명하지 못하고, teacher는 정답률을 쓸어보며 고르는 수밖에 없음
- **(선행 접근)**
    - 보간한 teacher를 들여다본 연구들은 어느 $\lambda$에서 teacher가 사고 과정을 통째로 버린다고 보고하지만 ([Wu et al., 2026a](https://arxiv.org/abs/2510.10977); [Lan et al., 2025](https://arxiv.org/abs/2509.22034)), 위치는 연구마다 엇갈리고 전이 자체를 부정하는 보고도 있음 ([Yang et al., 2026](https://arxiv.org/abs/2601.21414)). 이들은 모두 출력의 정답률과 길이로만 전이를 읽는데, 긴 CoT는 형식과 길이가 함께 움직여 출력만으로는 형식이 바뀐 것인지 길이만 바뀐 것인지 가릴 수 없음
- **(물음)**
    - 그래서 두 가지를 물음. teacher가 풀이 과정을 버릴 때 student의 가중치에서는 무엇이 달라지는가, 그리고 달라진 그 자리가 실제로 추론 형식을 담고 있는가
- **(설계)**
    - 전체 미세조정은 바뀐 곳이 넓게 흩어져 읽기 어려우므로, 학습을 LoRA로 묶어 그 업데이트를 측정 장치로 삼음. teacher만 $\lambda$로 움직여 아홉 칸에서 학습한 adapter들을 같은 자리에 겹쳐 놓고, 찾은 구역을 지우고 옮겨 심어 그 자리가 무엇을 나르는지 가름
- **(기여)**
    - teacher가 풀이 과정을 버리는 $\lambda$와 student의 LoRA가 방향을 갈아타는 $\lambda$가 쌍마다 같은 자리에 놓임
    - 풀이 과정을 쓰는 teacher에게 배운 adapter들이 16개 방향 중 5–8개를 늘 같은 자리에서 쓰고, teacher가 풀이를 버린 뒤 배운 adapter에서는 그 자리가 비어 있음
    - 그 자리를 지우면 반성 표현이 사라지고 키워도 늘지 않지만, 그 자리만 짧게 답하던 adapter에 옮겨 심었을 때 형식이 돌아오는지는 teacher 쌍에 따라 갈림
