<!-- Notion에서 동기화: 2026-10-04 -->
**제목: Where Do Distilled Students Store Thinking?**

# Abstract
- **(배경)**
    - 작은 student는 추론 모델의 긴 풀이를 배우면 오히려 성능이 떨어져, 최근에는 추론 모델과 짧게 답하는 지시 모델을 비율 $\lambda$로 섞은 teacher로 풀이를 짧게 줄임. 이렇게 섞은 teacher는 어느 $\lambda$에서 생각하며 푸는 것을 통째로 그만둠
- **(물음)**
    - 그 변화가 student의 가중치에 무엇을 남기는지는 알려져 있지 않음
- **(방법)**
    - teacher 3쌍과 student 3종에서 $\lambda$별로 학습한 LoRA(low-rank adaptation) adapter를 비교하고, 찾은 구역을 편집해 출력과 SAE(sparse autoencoder) feature를 함께 읽음
- **(발견)**
    - teacher가 생각을 그만두는 곳에서 student의 adapter는 가중치를 고치는 방향을 크게 바꾸며, 그 전까지는 모든 adapter가 같은 작은 구역을 함께 씀
    - 이 구역은 생각을 켜고 끄는 스위치처럼 작동함. 빼면 다른 곳에서 같은 에너지를 뺄 때보다 생각이 더 꺼지고, 키워도 생각이 늘지 않으며, 짧게 답하던 student에 넣으면 생각은 시작하지만 풀이를 끝맺는 일은 드묾
    - 이 구역을 빼면 MATH 밖에서도 생각이 꺼지며, 이 구역의 상당 부분은 보간 없이 증류한 adapter에서도 다시 나타남
- **(의의)**
    - 따라서 작은 student로의 추론 증류는 teacher가 생각하는 방식을 가중치의 작고 따로 떨어진 구역에 저장하는 것으로 보이며, 그 구역은 풀이를 나르지 않은 채 생각을 켬
