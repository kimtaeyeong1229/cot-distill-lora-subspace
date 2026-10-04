<!-- Notion에서 동기화: 2026-10-04 -->
# Related Work
## Chain-of-Thought Distillation to Small Students
- **(관행)**
    - 증류는 강한 teacher의 출력으로 작은 student를 학습시키는 방법이며 ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)), 추론 모델이 쓴 긴 풀이와 R1에서 증류한 작은 모델들이 주된 감독이자 후속 연구의 출발점이 되었음 ([Guo et al., 2025](https://arxiv.org/abs/2501.12948))
- **(흐름)**
    - 그런데 긴 풀이는 작은 student에게 해로울 수 있음
        - 작은 student는 강한 teacher의 짧은 풀이에서 더 잘 배우고 ([Li et al., 2025](https://arxiv.org/abs/2502.12143)), 긴 풀이를 적은 양만 학습해도 성능이 크게 떨어짐 ([Luo et al., 2025](https://arxiv.org/abs/2506.07712))
        - 가장 강한 teacher가 언제나 최선은 아니므로, 원인이 student 쪽에만 있는 것도 아님 ([Hu et al., 2026a](https://arxiv.org/abs/2605.26872))
    - 그래서 긴 풀이와 짧은 풀이 사이를 메우는 연구가 이어짐
        - 길이와 난이도에 걸친 커리큘럼을 쓰거나 ([Jiang et al., 2025](https://aclanthology.org/2025.emnlp-main.376/)), 한 teacher의 추론 모드를 켜고 꺼서 같은 프롬프트로 두 벌의 데이터를 만들거나 ([Boizard et al., 2026](https://arxiv.org/abs/2509.22193)), 섞은 teacher를 씀
        - 섞은 teacher로 증류한 student는 중간 $\lambda$에서 가장 잘하고, 풀이 길이가 $\lambda$를 따라 매끄럽게 변하며, 긴 풀이가 student를 더 집중된 방향으로 학습시킴 ([Lan et al., 2026](https://arxiv.org/abs/2608.29623)). teacher 자체는 풀이 스타일이 연속으로 변하다가 생각하는 것을 통째로 그만둠 ([Wu et al., 2026a](https://arxiv.org/abs/2510.10977); [Lan et al., 2025](https://arxiv.org/abs/2509.22034))
        - 섞는 비율에 따라 행동이 급격히 나타나는 현상은 전문가 혼합 모델을 조립할 때도 보고되었고 ([Klagges et al., 2025](https://arxiv.org/abs/2506.14794)), 그런 전환이 없다는 보고도 있으며 ([Yang et al., 2026](https://arxiv.org/abs/2601.21414)), LoRA 증류와 보간으로 함께 다듬은 checkpoint는 정답률만 봄 ([Saini & Pandey, 2026](https://arxiv.org/abs/2606.01080))
    - 다만 이 흐름은 teacher를 student의 정답률로 고르거나 teacher를 출력으로만 평가해, teacher가 생각을 그만둘 때 student의 가중치에 무엇이 남는지는 다루지 않음
## Reading and Editing Model Internals
- **(관행)**
    - LoRA 업데이트는 LoRA 원 논문부터 부분공간으로 비교되었고 ([Hu et al., 2022](https://arxiv.org/abs/2106.09685)), SAE의 개별 방향은 학습마다 흔들려도 그 부분공간은 재현됨 ([Gerasimov et al., 2026](https://arxiv.org/abs/2606.12138))
- **(흐름)**
    - 학습이 끝난 가중치를 열어 무엇이 담겼는지 보는 연구가 이어짐
        - R1의 풀이로 rank 1 LoRA만 학습해도 전체 미세조정과의 격차가 73–90% 회복되고, 그 adapter를 활성으로 읽음 ([Ward et al., 2025](https://arxiv.org/abs/2511.06739)). teacher의 추론 궤적은 저차원 부분공간에 가까움 ([Solgi et al., 2026](https://arxiv.org/abs/2606.05315))
        - 추론 모델과 지시 모델의 가중치 차이에서 추론 능력은 지시 모델의 주된 특이 방향 밖에 놓임 ([Ma et al., 2026b](https://arxiv.org/abs/2609.37976))
        - 초기화를 공유한 teacher와 student 사이에서 증류가 옮기는 것은 두 모델 표현의 정렬을 따름 ([Chauhan & Shah, 2026](https://arxiv.org/abs/2607.04432))
        - 서로 다른 과제의 adapter들이 크게 겹쳐 앞선 adapter가 뒤 과제를 대신하기도 하지만, 그 겹침을 우연 수준과 비교하지는 않음 ([Dieudonné et al., 2026](https://arxiv.org/abs/2606.28117))
    - 고친 방향을 스위치로 다루기도 함
        - 거부 방향을 빼면 거부하지 못하고 더하면 무해한 요청도 거부하며 ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)), 학습한 저랭크 파라미터 편집으로 특정 능력을 지우기도 함 ([Fu et al., 2026](https://arxiv.org/abs/2601.07411))
        - 추론 모델이 원래 모델에서 바뀐 부분의 가장 큰 저랭크 성분을 빼면 과하게 생각하는 것이 줄고 (Gold-Switch; [Lee et al., 2025](https://arxiv.org/abs/2510.06750)), 활성 방향 하나로 생각 모드와 비생각 모드를 오가기도 함 ([Liu & Honavar, 2026](https://arxiv.org/abs/2609.28682))
    - 고친 방향을 생각의 양을 정하는 다이얼로 다루기도 함
        - LoRA 방향 하나의 크기로 추론 길이를 조절하고 ([Ma et al., 2025a](https://aclanthology.org/2025.acl-long.300/)), 몇몇 attention head에서 짧게 추론하는 방향을 빼면 추론이 길어짐 ([Sun et al., 2025](https://aclanthology.org/2025.emnlp-main.861/))
        - 반성 SAE feature나 활성 방향을 키우면 풀이가 길어지고 ([Galichin et al., 2026](https://arxiv.org/abs/2503.18878); [Sheng et al., 2025](https://arxiv.org/abs/2506.08390)), 방향 하나의 세기로 생각하는 길이를 정하기도 함 ([Jayabahu & Adeleke, 2026](https://arxiv.org/abs/2608.28859))
    - 모델 내부의 신호를 읽어 생각하는 방식을 찾기도 함
        - 증류 모델에만 있는 반성과 검산 feature로 과하게 생각하거나 간결하게 생각하도록 조절함 ([Baek & Tegmark, 2025](https://arxiv.org/abs/2503.03730))
        - 증류된 모델이 바뀐 부분에서 망설임 표현에 필요충분하고 빼면 응답이 짧아지는 소수 성분을 찾고 ([Hu et al., 2026b](https://arxiv.org/abs/2602.20904)), RL은 원래 있던 추론 기제를 언제 쓸지 가르치는 반면 지도 증류는 새 기제를 심음을 보임 ([Venhoff et al., 2026](https://arxiv.org/abs/2510.07364))
        - 다만 긴 풀이와 짧은 풀이를 대조해 찾은 reasoning feature의 상당수는 생각이 아니라 반성 표현에 반응함 ([Ma et al., 2026a](https://arxiv.org/abs/2601.05679))
    - 다만 이 연구들은 teacher를 고정하거나 아예 두지 않고, 증류된 모델 하나를 원래 모델과 비교하며, 방향을 업데이트의 크기나 길이 대조로 찾음. 또 찾은 방향을 스위치나 다이얼 중 하나로 다뤄, 같은 구역이 빼기와 넣기와 키우기에 서로 다르게 반응하는지는 묻지 않음
- **(본 연구)**
    - 본 연구는 $\lambda$를 따라 움직인 teacher로 증류한 student들을 비교해 teacher가 생각을 그만둘 때 생기는 구역을 찾고, 그 구역을 빼거나 넣거나 키워서 생각과의 관계를 확인함
