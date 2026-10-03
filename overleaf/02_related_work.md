<!-- Notion에서 동기화: 2026-10-03 -->
# Related Work
## Chain-of-Thought Distillation to Small Students
- **(관행)**
    - 증류는 강한 teacher의 출력으로 작은 student를 학습시키는 방법이며 ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)), 추론 모델이 나온 뒤로는 그들이 쓴 긴 풀이가 주된 감독이 되었고 R1이 증류해 공개한 작은 모델들이 후속 연구의 출발점이 됨 ([Guo et al., 2025](https://arxiv.org/abs/2501.12948))
- **(흐름)**
    - 그런데 긴 풀이가 작은 student에게는 오히려 해롭다는 보고가 쌓임
        - 작은 student는 강한 teacher의 긴 풀이보다 짧은 풀이에서 더 잘 배우고 ([Li et al., 2025](https://arxiv.org/abs/2502.12143)), 긴 풀이는 적은 양만 학습시켜도 성능이 크게 떨어짐 ([Luo et al., 2025](https://arxiv.org/abs/2506.07712))
        - 여러 teacher와 student를 교차해 붙이면 가장 강한 teacher가 언제나 최선은 아니므로, 원인이 student 쪽에만 있는 것도 아님 ([Hu et al., 2026a](https://arxiv.org/abs/2605.26872))
    - 그래서 긴 풀이와 짧은 풀이 중 하나만 고르는 대신, 그 사이를 메우는 쪽으로 흐름이 옮겨감
        - 짧은 풀이와 긴 풀이를 섞거나 난이도 순서를 바꾸는 커리큘럼처럼, 만들어진 데이터의 구성과 순서를 조정함 ([Jiang et al., 2025](https://aclanthology.org/2025.emnlp-main.376/))
        - 한 teacher의 추론 모드만 켜고 꺼서 같은 프롬프트로 두 벌의 데이터를 만듦 ([Boizard et al., 2026](https://arxiv.org/abs/2509.22193))
        - 추론 모델과 지시 모델을 가중치에서 섞은 teacher로 증류하면 양 끝보다 중간 $\lambda$에서 student가 가장 잘하며, student의 풀이 길이는 $\lambda$를 따라 매끄럽게 변하고, 긴 풀이가 짧은 풀이보다 더 집중된 방향으로 student를 학습시킴 ([Lan et al., 2026](https://arxiv.org/abs/2608.29623))
            - 이렇게 섞은 teacher 자체는 $\lambda$에 따라 풀이 스타일이 연속으로 변하다가 어느 $\lambda$에서 생각하며 푸는 것을 통째로 그만둔다고 보고됨 ([Wu et al., 2026a](https://arxiv.org/abs/2510.10977); [Lan et al., 2025](https://arxiv.org/abs/2509.22034)). 섞는 비율에 따라 일부 행동이 급격히 나타나는 현상은 전문가 혼합 모델을 조립할 때도 보고되었으나 ([Klagges et al., 2025](https://arxiv.org/abs/2506.14794)), 그런 전환이 없다는 보고도 있음 ([Yang et al., 2026](https://arxiv.org/abs/2601.21414))
        - 추론 모델과 지시 모델 checkpoint를 LoRA 증류와 가중치 보간으로 함께 다듬기도 하지만, 정답률만 봄 ([Saini & Pandey, 2026](https://arxiv.org/abs/2606.01080))
    - 다만 이 흐름은 teacher를 student의 정답률로 고르거나 teacher 자신의 출력으로만 평가해, teacher가 생각을 그만둘 때 student의 가중치에 무엇이 남는지는 다루지 않음
## Reading and Editing Model Internals
- **(관행)**
    - LoRA 업데이트를 부분공간으로 보고 그 겹침으로 서로 다른 학습을 비교하는 방법은 LoRA 원 논문부터 쓰였고 ([Hu et al., 2022](https://arxiv.org/abs/2106.09685)), 개별 방향은 학습마다 흔들려도 부분공간은 재현된다는 점이 SAE에서 보고됨 ([Gerasimov et al., 2026](https://arxiv.org/abs/2606.12138))
- **(흐름)**
    - 학습이 끝난 가중치를 열어 무엇이 담겼는지 보는 연구가 이어짐
        - R1의 풀이로 rank 1 LoRA만 학습시켜도 전체 미세조정과의 성능 격차가 73–90% 회복되고, 그 adapter의 방향을 활성으로 읽어 해석함 ([Ward et al., 2025](https://arxiv.org/abs/2511.06739)). teacher의 추론 궤적 자체가 저차원 부분공간에 가깝다는 보고도 있음 ([Solgi et al., 2026](https://arxiv.org/abs/2606.05315))
        - 추론 모델과 지시 모델의 가중치 차이를 스펙트럼으로 나누면, 추론 능력이 지시 모델의 주된 특이 방향 밖에 놓인 성분에 몰려 있음 ([Ma et al., 2026b](https://arxiv.org/abs/2609.37976))
        - 초기화를 공유한 teacher와 student 사이에서는, 증류로 넘어가는 성질이 내부 표현의 정렬을 따라 열리고 닫힘 ([Chauhan & Shah, 2026](https://arxiv.org/abs/2607.04432))
        - 서로 다른 과제로 학습한 adapter들의 부분공간이 상당히 겹쳐, 앞선 adapter가 뒤 과제를 대신하기도 함. 다만 그 겹침을 우연 수준과 비교하지는 않음 ([Dieudonné et al., 2026](https://arxiv.org/abs/2606.28117))
    - 찾은 방향을 고쳐 행동을 바꾸는 연구는 그 방향을 켜고 끄는 스위치로 다루기도 함
        - 거부를 매개하는 방향 하나를 빼면 거부하지 못하고 더하면 무해한 요청도 거부하며 ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)), 같은 투영 제거로 특정 능력만 정밀하게 지우기도 함 ([Fu et al., 2026](https://arxiv.org/abs/2601.07411))
        - 생각에서도, 추론 모델이 원래 모델에서 바뀐 부분을 저랭크로 근사해 그 가장 큰 성분을 빼면 과하게 생각하는 것이 줄고 ([Lee et al., 2025](https://arxiv.org/abs/2510.06750)), 생각 모드와 비생각 모드를 가르는 활성 방향 하나로 양쪽을 오가며 그 영향을 재기도 함 ([Liu & Honavar, 2026](https://arxiv.org/abs/2609.28682))
    - 생각의 양을 그 방향의 세기로 조절하는 다이얼로 다루기도 함
        - LoRA 방향 하나의 크기로 추론 길이를 조절하도록 학습하거나 ([Ma et al., 2025a](https://aclanthology.org/2025.acl-long.300/)), 몇몇 attention head에서 짧게 추론하는 방향을 빼 추론을 늘림 ([Sun et al., 2025](https://aclanthology.org/2025.emnlp-main.861/))
        - 반성 SAE feature나 활성 방향을 키우면 풀이가 길어지고 ([Galichin et al., 2026](https://arxiv.org/abs/2503.18878); [Sheng et al., 2025](https://arxiv.org/abs/2506.08390)), 방향 하나의 세기로 생각하는 길이를 정하기도 함 ([Jayabahu & Adeleke, 2026](https://arxiv.org/abs/2608.28859))
    - 모델 내부의 신호를 SAE로 읽어 생각하는 방식을 찾기도 함
        - 증류 모델에만 있는 반성·검산 feature로 과하게 생각하거나 간결하게 생각하도록 조절함 ([Baek & Tegmark, 2025](https://arxiv.org/abs/2503.03730))
        - 증류된 추론 모델과 원래 모델의 차이를 해석 가능한 성분으로 근사하면, 망설임 표현을 내는 소수 feature가 그 표현에 필요충분하며 빼면 응답이 짧아짐 ([Hu et al., 2026b](https://arxiv.org/abs/2602.20904)). 같은 방식으로 보면 RL은 원래 있던 추론 기제를 언제 쓸지 가르치는 반면, 증류(SFT)는 새 기제를 심음 ([Venhoff et al., 2026](https://arxiv.org/abs/2510.07364))
        - 다만 긴 풀이와 짧은 풀이를 대조해 고른 reasoning feature의 상당수는 생각이 아니라 반성 단어 자체에 반응한다는 반박도 있음 ([Ma et al., 2026a](https://arxiv.org/abs/2601.05679))
    - 다만 이 연구들은 teacher가 하나로 고정되었거나 아예 없는 설정에서 쓰였고(증류 모델을 다룬 연구도 증류된 모델 하나를 원래 모델과 비교함), 방향도 업데이트의 크기나 긴/짧은 풀이의 대조로 찾음. 또 찾은 방향을 스위치나 다이얼 중 하나로만 다뤄, 같은 구역을 빼고 넣고 키울 때 서로 다르게 반응하는지는 보지 않음
- **(본 연구)**
    - 본 연구는 teacher를 $\lambda$로 움직여 증류한 student들의 LoRA를 나란히 비교해 teacher가 생각을 그만둘 때 student 가중치에 생기는 공유 구역을 찾고, 그 구역을 빼거나 넣거나 키워서 생각과의 관계를 확인함
