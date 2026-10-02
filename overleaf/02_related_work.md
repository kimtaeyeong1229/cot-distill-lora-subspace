<!-- Notion에서 동기화: 2026-10-02 -->
# Related Work

## Chain-of-Thought Distillation to Small Students

- **(관행)**
    - 증류는 강한 teacher의 출력으로 작은 student를 학습시키는 방법이며 ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)), 추론 모델이 나온 뒤로는 그들이 쓴 긴 풀이가 주된 감독이 되었고 R1이 증류해 공개한 작은 모델들이 후속 연구의 출발점이 됨 ([Guo et al., 2025](https://arxiv.org/abs/2501.12948))
- **(흐름)**
    - 그런데 긴 풀이가 작은 student에게는 오히려 해롭다는 보고가 쌓임
        - 작은 student는 강한 teacher의 긴 CoT보다 짧은 CoT에서 더 잘 배우고 ([Li et al., 2025](https://arxiv.org/abs/2502.12143)), 긴 CoT는 적은 양만 학습시켜도 성능이 크게 떨어짐 ([Luo et al., 2025](https://arxiv.org/abs/2506.07712))
        - 여러 teacher와 student를 교차해 붙이면 가장 강한 teacher가 언제나 최선은 아니므로, 원인이 student 쪽에만 있는 것도 아님 ([Hu et al., 2026](https://arxiv.org/abs/2605.26872))
    - 그래서 긴 풀이와 짧은 풀이 중 하나만 고르는 대신, 그 사이를 메우는 쪽으로 흐름이 옮겨감
        - 짧은 풀이와 긴 풀이를 섞거나 난이도 순서를 바꾸는 커리큘럼처럼, 만들어진 데이터의 구성과 순서를 조정함 ([Jiang et al., 2025](https://aclanthology.org/2025.emnlp-main.376/))
        - 한 teacher의 추론 모드만 켜고 꺼서 같은 프롬프트로 두 벌의 데이터를 만듦 ([Boizard et al., 2026](https://arxiv.org/abs/2509.22193))
        - 한편 증류와 별개로, 추론 모델과 지시 모델을 가중치에서 섞는 보간 연구들은 섞인 모델의 풀이 스타일이 연속으로 변하며 어느 $\lambda$에서 생각하며 푸는 것을 통째로 그만둔다고 보고함 ([Wu et al., 2026a](https://arxiv.org/abs/2510.10977); [Lan et al., 2025](https://arxiv.org/abs/2509.22034)). 다만 그런 전환이 없다는 보고도 있음 ([Yang et al., 2026](https://arxiv.org/abs/2601.21414))
        - 이 섞인 모델을 teacher로 써서 증류하면, 양 끝보다 중간 $\lambda$에서 student가 가장 잘함 ([Lan et al., 2026](https://arxiv.org/abs/2608.29623))
    - 다만 이 흐름은 teacher를 student의 정답률로 고르거나 teacher 자신의 출력으로만 평가해, teacher가 student의 가중치에 무엇을 남기는지는 다루지 않음

## Reading and Editing Weight Updates

- **(관행)**
    - LoRA 업데이트를 부분공간으로 보고 그 겹침으로 서로 다른 학습을 비교하는 방법은 LoRA 원 논문부터 쓰였고 ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)), 개별 방향은 학습마다 흔들려도 부분공간은 재현된다는 점이 SAE에서 보고됨 ([Gerasimov et al., 2026](https://arxiv.org/abs/2606.12138))
- **(흐름)**
    - 학습이 끝난 가중치를 열어 무엇이 담겼는지 보는 연구가 이어짐
        - R1의 풀이로 rank 1 LoRA만 학습시켜도 전체 미세조정과의 성능 격차가 73–90% 회복되고, 그 adapter의 방향을 활성으로 읽어 해석함 ([Ward et al., 2025](https://arxiv.org/abs/2511.06739))
        - adapter가 담는 양을 비트로 직접 재면, 파라미터 수보다 그 파라미터가 attention에 있는지 MLP에 있는지가 더 크게 좌우함 ([Tan et al., 2026](https://arxiv.org/abs/2607.21351))
        - 서로 다른 과제로 학습한 adapter들의 부분공간이 상당히 겹쳐, 앞선 adapter가 뒤 과제를 대신하기도 함. 다만 그 겹침을 우연 수준과 비교하지는 않음 ([Dieudonné et al., 2026](https://arxiv.org/abs/2606.28117))
        - 찾은 부분공간을 투영 제거해 거부 행동만 없애거나 능력을 정밀 제거하는 절차까지 확립됨 ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717); [Fu et al., 2026](https://arxiv.org/abs/2601.07411))
    - 가중치 일부만 고쳐 생각을 끄거나 늘리기도 함
        - 추론 모델의 업데이트에서 가장 큰 성분을 빼면 생각을 끌 수 있고 ([Lee et al., 2025](https://arxiv.org/abs/2510.06750)), 몇몇 attention head에서 한 방향을 빼면 오히려 추론이 길어짐 ([Sun et al., 2025](https://aclanthology.org/2025.emnlp-main.861/))
        - LoRA 방향 하나의 크기로 추론 길이를 조절하도록 학습할 수도 있음 ([Ma et al., 2025a](https://aclanthology.org/2025.acl-long.300/))
    - 모델 내부의 신호를 SAE로 읽어 생각하는 방식을 찾기도 함
        - 추론 모델에서 반성·불확실성 feature를 찾아 키우면 풀이가 길어지고 ([Galichin et al., 2026](https://arxiv.org/abs/2503.18878)), 증류 모델에만 있는 반성·검산 feature로 과하게 생각하거나 간결하게 생각하도록 조절함 ([Baek & Tegmark, 2025](https://arxiv.org/abs/2503.03730))
    - 다만 이 연구들은 teacher가 하나로 고정되었거나 아예 없는 설정에서 쓰였고, 방향도 업데이트의 크기나 긴/짧은 풀이의 대조로 찾아, teacher가 바뀔 때 student 안에 생기는 자리는 다루지 않음
- **(본 연구)**
    - 본 연구는 teacher를 $\lambda$로 움직여 증류한 student들의 LoRA 업데이트를 나란히 비교해, teacher가 student의 가중치에 무엇을 남기는지 탐색함
