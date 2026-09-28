# 문장별 초안

- (배경)
   - Small students distilled on chain-of-thought (CoT) from large reasoning models gain little from long CoT and more from short CoT, a gap usually attributed to learnability.
   - 대형 추론 모델의 chain-of-thought(CoT)로 증류한 작은 student는 긴 CoT에서는 얻는 것이 적고 짧은 CoT에서 더 많이 얻으며, 이 차이는 보통 학습 가능성 때문으로 여겨짐
- (갭)
   - This explanation rests on accuracy and training signals, and none of this evidence shows what the student actually retains.
   - 이 설명은 정확도와 학습 신호에 기대고 있으며, 어느 근거도 student가 실제로 무엇을 담아 두는지는 보여주지 않음
- (연구 제안)
   - We measure it in bits across teacher styles from long reasoning to short solutions, separating knowledge that transfers to unseen problems from memorization of training problems.
   - 우리는 긴 추론부터 짧은 풀이까지 teacher 스타일을 바꿔 가며 그것을 bits로 측정하고, 처음 보는 문제로 넘어가는 지식과 학습 문제의 암기를 나눔
- (발견, 결과)
   - Surprisingly, the picture is reversed.
   - 놀랍게도 그림은 정반대였음
   - The short rationales that give the best accuracy are among the most memorized, while the long rationales that lower accuracy are learned almost entirely as knowledge that transfers.
   - 정확도가 가장 좋은 짧은 풀이는 가장 많이 암기되는 축에 속하고, 정확도를 떨어뜨리는 긴 풀이는 거의 전부 옮겨 쓸 수 있는 지식으로 학습됨
   - What decides the outcome is not the teacher's style but how predictable a rationale already is to the student, so the same rationale can be memorized by one student and learned by another.
   - 결과를 가르는 것은 teacher의 스타일이 아니라 풀이가 student에게 이미 얼마나 예측 가능한가이며, 그래서 같은 풀이도 어떤 student에게는 암기되고 다른 student에게는 학습됨
- (분석)
   - Long CoT does not fail for lack of learning, since students trained on it often cannot finish their long answers within the generation budget.
   - 긴 CoT의 실패는 배우지 못해서가 아니며, 그것으로 학습한 student는 긴 답을 생성 한도 안에서 끝내지 못하는 경우가 많음
- (우리 연구의 기여)
   - Our findings suggest that accuracy gains from CoT distillation can hide memorization, and that student-side predictability, which can be computed before training, should be weighed alongside accuracy when choosing what to distill.
   - 우리의 발견은 CoT 증류로 얻은 정확도 향상이 암기를 가릴 수 있으며, 학습 전에 계산할 수 있는 student 기준 예측 가능성을 무엇을 증류할지 고를 때 정확도와 함께 따져야 함을 시사함

# 영문

Small students distilled on chain-of-thought (CoT) from large reasoning models gain little from long CoT and more from short CoT, a gap usually attributed to learnability. This explanation rests on accuracy and training signals, and none of this evidence shows what the student actually retains. We measure it in bits across teacher styles from long reasoning to short solutions, separating knowledge that transfers to unseen problems from memorization of training problems. Surprisingly, the picture is reversed. The short rationales that give the best accuracy are among the most memorized, while the long rationales that lower accuracy are learned almost entirely as knowledge that transfers. What decides the outcome is not the teacher's style but how predictable a rationale already is to the student, so the same rationale can be memorized by one student and learned by another. Long CoT does not fail for lack of learning, since students trained on it often cannot finish their long answers within the generation budget. Our findings suggest that accuracy gains from CoT distillation can hide memorization, and that student-side predictability, which can be computed before training, should be weighed alongside accuracy when choosing what to distill.

# 한문

대형 추론 모델의 chain-of-thought(CoT)로 증류한 작은 student는 긴 CoT에서는 얻는 것이 적고 짧은 CoT에서 더 많이 얻으며, 이 차이는 보통 학습 가능성 때문으로 여겨진다. 이 설명은 정확도와 학습 신호에 기대고 있으며, 어느 근거도 student가 실제로 무엇을 담아 두는지는 보여주지 않는다. 우리는 긴 추론부터 짧은 풀이까지 teacher 스타일을 바꿔 가며 그것을 bits로 측정하고, 처음 보는 문제로 넘어가는 지식과 학습 문제의 암기를 나눈다. 놀랍게도 그림은 정반대였다. 정확도가 가장 좋은 짧은 풀이는 가장 많이 암기되는 축에 속하고, 정확도를 떨어뜨리는 긴 풀이는 거의 전부 옮겨 쓸 수 있는 지식으로 학습된다. 결과를 가르는 것은 teacher의 스타일이 아니라 풀이가 student에게 이미 얼마나 예측 가능한가이며, 그래서 같은 풀이도 어떤 student에게는 암기되고 다른 student에게는 학습된다. 긴 CoT의 실패는 배우지 못해서가 아니며, 그것으로 학습한 student는 긴 답을 생성 한도 안에서 끝내지 못하는 경우가 많다. 우리의 발견은 CoT 증류로 얻은 정확도 향상이 암기를 가릴 수 있으며, 학습 전에 계산할 수 있는 student 기준 예측 가능성을 무엇을 증류할지 고를 때 정확도와 함께 따져야 함을 시사한다.
