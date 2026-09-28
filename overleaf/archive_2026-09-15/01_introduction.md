# 문장별 초안

- (배경)
   - Large reasoning models such as DeepSeek-R1 solve complex mathematical and coding problems by writing long chain-of-thought (CoT) rationales (guo2025deepseekr1).
   - DeepSeek-R1 같은 대형 추론 모델은 긴 chain-of-thought(CoT) 풀이를 써서 복잡한 수학·코딩 문제를 풂
   - These models are expensive to serve, so their reasoning is commonly distilled into small students by fine-tuning on teacher rationales (guo2025deepseekr1, li2025small), often with parameter-efficient methods such as low-rank adaptation (LoRA) (hu2021lora).
   - 이런 모델은 운영 비용이 커서 teacher 풀이로 작은 student를 파인튜닝해 추론 능력을 증류하는 방식이 흔히 쓰이며, low-rank adaptation(LoRA) 같은 parameter-efficient 방식을 함께 쓰기도 함
- (배경: 기존 설명)
   - Small students, however, gain little from long CoT and often perform better when trained on short CoT (li2025small, lan2026midistillation).
   - 그러나 작은 student는 긴 CoT에서 얻는 것이 적고, 짧은 CoT로 학습할 때 오히려 더 나은 경우가 많음
   - The usual explanation is learnability, since short rationales are easier for a small student to learn from while long rationales are rich but hard to imitate (lan2026midistillation, li2025small).
   - 보통의 설명은 학습 가능성(learnability)이며, 짧은 풀이는 작은 student가 배우기 쉽고 긴 풀이는 풍부하지만 따라 하기 어렵다는 것임
   - Methods built on this view choose data that a student can learn from, by mixing long and short CoT (li2025small), raising difficulty through a curriculum (jiang2025teach), interpolating teacher weights (lan2026midistillation), or ranking rationales with student-side scores (yang2026reasoning, just2026signal, panigrahi2025grace).
   - 이 관점에 기반한 방법들은 긴 CoT와 짧은 CoT를 섞거나, curriculum으로 난이도를 올리거나, teacher 가중치를 보간하거나, student 기준 점수로 풀이의 순위를 매겨 student가 배울 수 있는 데이터를 고름
- (한계)
   - These methods judge a student by its accuracy or by training signals, but accuracy alone cannot show whether a student learned to reason or memorized its training rationales.
   - 이 방법들은 student를 정확도나 학습 신호로 판단하지만, 정확도만으로는 student가 추론을 배웠는지 학습 풀이를 외웠는지 알 수 없음
   - Bit-level measurements of what a trained model stores exist (morris2025memorize, tan2026bits), but they have not been applied to CoT distillation.
   - 학습된 모델이 무엇을 저장하는지 bits로 재는 측정은 있지만, CoT 증류에는 적용된 적이 없음
- (연구 질문)
   - When distillation data improves accuracy, does the data teach the student something that carries over to new problems, or does it make the student memorize the training problems?
   - 증류 데이터가 정확도를 올릴 때, 그 데이터는 student에게 새로운 문제로 넘어가는 무언가를 가르치는가, 아니면 학습 문제를 외우게 하는가?
- (우리 방법)
   - We answer this question by measuring in bits what CoT distillation writes into a student (Figure 1).
   - 우리는 CoT 증류가 student 안에 무엇을 써 넣는지 bits로 측정해 이 질문에 답함
   - Building on compression-based memorization measurement (morris2025memorize, tan2026bits), we split the drop in the codelength of teacher rationales after training into transfer, the part that carries over to unseen problems, and memorization, the part that does not.
   - 압축 기반 암기 측정을 바탕으로, 학습 후 teacher 풀이의 부호 길이가 줄어든 양을 처음 보는 문제로 넘어가는 부분인 전이와 넘어가지 않는 부분인 암기로 나눔
   - To change only the style of the teacher, we use the weight-interpolated teacher spectrum of MI-Distillation (lan2026midistillation), keep the problems, LoRA configuration, and training recipe fixed, and repeat the measurement across several teacher pairs and students from three model families.
   - teacher의 스타일만 바꾸기 위해 MI-Distillation의 가중치 보간 teacher 스펙트럼을 쓰고, 문제·LoRA 설정·학습 방식을 고정한 채 여러 teacher 쌍과 세 모델 계열의 student에서 측정을 반복함
- (발견, 결과)
   - Contrary to the learnability view, short rationales that give the best accuracy are among the most memorized, while long rationales that lower accuracy transfer more than 90% [TBD] of the information the student absorbs (Section 4.2).
   - 학습 가능성 관점과 반대로, 정확도가 가장 좋은 짧은 풀이는 가장 많이 암기되는 축에 속하고, 정확도를 떨어뜨리는 긴 풀이는 student가 흡수한 정보의 90% [TBD] 이상을 처음 보는 문제로 넘김
   - The split follows how predictable a rationale already is to the student far more closely than the style of the teacher, so the same short rationale is mostly memorized by Qwen2.5-0.5B and largely transferred by Llama-3.2-1B (Section 5.1).
   - 이 비율은 teacher의 스타일보다 풀이가 student에게 이미 얼마나 예측 가능한가를 훨씬 가깝게 따라가며, 그래서 같은 짧은 풀이가 Qwen2.5-0.5B에게는 대부분 암기되고 Llama-3.2-1B에게는 상당 부분 전이됨
   - The accuracy drop under long CoT coincides with students failing to finish their answers within the generation budget (Section 6.2).
   - 긴 CoT에서의 정확도 하락은 student가 생성 한도 안에서 답을 끝내지 못하는 현상과 함께 나타남
- (기여)
   - Our contributions are as follows.
   - 우리의 기여는 다음과 같음
   - We introduce a bit-level measurement that splits what CoT distillation writes into a student into transfer and memorization.
   - CoT 증류가 student에 써 넣은 것을 전이와 암기로 나누는 bit 단위 측정을 제시함
   - Across teacher styles and students from three model families, we show that the most accurate rationales are among the most memorized and that the split tracks student-side predictability, and we test this relation with a token-matched control, repeated training, and multiple seeds.
   - teacher 스타일과 세 모델 계열의 student에서, 정확도가 가장 좋은 풀이가 가장 많이 암기되는 축에 속하고 그 비율이 student 기준 예측 가능성을 따라감을 보이며, 토큰 수 통제, 반복 학습, 여러 seed로 이 관계를 확인함
   - We show that accuracy gains from CoT distillation can hide memorization, and suggest weighing student-side predictability, which can be computed before training, when choosing distillation data.
   - CoT 증류로 얻은 정확도 향상이 암기를 가릴 수 있음을 보이고, 증류 데이터를 고를 때 학습 전에 계산할 수 있는 student 기준 예측 가능성을 함께 따질 것을 제안함

# 영문

Large reasoning models such as DeepSeek-R1 solve complex mathematical and coding problems by writing long chain-of-thought (CoT) rationales (guo2025deepseekr1). These models are expensive to serve, so their reasoning is commonly distilled into small students by fine-tuning on teacher rationales (guo2025deepseekr1, li2025small), often with parameter-efficient methods such as low-rank adaptation (LoRA) (hu2021lora).

Small students, however, gain little from long CoT and often perform better when trained on short CoT (li2025small, lan2026midistillation). The usual explanation is learnability, since short rationales are easier for a small student to learn from while long rationales are rich but hard to imitate (lan2026midistillation, li2025small). Methods built on this view choose data that a student can learn from, by mixing long and short CoT (li2025small), raising difficulty through a curriculum (jiang2025teach), interpolating teacher weights (lan2026midistillation), or ranking rationales with student-side scores (yang2026reasoning, just2026signal, panigrahi2025grace).

These methods judge a student by its accuracy or by training signals, but accuracy alone cannot show whether a student learned to reason or memorized its training rationales. Bit-level measurements of what a trained model stores exist (morris2025memorize, tan2026bits), but they have not been applied to CoT distillation. When distillation data improves accuracy, does the data teach the student something that carries over to new problems, or does it make the student memorize the training problems?

We answer this question by measuring in bits what CoT distillation writes into a student (Figure 1). Building on compression-based memorization measurement (morris2025memorize, tan2026bits), we split the drop in the codelength of teacher rationales after training into transfer, the part that carries over to unseen problems, and memorization, the part that does not. To change only the style of the teacher, we use the weight-interpolated teacher spectrum of MI-Distillation (lan2026midistillation), keep the problems, LoRA configuration, and training recipe fixed, and repeat the measurement across several teacher pairs and students from three model families.

Contrary to the learnability view, short rationales that give the best accuracy are among the most memorized, while long rationales that lower accuracy transfer more than 90% [TBD] of the information the student absorbs (Section 4.2). The split follows how predictable a rationale already is to the student far more closely than the style of the teacher, so the same short rationale is mostly memorized by Qwen2.5-0.5B and largely transferred by Llama-3.2-1B (Section 5.1). The accuracy drop under long CoT coincides with students failing to finish their answers within the generation budget (Section 6.2).

Our contributions are as follows.

- We introduce a bit-level measurement that splits what CoT distillation writes into a student into transfer and memorization.
- Across teacher styles and students from three model families, we show that the most accurate rationales are among the most memorized and that the split tracks student-side predictability, and we test this relation with a token-matched control, repeated training, and multiple seeds.
- We show that accuracy gains from CoT distillation can hide memorization, and suggest weighing student-side predictability, which can be computed before training, when choosing distillation data.

# 한문

DeepSeek-R1 같은 대형 추론 모델은 긴 chain-of-thought(CoT) 풀이를 써서 복잡한 수학·코딩 문제를 푼다. 이런 모델은 운영 비용이 커서 teacher 풀이로 작은 student를 파인튜닝해 추론 능력을 증류하는 방식이 흔히 쓰이며, low-rank adaptation(LoRA) 같은 parameter-efficient 방식을 함께 쓰기도 한다.

그러나 작은 student는 긴 CoT에서 얻는 것이 적고, 짧은 CoT로 학습할 때 오히려 더 나은 경우가 많다. 보통의 설명은 학습 가능성(learnability)이며, 짧은 풀이는 작은 student가 배우기 쉽고 긴 풀이는 풍부하지만 따라 하기 어렵다는 것이다. 이 관점에 기반한 방법들은 긴 CoT와 짧은 CoT를 섞거나, curriculum으로 난이도를 올리거나, teacher 가중치를 보간하거나, student 기준 점수로 풀이의 순위를 매겨 student가 배울 수 있는 데이터를 고른다.

이 방법들은 student를 정확도나 학습 신호로 판단하지만, 정확도만으로는 student가 추론을 배웠는지 학습 풀이를 외웠는지 알 수 없다. 학습된 모델이 무엇을 저장하는지 bits로 재는 측정은 있지만, CoT 증류에는 적용된 적이 없다. 증류 데이터가 정확도를 올릴 때, 그 데이터는 student에게 새로운 문제로 넘어가는 무언가를 가르치는가, 아니면 학습 문제를 외우게 하는가?

우리는 CoT 증류가 student 안에 무엇을 써 넣는지 bits로 측정해 이 질문에 답한다. 압축 기반 암기 측정을 바탕으로, 학습 후 teacher 풀이의 부호 길이가 줄어든 양을 처음 보는 문제로 넘어가는 부분인 전이와 넘어가지 않는 부분인 암기로 나눈다. teacher의 스타일만 바꾸기 위해 MI-Distillation의 가중치 보간 teacher 스펙트럼을 쓰고, 문제·LoRA 설정·학습 방식을 고정한 채 여러 teacher 쌍과 세 모델 계열의 student에서 측정을 반복한다.

학습 가능성 관점과 반대로, 정확도가 가장 좋은 짧은 풀이는 가장 많이 암기되는 축에 속하고, 정확도를 떨어뜨리는 긴 풀이는 student가 흡수한 정보의 90% [TBD] 이상을 처음 보는 문제로 넘긴다(4.2절). 이 비율은 teacher의 스타일보다 풀이가 student에게 이미 얼마나 예측 가능한가를 훨씬 가깝게 따라가며, 그래서 같은 짧은 풀이가 Qwen2.5-0.5B에게는 대부분 암기되고 Llama-3.2-1B에게는 상당 부분 전이된다(5.1절). 긴 CoT에서의 정확도 하락은 student가 생성 한도 안에서 답을 끝내지 못하는 현상과 함께 나타난다(6.2절).

우리의 기여는 다음과 같다.

- CoT 증류가 student에 써 넣은 것을 전이와 암기로 나누는 bit 단위 측정을 제시한다.
- teacher 스타일과 세 모델 계열의 student에서, 정확도가 가장 좋은 풀이가 가장 많이 암기되는 축에 속하고 그 비율이 student 기준 예측 가능성을 따라감을 보이며, 토큰 수 통제, 반복 학습, 여러 seed로 이 관계를 확인한다.
- CoT 증류로 얻은 정확도 향상이 암기를 가릴 수 있음을 보이고, 증류 데이터를 고를 때 학습 전에 계산할 수 있는 student 기준 예측 가능성을 함께 따질 것을 제안한다.

# Figure 1 (2쪽 상단, 2컬럼)

- 파일: `figures/main_figure.pdf` (원본 `materials/main_figure_pptx/main_figure.pptx`)
- 캡션 (영문)
   - Overview of our study. *Left* shows the teacher spectrum, in which the weights of a reasoning teacher and an instruction teacher from the same base are interpolated so that rationales range from long CoT at λ = 0 to short solutions at λ = 1. *Middle* shows distillation, in which the rationales of each teacher train a LoRA adapter on a frozen student while unseen problems are held out. *Right* shows what the bit measurement reveals. The same rationale is mostly memorized by a student that already finds it predictable (low ρ) and mostly transferred by a student that finds it new (high ρ), so the split depends on the student rather than on the teacher.
- 캡션 (한문)
   - 연구 개요. 왼쪽은 teacher 스펙트럼으로, 같은 base의 추론 teacher와 지시 teacher의 가중치를 보간해 풀이가 λ = 0의 긴 CoT부터 λ = 1의 짧은 풀이까지 이어짐. 가운데는 증류로, 각 teacher의 풀이로 고정된 student 위의 LoRA adapter를 학습하고 처음 보는 문제는 따로 둠. 오른쪽은 bit 측정이 드러내는 것으로, 같은 풀이도 이미 예측 가능한 student(낮은 ρ)는 대부분 암기하고 새롭게 느끼는 student(높은 ρ)는 대부분 옮겨 쓸 수 있는 지식으로 배우므로 나뉨은 teacher가 아니라 student에 달려 있음
