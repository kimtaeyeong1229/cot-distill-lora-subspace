# 섹션 구성

- 2 Related Work (소제목은 번호 없는 `\paragraph{...}`로 씀)
   - Chain-of-Thought Distillation.
   - Measuring Memorization.
   - Memorization in Distillation and Fine-Tuning.
- 참고한 선행 연구의 섹션명
   - MI-Distillation: "Reasoning in Large Language Models", "Knowledge Distillation" (소절 2개)
   - How Many Bits Can an Adapter Write?: "Measuring memorization.", "PEFT and privacy.", "Low-bit adaptation and what RLVR learns.", "Adapters as shared artifacts." (문단 제목 4개)
- CLAUDE.md 규칙에 따라 각 소절은 "우리 연구"를 끌어들이지 않고 문헌 흐름으로 마무리함
- 우리 연구의 위치는 Related Work 전체의 맨 끝(마지막 paragraph의 마지막 문장)에서만 밝힘 (MI-Distillation 2.2 마지막 문장 "Our work instead …" 방식)

# 문장별 초안

## Chain-of-Thought Distillation

- (현상)
   - Small students often fail to benefit from long CoT written by large reasoning models (guo2025deepseekr1) and learn better from short CoT and smaller teachers (li2025small), and CoT distillation lowers accuracy relative to the undistilled student in many settings (kajitsuka2026capacity).
   - 작은 student는 대형 추론 모델이 쓴 긴 CoT에서 이득을 얻지 못하고 짧은 CoT와 더 작은 teacher로 더 잘 배우는 경우가 많으며, CoT 증류는 여러 설정에서 증류 전 student보다 정확도를 떨어뜨림
- (기존 방법)
   - Methods either adjust the style of rationales, by mixing long and short CoT (li2025small), following a curriculum (jiang2025teach), or interpolating a reasoning model with an instruction model (lan2026midistillation), or rank rationales with student-side scores built on token rank and surprisal (yang2026reasoning), step-level likelihood (just2026signal), or gradient spectra (panigrahi2025grace).
   - 기존 방법들은 긴 CoT와 짧은 CoT를 섞거나, curriculum을 따르거나, 추론 모델과 지시 모델을 보간해 풀이의 스타일을 조절하거나, 토큰 순위와 surprisal, 단계 단위 likelihood, 그래디언트 스펙트럼으로 만든 student 기준 점수로 풀이의 순위를 매김
   - On-policy distillation instead trains on student samples and works best when teacher and student share thinking patterns (li2026onpolicy).
   - on-policy 증류는 대신 student가 생성한 샘플로 학습하며, teacher와 student의 사고 패턴이 비슷할 때 가장 잘 작동함
- (평가 방식의 공통점)
   - Some studies observe that students copy the surface form of teacher rationales (dai2025capture, cui2026whole), yet across this line of work what a student learns is judged by accuracy or training signals.
   - 일부 연구는 student가 teacher 풀이의 표면 형식을 따라 한다고 관찰하지만, 이 연구 흐름 전반에서 student가 무엇을 배웠는지는 정확도나 학습 신호로 판단함

## Measuring Memorization

- (추출 기반 측정)
   - Early work measures memorization by testing whether a model reproduces secrets inserted into its training data (carlini2019secret), and finds that extractable memorization grows with model size, duplication, and prompt length (carlini2023quantifying).
   - 초기 연구는 학습 데이터에 넣은 비밀을 모델이 재현하는지로 암기를 측정하며, 추출 가능한 암기가 모델 크기, 중복 횟수, 프롬프트 길이에 따라 커짐을 보임
- (압축 기반 측정)
   - Because extraction cannot separate memorization from prediction, compression-based measurement counts the bits a trained model saves on a sample relative to a reference model and separates unintended memorization from generalization (morris2025memorize).
   - 추출은 암기와 예측을 구분하지 못하므로, 압축 기반 측정은 학습된 모델이 참조 모델보다 샘플에서 절약하는 bits를 세어 의도치 않은 암기를 일반화와 분리함
   - Applied to LoRA adapters on random sequences and planted secrets, the same measurement shows that an adapter stores a couple of bits per trainable parameter and that leakage tracks the bits it writes rather than its parameter count (tan2026bits).
   - 같은 측정을 무작위 시퀀스와 심어 둔 비밀로 LoRA adapter에 적용하면, adapter는 학습 파라미터당 몇 bits를 저장하고 유출은 파라미터 수가 아니라 adapter가 쓴 bits를 따라감

## Memorization in Distillation and Fine-Tuning

- (증류에서의 암기)
   - Knowledge distillation memorizes less than standard fine-tuning of a student of the same size, with reverse-KL distillation memorizing the least (borkar2026memorization, singh2025teacher).
   - 지식 증류는 같은 크기의 student를 일반 파인튜닝할 때보다 덜 암기하며, reverse-KL 증류가 가장 적게 암기함
   - Memorized examples have low entropy and can be predicted before training, and sequence-level distillation passes more teacher-memorized examples to the student than logit-level distillation (borkar2026memorization).
   - 암기되는 예시는 엔트로피가 낮고 학습 전에 예측할 수 있으며, 시퀀스 단위 증류는 logit 단위 증류보다 teacher가 암기한 예시를 student에게 더 많이 넘김
- (LoRA 파인튜닝에서의 암기)
   - LoRA memorizes sensitive records up to ten times less than full fine-tuning, and raising its rank increases memorization while accuracy stays nearly flat (bossy2026mitigating).
   - LoRA는 full fine-tuning보다 민감한 기록을 최대 10배 덜 암기하며, rank를 올리면 정확도는 거의 그대로인 채 암기가 늘어남
- (우리 연구의 위치)
   - These studies vary the training objective, model size, or adapter rank but not the style of teacher rationales, and our work measures in bits what CoT distillation writes into a student across a spectrum of teacher styles, separating the part that transfers to unseen problems from the part that is memorized.
   - 이 연구들은 학습 목적 함수, 모델 크기, adapter rank는 바꾸지만 teacher 풀이의 스타일은 바꾸지 않았으며, 우리 연구는 teacher 스타일 스펙트럼에 걸쳐 CoT 증류가 student에 써 넣은 것을 bits로 재고 처음 보는 문제로 넘어가는 부분과 암기된 부분을 나눔

# 영문

## Chain-of-Thought Distillation

Small students often fail to benefit from long CoT written by large reasoning models (guo2025deepseekr1) and learn better from short CoT and smaller teachers (li2025small), and CoT distillation lowers accuracy relative to the undistilled student in many settings (kajitsuka2026capacity). Methods either adjust the style of rationales, by mixing long and short CoT (li2025small), following a curriculum (jiang2025teach), or interpolating a reasoning model with an instruction model (lan2026midistillation), or rank rationales with student-side scores built on token rank and surprisal (yang2026reasoning), step-level likelihood (just2026signal), or gradient spectra (panigrahi2025grace). On-policy distillation instead trains on student samples and works best when teacher and student share thinking patterns (li2026onpolicy). Some studies observe that students copy the surface form of teacher rationales (dai2025capture, cui2026whole), yet across this line of work what a student learns is judged by accuracy or training signals.

## Measuring Memorization

Early work measures memorization by testing whether a model reproduces secrets inserted into its training data (carlini2019secret), and finds that extractable memorization grows with model size, duplication, and prompt length (carlini2023quantifying). Because extraction cannot separate memorization from prediction, compression-based measurement counts the bits a trained model saves on a sample relative to a reference model and separates unintended memorization from generalization (morris2025memorize). Applied to LoRA adapters on random sequences and planted secrets, the same measurement shows that an adapter stores a couple of bits per trainable parameter and that leakage tracks the bits it writes rather than its parameter count (tan2026bits).

## Memorization in Distillation and Fine-Tuning

Knowledge distillation memorizes less than standard fine-tuning of a student of the same size, with reverse-KL distillation memorizing the least (borkar2026memorization, singh2025teacher). Memorized examples have low entropy and can be predicted before training, and sequence-level distillation passes more teacher-memorized examples to the student than logit-level distillation (borkar2026memorization). LoRA memorizes sensitive records up to ten times less than full fine-tuning, and raising its rank increases memorization while accuracy stays nearly flat (bossy2026mitigating). These studies vary the training objective, model size, or adapter rank but not the style of teacher rationales, and our work measures in bits what CoT distillation writes into a student across a spectrum of teacher styles, separating the part that transfers to unseen problems from the part that is memorized.

# 한문

## Chain-of-Thought Distillation

작은 student는 대형 추론 모델이 쓴 긴 CoT에서 이득을 얻지 못하고 짧은 CoT와 더 작은 teacher로 더 잘 배우는 경우가 많으며, CoT 증류는 여러 설정에서 증류 전 student보다 정확도를 떨어뜨린다. 기존 방법들은 긴 CoT와 짧은 CoT를 섞거나, curriculum을 따르거나, 추론 모델과 지시 모델을 보간해 풀이의 스타일을 조절하거나, 토큰 순위와 surprisal, 단계 단위 likelihood, 그래디언트 스펙트럼으로 만든 student 기준 점수로 풀이의 순위를 매긴다. on-policy 증류는 대신 student가 생성한 샘플로 학습하며, teacher와 student의 사고 패턴이 비슷할 때 가장 잘 작동한다. 일부 연구는 student가 teacher 풀이의 표면 형식을 따라 한다고 관찰하지만, 이 연구 흐름 전반에서 student가 무엇을 배웠는지는 정확도나 학습 신호로 판단한다.

## Measuring Memorization

초기 연구는 학습 데이터에 넣은 비밀을 모델이 재현하는지로 암기를 측정하며, 추출 가능한 암기가 모델 크기, 중복 횟수, 프롬프트 길이에 따라 커짐을 보였다. 추출은 암기와 예측을 구분하지 못하므로, 압축 기반 측정은 학습된 모델이 참조 모델보다 샘플에서 절약하는 bits를 세어 의도치 않은 암기를 일반화와 분리한다. 같은 측정을 무작위 시퀀스와 심어 둔 비밀로 LoRA adapter에 적용하면, adapter는 학습 파라미터당 몇 bits를 저장하고 유출은 파라미터 수가 아니라 adapter가 쓴 bits를 따라간다.

## Memorization in Distillation and Fine-Tuning

지식 증류는 같은 크기의 student를 일반 파인튜닝할 때보다 덜 암기하며, reverse-KL 증류가 가장 적게 암기한다. 암기되는 예시는 엔트로피가 낮고 학습 전에 예측할 수 있으며, 시퀀스 단위 증류는 logit 단위 증류보다 teacher가 암기한 예시를 student에게 더 많이 넘긴다. LoRA는 full fine-tuning보다 민감한 기록을 최대 10배 덜 암기하며, rank를 올리면 정확도는 거의 그대로인 채 암기가 늘어난다. 이 연구들은 학습 목적 함수, 모델 크기, adapter rank는 바꾸지만 teacher 풀이의 스타일은 바꾸지 않았으며, 우리 연구는 teacher 스타일 스펙트럼에 걸쳐 CoT 증류가 student에 써 넣은 것을 bits로 재고 처음 보는 문제로 넘어가는 부분과 암기된 부분을 나눈다.
