# 8. Conclusion — 문장 분해

출처: arXiv 2607.21351v1, `sections/7_conclusion.tex`의 Conclusion 절

## 문단 1 — 결과 요약

- (발견 1: 용량)
   - Weighed in bits, a parameter-efficient adapter is not the harmless diff its reputation suggests: it holds a couple of bits per trainable parameter, and its capacity is a property of the update's shape and the substrate it writes through, not of parameter count alone.
- (함의)
   - That turns the privacy of a shared adapter into something one can measure in advance rather than discover only under attack—set the information in an adaptation set against the adapter's capacity, and the regime is fixed before the first gradient step.
- (발견 2: RL)
   - Read at the same bit level, reinforcement learning with verifiable rewards deposits almost nothing into the weights where supervised fine-tuning writes its data down verbatim, settling with direct evidence a question argued until now from behavior alone.

## 문단 2 — 남기고 싶은 것

- (기여: 방법)
   - What we mean to leave behind is the method.
   - The harness we release measures, in bits and on any frozen base, what an adapter has written, and the three readings it returns are cheap enough to run before a checkpoint is shared, not only after it is attacked.
- (일반화된 메시지)
   - What transfers, at any scale, is that move: a scale you can hang on an adapter, and the habit of asking not what an attacker can recover from a fine-tune, but what the fine-tune was ever able to write.

## 구조 메모

- 2문단, 5문장으로 짧음
- 문단 1은 발견 두 개와 함의 하나를 각각 한 문장으로 요약함
- 문단 2는 수치 없이 "방법"과 "질문하는 습관"을 기여로 남기며 끝냄
- 마지막 문장이 Abstract 마지막 문장("Measuring what fine-tuning writes—rather than attacking it after the fact")과 같은 대비 구조를 반복함
