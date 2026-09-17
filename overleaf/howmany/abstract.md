# How Many Bits Can an Adapter Write? — Abstract 문장 분해

출처: arXiv 2607.21351v1 (Tan, Du, Feng, 2026), `materials/2607.21351.pdf` 1쪽

- (배경)
   - A LoRA adapter is a few megabytes that almost everyone treats as a skill rather than a record of the data behind it.
- (갭)
   - 초록에는 갭을 따로 쓴 문장이 없음
      - 배경 문장의 "almost everyone treats as a skill"이 검증되지 않은 통념을 제시하는 방식으로 갭을 대신함
      - 구체적인 갭(지금까지의 근거가 전부 공격 실험에서 나왔다는 점)은 Introduction 2문단에 있음
- (연구 제안)
   - We put that assumption on a scale.
   - Extending compression-based memorization analysis to the frozen-base setting, we measure directly—in bits—how much a low-rank adapter writes into a model it never changes.
- (발견, 결과)
   - The answer is both smaller than full fine-tuning and less lawful than parameter counting would predict.
   - Adapters store a couple of bits per trainable parameter, well short of a full model's budget; but that figure turns less on how many parameters an adapter carries than on where they sit.
   - Move the same parameter budget from attention into the MLP and it holds nearly twice as much; strip the frozen base of its structure and the capacity all but disappears.
- (분석, 응용)
   - Turned on realistic fine-tunes of Qwen2.5, the same instrument shows privacy leakage rising with the bits an adapter writes rather than the parameters it nominally has,
   - and it draws a clean line between supervised and reinforcement learning: the secrets that supervised fine-tuning copies down verbatim, an adapter trained on verifiable rewards never records.
- (우리 연구의 기여)
   - Measuring what fine-tuning writes—rather than attacking it after the fact—turns a piece of folklore into a quantity one can design against.

## 구조 메모

- 9문장, 약 190단어로 짧음
- 갭을 "기존 연구가 X를 안 했음"으로 쓰지 않고, 모두가 믿는 통념 하나로 제시한 뒤 곧바로 "그걸 재 보겠음"으로 넘어감
- 발견은 요약 한 문장("smaller … and less lawful …") 뒤에 구체 수치 두 문장으로 풀어 쓰는 계단식임
   - 요약 문장의 두 형용사가 뒤 두 문장과 하나씩 대응함 (smaller → a couple of bits, less lawful → where they sit)
- 응용은 "the same instrument"로 시작해, 측정 도구 하나로 서로 다른 질문(프라이버시, SFT vs RL)에 답했음을 강조함
- 기여 문장은 수치 없이 관점의 전환("folklore → quantity")으로 끝남
- 대비 구문을 반복해서 씀
   - skill rather than a record, bits it writes rather than parameters it nominally has, measuring rather than attacking
