# 4. The Capacity of Adapters — 문장 분해

출처: arXiv 2607.21351v1, `sections/4_capacity.tex` (그림·표 캡션 제외)

## 4.1 Plateaus, far below the full-model line

### 문단 1 — 전체 양상

- (발견)
   - Every configuration tells the same story (Figure 1): memorization climbs with the data while data is scarce, then bends onto a plateau once the dataset carries more bits than the adapter can hold.
- (검증)
   - The two largest readings per configuration agree to within a few percent, and held-out sequences stay at zero throughout—so the plateau is a ceiling on storage, not an accident of the fit.

### 문단 2 — 핵심 수치

- (결과: 기준선)
   - Two numbers frame the result (Table 1).
   - Full-model training stores about 3.6 bits per parameter (Morris et al. 2026b); fine-tuning the same base under our own budget confirms it climbs past 3.
- (결과: adapter)
   - Adapters, given that same budget, stop at 1.7 to 2.8.
- (반론 차단)
   - The gap is real and not an artifact of stopping early—it survives the matched budget, and the readings are stable across seeds and only creep upward with more compute, never enough to close it.
- (해석)
   - An adapter, whatever else it is, is a far leakier vessel per parameter than the model it patches, but a much smaller one in absolute terms.

### 문단 3 — 용량을 넘었을 때

- (관찰)
   - Past capacity the picture changes character.
   - Push a dataset well beyond what an adapter can hold and its memorization does not simply saturate: under a fixed budget it falls, the over-capacity runs tracing a descent rather than a plateau (the hollow points of Figure 1).
- (해석: 선행 연구와 연결)
   - This is the adapter-scale shadow of the grokking transition the full-model work reports—once the data outruns the room, gradient interference under a bounded budget wins, and a model that can no longer memorize everything ends up memorizing less of it.
- (범위 한정)
   - We read this tail as an optimization regime rather than a storage reading and keep our dataset grid below it, but it marks the seam where the memorization story ends and a generalization story would begin.

## 4.2 It is not the parameter count

### 문단 1 — 위치 실험

- (반론 제시)
   - The natural objection to a capacity plateau is that it just counts parameters: more weights, more room.
- (반론 기각)
   - It does not.
- (실험과 결과)
   - Fix the trainable count at about 37K and change only where the weights live—attention alone, all modules, or the MLP alone, with ranks chosen to keep the count equal—and per-parameter capacity runs from 1.30 to 2.43 bits, a near-doubling with parameter count held flat (Figure 2, left; three seeds, tight).
- (해석)
   - The same budget of numbers stores far more when it sits in the MLP than in attention.
   - What an adapter can hold is a property of the update's shape, not its size.

### 문단 2 — 정밀도 실험과 측정 하한

- (결과: 정밀도)
   - Precision tells the same story from another angle: casting a trained adapter to bf16 costs about one percent of its memorization while halving the file, and fp16 costs essentially nothing.
- (해석)
   - The bits an adapter has written are robust to how coarsely its weights are stored.
- (한계: 측정 하한)
   - Only at the extreme of weight tying does the instrument hit its floor—the 4,096-parameter TinyLoRA configurations never repay the base's own surprise at random data within the budget (a large deficit we quantify in Appendix C), so we report them as below the measurement floor rather than as zero.

## 4.3 And it is not the adapter alone

### 문단 1 — base 교체 실험

- (실험)
   - Hold the adapter, the optimizer, the data, and the budget fixed, and change only the frozen base underneath, and the capacity moves anyway (Figure 3).
- (결과)
   - The same LoRA memorizes 29% of its data on a randomly initialized base, 79% on one pretrained on structureless synthetic text, and 98% on one pretrained on WikiText—while full fine-tuning of that random base reaches 99%, so the shortfall is specific to writing through frozen weights.

### 문단 2 — 두 결과를 묶는 기제

- (종합)
   - The two dissociations point at one mechanism.
- (기제)
   - A low-rank adapter cannot move a frozen model wherever it likes; it can only push along directions the base already supports, and how much it can store is set by how rich those directions are.
- (기제로 두 결과 설명)
   - That is why the MLP, which carries most of a transformer's representational width, holds more per parameter than attention, and why a base with real linguistic structure offers far more to steer than a random one.
   - An adapter does not deposit its data into a vacuum—it re-uses what the base already knows how to represent, and its capacity is the room that re-use leaves it.
- (결론과 주의)
   - Capacity, then, belongs jointly to the adapter and the substrate it writes into, which is a caution as much as a finding: a bits-per-parameter number measured on one base should not be read off another.

## 구조 메모

- 소제목 자체가 주장임: "Plateaus, far below …", "It is not the parameter count", "And it is not the adapter alone"
   - 4.2와 4.3 제목이 "It is not … / And it is not …"으로 이어져 목차만 읽어도 논리가 보임
- 결과 문단마다 "결과 → 반론 차단 → 해석 한 문장" 순서를 지킴
- 4.2는 예상 반론을 먼저 말하고 두 단어("It does not.")로 기각한 뒤 실험을 보여주는 수사 구조임
- 4.3 마지막 문단에서 두 개의 dissociation을 하나의 기제로 묶어 절을 마무리함
- 용량을 넘는 구간(성능 하락)은 결과에서 제외하되, 왜 제외했는지와 그 의미를 한 문단으로 따로 설명함
