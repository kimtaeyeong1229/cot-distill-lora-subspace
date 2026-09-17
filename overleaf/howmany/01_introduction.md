# 1. Introduction — 문장 분해

출처: arXiv 2607.21351v1, `sections/1_introduction.tex` (인용은 저자-연도로 풀어 씀)

## 문단 1 — 통념과 그 위험의 규모

- (배경)
   - Adapter files circulate as if they were harmless diffs.
   - LoRA adapters (Hu et al. 2022) are shared through public model hubs, exchanged as low-rank updates between federated clients, and passed between teams as task adapters—on the tacit understanding that a small file carries a skill, not the data that taught it.
- (갭)
   - Nobody has checked that understanding against a measurement.
- (동기, 규모 계산)
   - The arithmetic alone should give pause: a rank-16 adapter for a 7B model holds some 4×10^7 trainable parameters, and if it stored information at the 3.6 bits per parameter that Morris et al. (2026b) found for full models, one such file could carry the equivalent of twenty megabytes of training text—a whole fine-tuning corpus of medical notes or support transcripts, passed around as a convenience.

## 문단 2 — 왜 fine-tuning에서 문제가 되는가, 기존 연구의 한계

- (동기)
   - What makes this more than a curiosity is where it bites.
- (선행 연구)
   - Morris et al. (2026b) showed that pretraining lives in the comfortable regime: the corpus dwarfs what the model can hold, so any single example is diluted and membership inference falls to chance.
- (대비, 문제 상황)
   - Fine-tuning inverts that picture.
   - Adaptation sets are small and often sensitive, and they routinely hold fewer bits than even a modest adapter can absorb—the regime in which, by the same theory, memorization is total by default.
   - The one reassurance the field offers about memorization is calibrated on exactly the case that does not apply, and the only physical quantity left to bound the leak is the size of the adapter itself.
- (갭)
   - Yet everything we know about that quantity comes from attacks: prior work plants secrets and tries to extract them, or runs membership inference and reports an AUC (Wang and Li 2025; Bossy et al. 2026; Malekmohammadi and Farnadi 2024).
   - An attack tells you what one adversary recovered.
   - It does not tell you what the artifact holds.

## 문단 3 — 접근 방법

- (연구 제안)
   - So we weigh the artifact.
- (방법)
   - Following Morris et al. (2026b), we read memorization off a compression advantage: a model that has memorized a string can describe it in fewer bits than a reference can, and the gap is the information the model absorbed.
   - To make the reference exact we let frozen bases memorize uniformly random token sequences—strings with nothing to learn, so that every bit recovered is a bit stored, not a bit inferred.
- (방법의 정당화)
   - Random strings are not a use case; they are the calibration weight that lets us then trust the scale on real text.

## 문단 4 — 첫 번째 발견: 용량

- (발견 요약)
   - What the scale reads back disagrees with the folklore in two directions at once.
- (발견 1: 작다)
   - Adapters do far less than full models—a couple of bits per trainable parameter against the full-model 3.6, confirmed by fine-tuning the same base under the same budget.
- (발견 2: 파라미터 수로 설명되지 않는다)
   - But the per-parameter figure is not a property of the parameters.
   - It moves by nearly a factor of two when the same budget of weights is placed in the MLP rather than in attention, it barely notices half-precision storage, and it collapses when the frozen base has no structure to steer: the adapter that memorizes 98% of its data on a pretrained base manages 29% on a random one.
- (해석)
   - What an adapter can write is set by where it writes and what it writes into, not by how many numbers it contains.

## 문단 5 — 두 번째 발견(RL)과 기여

- (확장, 배경)
   - The same scale settles a second, noisier debate.
   - A recent line argues over what reinforcement learning with verifiable rewards actually deposits in the weights—new capability, or a sharpening of what pretraining already knew (Chu et al. 2025; Morris et al. 2026a)—and argues it entirely from behavior.
- (연구 제안)
   - We bring bits instead.
- (발견)
   - At matched accuracy, an RLVR adapter writes nothing our estimator can distinguish from zero, and in a direct test the private secrets that supervised fine-tuning copies down verbatim, reinforcement learning never records at all.
- (기여)
   - We contribute a measurement protocol for adapter capacity, released as a harness; capacity readings that separate written bits from parameter count, across placement, precision, and substrate; an audit tying those bits to real canary extraction and membership inference on identical checkpoints; and the first bit-level comparison of supervised against reinforcement fine-tuning.
- (그림 예고)
   - Figure 1 previews the core of it: capacity plateaus that sit well below the full-model line and move when parameter count does not.

## 구조 메모

- 5문단, 배경 → 갭 → 방법 → 발견 1 → 발견 2 + 기여 순서임
- 갭을 두 번 씀
   - 문단 1: "Nobody has checked" (측정 자체가 없음)
   - 문단 2: "everything we know … comes from attacks" (있는 근거가 간접적임)
- 추상적 위험을 구체 숫자로 바꿔 동기를 만듦 (rank-16, 7B, 4×10^7 파라미터 → 20MB 텍스트)
- 각 문단 첫 문장이 짧고 요약적임 ("So we weigh the artifact.", "The same scale settles a second, noisier debate.")
- 기여 목록을 bullet 없이 세미콜론으로 이어진 한 문장으로 씀 (프로토콜, 용량 측정, 유출 감사, SFT vs RL)
- 방법의 한계(랜덤 문자열은 실제 사용 사례가 아님)를 Introduction에서 먼저 짚고 "calibration weight"로 정당화함
