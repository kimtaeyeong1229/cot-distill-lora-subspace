# 7. Discussion — 문장 분해

출처: arXiv 2607.21351v1, `sections/7_conclusion.tex`의 Discussion 절

## 문단 1 — 핵심 결과 재진술과 관점 전환

- (핵심 결과 요약)
   - An adapter's capacity is a number you can put on it.
   - Weighed in bits, a LoRA file holds a couple of bits per trainable parameter—less than a full model, but far from the nothing its reputation as a harmless diff suggests—and how much it holds is decided as much by where the update sits and what base it writes through as by how large it is.
- (관점 전환)
   - That reframes the privacy question a shared adapter raises.
   - Instead of training first and attacking afterward to find out what leaked, one can weigh an adaptation set's information content against the adapter's measured capacity beforehand, and know which regime one is in.

## 문단 2 — 구체적 사용법

- (적용 예시)
   - The check is concrete.
   - Take the rank-16 adapter for a 7B model from our opening: at the two to three bits per parameter we measure, it holds on the order of ten megabytes of text—several thousand pages, more than most fine-tuning corpora contain.
- (판단 기준)
   - Set that capacity against the corpus.
   - A corpus that carries fewer bits than the adapter can hold sits squarely in the regime where, by the theory we build on, memorization is total, and the resulting file should be treated as carrying its training data until an audit shows otherwise; a corpus that carries far more dilutes any single example and pulls the risk back down.
   - The ratio of dataset bits to adapter capacity—both computable before a single gradient step—places a fine-tune on that spectrum in advance.
- (조정 수단)
   - And the levers that move it are exactly the ones no one currently adjusts for privacy: not the rank alone, but where the adapter is placed and how much structure the frozen base already carries.

## 문단 3 — adapter 공유 관행에 대한 함의

- (함의)
   - That last point changes what it means to share an adapter at all.
   - A hub hosting a LoRA, a federated client uploading an update, a marketplace trading a task adapter—each is, on this accounting, distributing a lossy but often complete copy of whatever data the adapter was small enough to memorize and large enough to hold.
- (제안)
   - Treating adapters as data artifacts rather than skill patches, and publishing a measured capacity alongside a checkpoint the way a file lists its size, would let whoever downloads it reason about exposure before they ever run an attack.
   - It would also give the party doing the fine-tuning something to act on: a target capacity to design for, met by choosing a rank, a placement, and a base whose measured room sits below the information the data must not reveal, rather than a hope that a small file will keep a secret it was never too small to hold.

## 문단 4 — RL 논쟁에 대한 함의

- (연결)
   - The same instrument speaks to a live argument it was not built for.
- (배경)
   - Whether reinforcement learning with verifiable rewards teaches a model anything new, or merely sharpens what pretraining left, has been fought almost entirely with behavioral evidence.
- (결과 재진술)
   - Reading the bits directly gives a different kind of answer: at matched accuracy the reinforcement-learned adapter writes essentially nothing, and the private facts supervised fine-tuning memorizes verbatim, it never records.
- (실무적 권고)
   - Taken with the direction of the thirteen-parameter result, this is a reason to reach for verifiable-reward training precisely where an adapter will be shared and its training data must not travel with it—and a caution that the same is not true of supervised fine-tuning, which will carry the data along by construction.

## 문단 5 — 측정의 신뢰성

- (보수성)
   - The accounting is conservative by construction.
   - Every capacity we report is a lower bound at a declared optimization budget, so a measurement can understate what an adapter holds but never inflate it—the safe direction for a privacy claim—and the protocol runs unchanged from a few-million-parameter base up to 0.5B, where it returns the same picture.
- (방법 한계에 대한 답)
   - The random-string probe, in turn, is a calibration instrument rather than a threat model: it isolates storage by stripping away everything a model could otherwise learn, which is precisely why the text audits of Sections 5 and 6 carry the leakage argument on real data.

## 구조 메모

- 결과 요약 → 관점 전환 → 구체 사용법(숫자 예시) → 생태계 함의 → 두 번째 결과의 함의 → 측정의 신뢰성 순서임
- Introduction의 7B rank-16 예시를 다시 불러와 처음과 끝을 연결함
- 새로운 결과를 추가하지 않고, 이미 보인 결과를 "학습 전에 계산 가능한 판단 기준"으로 바꾸는 데 집중함
- 따로 Limitations 절이 없는 대신(AAAI 형식), 마지막 문단에서 하한값·랜덤 문자열이라는 방법 한계를 긍정적으로 재해석함
