# 5. From Bits to Leakage — 문장 분해

출처: arXiv 2607.21351v1, `sections/5_leakage.tex` (그림 캡션 제외)

## 문단 1 — 질문과 설정

- (동기)
   - A number measured on random strings earns its keep only if it predicts what a privacy audit would find on real text.
- (연구 질문과 설정)
   - So we plant secrets in a document corpus, fine-tune Qwen2.5-0.5B (Qwen Team 2024) on it, and ask whether the bits we measure line up with what an attacker can actually pull out.

## 문단 2 — 공격 실험 설계

- (방법)
   - The probe is the secret-sharer construction (Carlini et al. 2019): short random secrets set into a carrier corpus at controlled repetition, so that exposure—the log-rank of the true secret against ten thousand decoys—carries a known scale and a clean ceiling at rank one.
- (설계 이유)
   - Repeating a secret a handful of times, rather than once, lets us watch memorization arrive by degrees instead of as a threshold, and a held-out set matched in size keeps the behavior-bit accounting honest.
- (설계의 성격)
   - It is a deliberately favorable setting for the attacker; that is the point, since we want to know whether measured capacity tracks leakage when leakage is there to be found.

## 문단 3 — rank 실험

- (결과 요약)
   - The clearest read is a tale of two steps up the rank ladder (Figure 4).
- (결과 1)
   - Going from rank 4 to rank 16 quadruples the trainable parameters but barely changes the bits the adapter writes—and exposure and membership inference barely move.
- (결과 2)
   - Going from rank 16 to rank 64 quadruples the count again, but this time the written bits jump by half, and the secrets go with them: exposure climbs eleven bits, and the frequently repeated ones become extractable outright, ranked first among ten thousand candidates.
- (해석)
   - The step that writes is the step that leaks.
   - Parameter count, which rose the same way at both steps, does not tell the two apart.

## 문단 4 — 위치 실험으로 교란 제거

- (교란 요인 제거)
   - Because rank moves bits and parameters together, we break the tie with a placement test that holds the count fixed.
- (결과)
   - Three Qwen adapters of about 1.65M parameters each—rank chosen per placement to match—write different amounts by virtue of where they sit, and their canary exposure sorts the same way: the placement that writes the most bits leaks the most, the leanest the least, with the count held constant throughout.
- (해석)
   - Leakage is tracking what the adapter stored, not how large it is.
- (한계와 보강)
   - (This is four planted-secret configurations plus the placement triple, not a scaling study; the full regression, with every predictor and a leave-one-out check, is in the supplement, and it neither adds nor removes the effect.)

## 문단 5 — 작동하지 않은 공격

- (부정적 결과 예고)
   - One attack declines to cooperate, and it is worth saying why plainly.
- (결과)
   - Loss-based membership inference sits at chance across the entire rank sweep, and a stronger calibrated attack (Min-K%) does no better.
- (해석)
   - At this corpus size, membership inference simply does not resolve the difference between these adapters—extraction does.
- (주장 범위 한정)
   - The bits an adapter writes predict what can be pulled back out of it verbatim; whether they also predict membership at larger scale is a question our corpus is too small to answer.

## 문단 6 — 실무적 함의

- (함의)
   - For anyone who has to share the result, the reading is practical.
   - The rank that protects privacy is not the smallest the budget allows but the smallest that still writes the data in—and the placement test adds that the same protection can come from where the update sits rather than how large it is.
- (마무리)
   - These are knobs a practitioner already turns for accuracy; the measurement gives a second reason to turn them, and a number to turn them against.

## 구조 메모

- 첫 문장이 절의 존재 이유를 말함: 랜덤 문자열에서 잰 값은 실제 텍스트 유출을 예측할 때만 의미가 있음
- 설계 문단에서 "공격자에게 유리한 설정"임을 스스로 밝히고 그 이유를 댐
- 교란 요인(rank는 bits와 파라미터를 함께 바꿈)을 짚고, 파라미터를 고정한 위치 실험으로 분리함
- 작동하지 않은 공격(MIA)을 숨기지 않고 한 문단으로 설명하고, 주장 범위를 "verbatim 추출"로 좁힘
- 괄호 문장으로 표본 수가 작다는 한계와 보조 분석 위치를 본문에 짧게 남김
- 마지막 문단은 실무자가 이미 조정하는 손잡이(rank, 위치)에 연결해 끝냄
