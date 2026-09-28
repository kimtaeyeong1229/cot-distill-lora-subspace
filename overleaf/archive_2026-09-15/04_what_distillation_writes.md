# 4. What Does Distillation Write? (RQ1)

- 아래 bullet은 이전 뼈대에서 옮겨 온 초안 메모임. 영문 작성 시 `00_abstract.md`와 같은 형식(문장별 초안 / 영문 / 한문)으로 다시 씀
- `[TBD]`는 쌍 B·C·D와 3B student 결과가 들어오면 확정할 수치임
- 섹션 끝에 1–3문장 해석: 정확도 향상이 학습된 추론을 뜻하지 않을 수 있음

## 4.1 Transfer and Memorization Across the Teacher Spectrum

- teacher 스타일이 바뀌면 adapter에 써지는 정보의 성격이 바뀌었음
   - Qwen2.5-0.5B에서 암기 비율은 λ = 0의 0.09에서 λ = 1의 1.20까지 연속적으로 올랐음
   - held-out 이득은 λ ≈ 0.5 부근에서 양수에서 음수로 뒤집혔음
   - 암기 비율이 1을 넘는 조건에서는 처음 보는 문제의 teacher 풀이 예측이 학습 전보다 나빠졌음
- 흡수 비율 W/H는 조건 간 비교적 일정했음
   - 쌍 A에서 W/H는 약 0.38–0.61 범위에 있었음

## 4.2 The Most Accurate Rationales Are the Most Memorized

- 각 student에서 held-out 정확도가 가장 좋은 λ는 암기 비율이 가장 높은 쪽에 있었음 (쌍 A)
   - Qwen2.5-0.5B: 최고 정확도 λ0.8 (55.7%), 암기 비율 1.18
   - Llama-3.2-1B: 최고 정확도 λ1.0 (55.4%), 암기 비율 0.40 (student 안에서 최고 수준)
   - Qwen2.5-1.5B: 최고 정확도 λ1.0 (83.8%), 암기 비율 1.09
   - OLMo-2-1B는 모든 λ가 base보다 정확도가 낮아 예외로 따로 서술 [TBD]
- 긴 CoT(λ0.0)는 흡수한 정보의 90% 이상이 처음 보는 문제로 넘어갔지만 정확도는 가장 낮았음 (암기 비율 0.06–0.09)
- [TBD: 쌍 B·C·D와 3B student에서 반전이 유지되는지 확인]
