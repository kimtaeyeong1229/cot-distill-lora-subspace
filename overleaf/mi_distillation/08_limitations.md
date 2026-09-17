# Limitations — 문장 분해

출처: arXiv 2608.29623, `acl_latex.tex`

- (도입)
   - While MI-Distillation with SeqLSS shows consistent benefits, several limitations remain.
- (한계 1: 도메인)
   - First, our experiments mainly focus on mathematical reasoning benchmarks, where final-answer verification is relatively reliable.
   - Extending MI-Distillation to open-ended reasoning, code generation, multilingual tasks, and domains with weaker automatic verification remains an important direction for future work.
- (한계 2: 비용)
   - Second, SeqLSS requires scoring multiple candidate rationales with the student model, introducing additional offline data-construction cost compared with distillation from a single teacher.
   - While this cost is incurred before training, more efficient scoring or approximation strategies would further improve scalability to larger corpora and broader teacher spectra.
- (한계 3: 지표가 담지 못하는 것)
   - Finally, SeqLSS estimates student-side learnability through token-level likelihoods, which may not fully capture higher-level reasoning properties such as logical faithfulness, conciseness, robustness, or step-level correctness.
   - Future work could integrate learnability-aware selection with process-level verification, step-wise feedback, or stronger rationale-quality assessment.

## 구조 메모

- ACL 필수 Limitations 절의 전형적 형태: 도입 1문장 + "First / Second / Finally" 한계 3개
- 한계마다 "한계 문장 → 향후 방향 문장" 두 문장씩 짝을 이룸
- 한계를 결과를 약하게 만드는 쪽이 아니라 확장 가능성(도메인, 비용, 지표)으로 씀
- 약 170단어로 짧음
