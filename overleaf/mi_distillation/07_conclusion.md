# 7. Conclusion — 문장 분해

출처: arXiv 2608.29623, `acl_latex.tex`

- (연구 관점)
   - In this work, we investigate Long CoT distillation for small student models from a gradient-centric perspective.
- (분석 결과)
   - Our analysis shows that Long CoT supervision induces larger and more concentrated gradient updates than Short CoT, with the effect becoming stronger as model capacity increases.
- (원리)
   - These results suggest that effective reasoning distillation requires balancing rationale informativeness with student learnability.
- (방법)
   - Based on this insight, we propose MI-Distillation, which constructs a continuous Instruct-Reasoning data spectrum via model interpolation, and SeqLSS, a student-aware criterion for selecting informative and learnable trajectories.
- (실험 결과와 의의)
   - Extensive experiments demonstrate that MI-Distillation consistently improves small-model CoT distillation over strong baselines, highlighting a practical path toward transferring reasoning abilities from LRMs to compact models.

## 구조 메모

- 1문단, 5문장이며 Abstract와 거의 같은 순서(관점 → 분석 → 원리 → 방법 → 결과)를 반복함
- 새로운 내용이나 향후 연구 방향은 넣지 않고, 한계는 Limitations로 넘김
