# MI-Distillation — Abstract 문장 분해

출처: arXiv 2608.29623 (Model Interpolation Distillation), arXiv LaTeX 원본 `acl_latex.tex`

- (배경)
   - Recent advances in large reasoning models (LRMs) have shown strong performance on complex problems through long chain-of-thought (Long CoT) reasoning.
- (갭, 문제 현상)
   - However, distilling such trajectories into smaller student models remains challenging: direct Long CoT supervision often provides limited gains and can be less effective than concise Short CoT rationales.
- (연구 관점)
   - In this work, we investigate this phenomenon from a gradient-centric perspective.
- (분석 결과)
   - Our analysis shows that Long CoT induces larger gradient magnitudes and more concentrated update directions than Short CoT, with this effect becoming more pronounced as student model capacity increases.
- (해석, 원리)
   - These findings suggest that effective Long CoT distillation requires balancing the reasoning information density of reasoning trajectories with their distributional alignment to the student model.
- (방법 제안)
   - Motivated by this insight, we propose Model Interpolation Distillation (MI-Distillation), a framework that constructs a continuous Instruct-Reasoning data spectrum through model interpolation.
   - To select suitable trajectories from this spectrum, we further introduce Sequential Learnable Surprisal Score (SeqLSS), which favors reasoning paths that are both informative and learnable for the student.
- (실험 결과)
   - Extensive experiments on reasoning benchmarks show that MI-Distillation consistently improves small model CoT distillation over strong Long CoT baselines.

## 구조 메모

- 8문장, 약 180단어이며 전형적인 "방법 제안" 논문 초록임
- 흐름: 배경 → 현상(갭) → 분석 관점 → 분석 결과 → 원리 → 방법 2개 → 실험 결과
- 분석 결과를 방법의 동기로 삼는 구조임 ("Motivated by this insight")
   - 분석 논문인 우리 논문은 이 부분(분석 → 원리)을 초록의 중심으로 가져오면 됨
- 원리 문장의 "information density … distributional alignment"가 우리 논문의 핵심 변수(밀도)와 직접 연결됨
- 결과 문장에 구체 수치가 없음
