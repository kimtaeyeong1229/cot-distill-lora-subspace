# 방어 가능한 방법론 (초안, 2026-09-16)

원칙: 각 측정은 선행 연구의 정의를 그대로 따르고, 정의가 허락하는 말만 주장함. 해석이 필요한 부분은 정답을 아는 대조군이나 섭동 대조군으로 검증함.

# 0. 설정

- teacher 쌍 4개, λ ∈ {0, .2, .4, .6, .8, 1}, 가중치 보간 (Lan et al. 2026; Wu et al. 2025b)
- λ마다 teacher가 MATH 문제 4,000개를 temperature 0.6, top-p 0.95로 생성함 → 이 샘플러가 데이터의 참 생성기 G_λ
- 모든 λ가 맞힌 문제를 무작위로 학습용 S(약 2,100)와 숨겨 둔 H(300)로 나눔
- student θ₀에 LoRA를 붙여 λ마다 S로 SFT → θ_A

# 1. 측정 단위: codelength

- 정의: L_θ(y|x) = −log₂ p_θ(y|x), 풀이 토큰 + EOS, 문제 x는 조건
- 근거: 확률 모델은 산술 부호화로 이 길이의 무손실 압축기가 됨 (Shannon 1948; Delétang et al. ICLR 2024)
- 주장 범위: "모델 θ가 이 풀이를 적는 데 드는 bits". 풀이의 진짜 정보량이 아님

# 2. 세 가지 양과 각각의 근거

## 2.1 학습이 바꾼 예측: W
- W(D) = Σ [L_θ₀(y|x) − L_θA(y|x)]
- 근거: behavior-write bits (Tan et al. 2026, Eq. 2), 기준 모델 대비 codelength 절감 (Morris et al. ICML 2026)
- 주장: "학습으로 student의 예측이 이 풀이들에 대해 몇 bits 좋아졌나". 흡수한 정보량이라고 주장하지 않음

## 2.2 학습용에만 생긴 초과 압축: ΔW (반사실 기준)
- ΔW = W(S) − (T_S/T_H)·W(H)
- 근거
   - 반사실 암기: 포함했을 때 − 제외했을 때 (Feldman STOC 2020; Zhang et al. NeurIPS 2023)
   - 무작위 분할이므로 E[L_θ₀] 가 S와 H에서 같음 → ΔW는 학습 후 로그 손실의 train–held 간격 (일반화 간격; Xu & Raginsky 2017; Steinke & Zakynthinou 2020)
   - 같은 형태의 초과량 정의 (Tan et al. 2026 ΔW)
- 주장: "학습용 표본에만 특이하게 맞춰진 정도(sample-specific fit)". 명칭은 train-specific excess compression
- 주장하지 않음: 간격 = 외운 정보량 (정보이론적으로 상한 관계일 뿐)
- 보고: 분할 재추출 여러 번의 평균과 분산 (Zhang et al.은 기댓값으로 정의하므로 단일 분할은 표본 1개)

## 2.3 엄밀한 암기의 하한: 생성기 기준 M_G
- 생성기 코드길이 L_G(y|x) = −log₂ p_G(y|x): 실제 샘플링 분포(temperature 0.6 재정규화, top-p 절단)의 확률, 정답 필터링 보정 + log₂π(x)
- M_G(D) = max(0, mean_D[L_G] − mean_D[L_θA]) × |D| (집합 평균 기준)
- 근거
   - Morris et al.: 비의도적 암기 = 참 생성 분포 대비 추가로 줄인 bits. 생성 분포를 알 때는 그것을 reference로 씀 (합성 데이터 설정)
   - Gibbs 부등식: 어떤 모델도 기댓값으로 참 생성기의 교차엔트로피보다 짧게 적을 수 없음
   - Achille & Soatto (JMLR 2018) 분해: 참 엔트로피 아래로 내려간 학습 손실은 가중치의 암기 항만이 만듦
- 검증 조건: mean_H[L_θA] ≥ mean_H[L_G] (숨겨 둔 풀이는 하한을 넘지 않아야 함; Tan et al.의 "held-out mem_U ≈ 0" 점검과 같은 구조)
- 주장: "적어도 이만큼은 표본 특이 정보가 adapter에 들어갔다" (보수적 하한)
- 한계: 하한 위에서 생긴 표본 특이 적합은 세지 못함 → ΔW와 함께 보고

# 3. 정답을 아는 대조군 (측정기 보정)

- 반복 canary: 무작위 64자 코드 100개를 1, 4, 16회 반복 삽입 + 삽입하지 않은 코드 100개
   - 진짜 정보량을 앎 (코드당 331 bits) → M_G와 ΔW가 반복 횟수에 따라 암기를 잡는지, 삽입 안 한 코드는 0인지 확인
   - 근거: random-string 보정 (Morris et al.; Tan et al.), secret sharer (Carlini et al. USENIX Security 2019)
- 수렴 타당도: 추출 테스트 (Carlini et al. ICLR 2023; Ippolito et al. INLG 2023)에서 학습용 vs 숨겨 둔 풀이의 그대로 이어 쓰기 비율이 ΔW, M_G와 같은 방향인지

# 4. 원인 분해 (왜 줄었나)

## 4.1 숨겨 둔 풀이의 이득: 문제 조건부 내용 vs 스타일·형식
- 짝 바꾸기 대조: 문제 i에 다른 문제 j의 풀이를 붙여 W를 잼
   - W_match − W_mismatch = 문제에 조건부인 이득 (내용)
   - W_mismatch = 문제와 무관한 이득 (스타일·형식·도메인)
   - 근거: 조건부 정보 = 조건을 줬을 때와 안 줬을 때의 codelength 차이 (정보이론의 조건부 상호정보량 정의)
- 토큰 범주 분해: 숫자·수식(bridging objects) vs 문장 틀(language templates) (Wang et al. ACL 2023); LoRA가 스타일 토큰 위주로 학습한다는 가설 (Ghosh et al. ICML 2024; Lin et al. ICLR 2024 URIAL)
- 섭동 이중 차분: 숫자 치환·단계 섞기한 풀이의 codelength margin이 학습 전후로 커지는지 (Li et al. Findings EMNLP 2025; Mirzadeh et al. ICLR 2025)

## 4.2 학습용에만 생긴 이득: 표본(문장) 암기 vs 문제 내용 기억
- teacher 재샘플 대조: 학습 문제를 같은 G_λ로 새로 생성한 풀이 S′
   - L(S) ≪ L(S′) ≈ L(H): 그 문장 자체를 외움 (verbatim)
   - L(S′) < L(H): 문제별 내용을 기억
   - 근거: 반사실 암기의 내용/형식 분리 제안 (Zhang et al. 2023, paraphrase 논의), 같은 생성기 재샘플은 스타일을 고정한 채 문장만 바꿈
- n-gram 층화: 표본 고유 구간 vs 학습 전체에 흔한 구간 (Prashanth et al. ICLR 2025 recite/reconstruct/recollect)

# 5. 교육학 해석 (해석 어휘로만)

- ΔW, M_G ↔ 파지(retention) 검사에서만 드러나는 학습, W(H)·W_match−W_mismatch ↔ 전이(transfer) 검사 (Mayer 2002)
- 우리 전이는 같은 teacher·같은 도메인 = 근거리 전이 (Barnett & Ceci 2002). 벤치마크 정답률이 원거리 쪽
- "이해(understanding)"라는 용어는 쓰지 않음

# 6. 논문에서 쓸 주장과 쓰지 않을 주장

- 씀
   - "train-specific excess compression" (ΔW), "generator-floor memorization lower bound" (M_G), "problem-conditional transfer" (W_match − W_mismatch)
   - "정답률은 이 분해를 식별하지 못한다" (Voita & Titov EMNLP 2020의 MDL 대체 선례)
- 쓰지 않음
   - 풀이의 진짜 정보량, "bits memorized"(M_G 하한 외), "이해/추론을 배웠다", 인과
