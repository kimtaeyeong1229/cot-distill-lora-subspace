# 암기/전이 판정의 타당성: 조사 결과와 해결안 (2026-09-15)

세 갈래 조사(로컬 자료 전수, 이론·선행, 원인 분해 방법)를 합친 기록. 서지는 조사 에이전트가 원문 페이지로 확인함.

# Q1. 실제 데이터에서 "학습용에서만 줄어든 bits = 암기"라고 말할 수 있나

## 확인된 사실
- How Many Bits (Tan et al., arXiv 2607.21351)
   - 저자 인정: "On ordinary text that figure tangles memorization together with understanding" / mem_U on random data "ties both to a ground truth that neither has on real text"
   - 실제 텍스트에서 ΔW가 뚜렷하게 잡힌 건 무작위 비밀을 심었을 때뿐. 정답률 100% kv SFT에서 ΔW = −368 bits (Table table3_b4)
   - appendix(supplement)는 로컬에 없음
- Morris et al. (ICML 2026)
   - 정의: unintended memorization = 참 생성 분포(reference) 대비 학습 모델이 추가로 줄인 bits. Def. 3 "Let θ be a reference model that approximates the true distribution of data"
   - 합성 데이터는 "exact underlying data distribution"을 reference로, 실제 텍스트는 같은 계열 큰 모델로 근사
   - 경고: likelihood는 "dependent on the parameters of the decoding algorithm, such as temperature or top-k sampling"
- 정보이론적 일반화 (Xu & Raginsky 2017, Steinke & Zakynthinou 2020, Aminian et al. 2021)
   - 로그 손실의 train–test 간격은 학습 데이터에 대한 정보량의 기댓값 상한과 관련될 뿐, 같지 않음
- 반사실 암기 (Feldman STOC 2020, Zhang et al. NeurIPS 2023): 포함 vs 제외 비교가 실제 데이터에서의 표준 조작적 정의
- Achille & Soatto (JMLR 2018): 교차엔트로피 분해에서 참 엔트로피 아래로 내려간 학습 손실은 가중치의 암기 항만이 만들 수 있음

## 판정
- ΔW (학습용 − 숨겨 둔): "train-specific excess compression" (반사실 암기 / 일반화 간격)으로는 방어 가능. "bits memorized"로 부르면 안 됨
- 생성기 하한 (teacher 샘플링 분포의 self-information)
   - 원리는 확립(Morris memU + Achille & Soatto), 적용은 선행 미발견 → 새 기여 후보
   - 조건: 집합 평균에서만 유효 / 정답 필터링 보정 −log₂π(x) / temperature 0.6·top-p 0.95 재정규화 확률 / 문자열 단위 bits(토크나이저) / 생성 때와 같은 템플릿·λ teacher·정밀도
   - 검증: 숨겨 둔 풀이의 평균 codelength는 하한 아래로 내려가면 안 됨
   - 성질: 하한 아래 부분만 세므로 보수적 하한. 하한 위에서 생긴 표본 특이 적합은 놓침 → ΔW와 함께 보고
- 기존 결과 단서
   - epoch 17 λ1 (Qwen-0.5B, n375): 학습 풀이 codelength ≈ 0.001 bits/token → 어떤 teacher 샘플링 엔트로피보다 낮을 가능성이 큼 → 엄밀한 암기 사례 후보
   - canary 3회 노출: 학습 코드도 진짜 정보량(331 bits) 아래로 안 내려감 → 하한 방식은 보수적임을 보여 줌

# Q2. 줄어든 이유를 어떻게 가르나

선행: fine-tuning 전후 per-token codelength 변화를 train/held로 나눠 범주별로 분해한 연구는 찾지 못함 (방법 재료는 있음)

## 숨겨 둔 문제에서 줄어든 이유 (내용 vs 말투·형식)
- P1 토큰 범주별 Δbits: 숫자·수식(bridging objects) vs 나머지 문장 틀(language templates) (Wang et al. ACL 2023), 명사·숫자 우선 암기(Tirumala et al. NeurIPS 2022), 범주 선례(Just et al. 2026), 순위 변화(URIAL, Lin et al. ICLR 2024), LoRA는 응답 시작·스타일 토큰 위주 학습(Ghosh et al. ICML 2024)
- P2 다른 문제 풀이 대조: 문제 i에 문제 j 풀이를 붙여 측정. matched − mismatched = 문제에 조건부인 내용 이득, mismatched 이득 = 스타일·도메인
- P3 숫자 치환·단계 섞기 (Li et al. Findings EMNLP 2025, GSM-Symbolic): 학습 전후 margin 차이(이중 차분)

## 학습용에서만 줄어든 이유 (표본 암기 vs 문장 패턴 과적합)
- T1 teacher 재샘플: 학습 문제를 같은 teacher로 다시 생성. 학습 원본 ≫ 재샘플 ≈ 숨겨 둔 → 표본 암기 / 재샘플 > 숨겨 둔 → 문제별 내용 기억
- T2 n-gram 층화 (Prashanth et al. ICLR 2025 recite/reconstruct/recollect)
- T3 추출 테스트 확대 (Carlini et al. ICLR 2023, Ippolito et al. INLG 2023)
- 생성기 하한 아래로 내려간 부분 = 엄밀한 암기

## 주의
- 토크나이저: 범주는 문자 span으로 정의, 계열 간 비교는 문제당·문자당 bits
- 'wait' 같은 반성 어휘는 추론 분기점(Wang et al. 2025)일 수 있어 "말투"로 단정 금지, 어휘 목록 사전 등록
- 길이 교란, 섭동 텍스트의 분포 밖 효과(반드시 전후 차이로), prompt 토큰 제외

# 필요한 코드 변경
- measure_bits.py: 토큰별 H_before/H_after 저장
- teacher 샘플링 확률 계산 스크립트 (재정규화 확률, 필터링 보정용 π(x) 추정)
- 대조 데이터 생성: mismatched, 숫자 치환, teacher 재샘플
