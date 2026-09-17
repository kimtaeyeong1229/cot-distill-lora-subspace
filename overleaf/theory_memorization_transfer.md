# 암기와 전이의 정의: 이론 근거 조사 (2026-09-15)

세 갈래 조사(ML 정의, 인지심리·교육학, 기준을 뒤집는 틀)를 합친 기록임. 모든 서지는 조사 에이전트가 원문 페이지·DOI로 확인했고, [2차]는 2차 출처로만 확인한 인용임.

# 결정안 (사용자 확정 전)

- 정의의 층위
   - 이론 (정의): MDL / 정보이론. 학습 = 데이터의 규칙성을 압축하는 것 (Grünwald 2007). held-out 코드길이 감소 = 학습된 규칙성, train에서만 추가로 줄어든 bits = 표본 특이적 정보
   - 조작적 정의 (측정): 반사실 기억 (Feldman 2020, Zhang et al. NeurIPS 2023)의 M을 codelength(−log₂p)로 바꾼 것. bits 단위 형식화는 Morris et al. (ICML 2026)
   - 해석 어휘: Mayer (2002)의 retention vs transfer. rote learning = 파지는 되지만 전이가 안 되는 학습
- 용어
   - "이해(understanding)" 대신 "전이(transfer)"를 씀. held-out 감소에는 문체·형식 학습도 섞이고, Bloom의 Understand는 설명·해석을 요구하므로 "이해"는 과장임
   - 우리 held-out은 같은 teacher·같은 도메인이라 near transfer (Barnett & Ceci 2002). 벤치마크 정확도가 far transfer 쪽
- 삼각형(Bloom 피라미드)
   - 개정판 공저자 Anderson이 "The triangle does not appear anywhere in either Taxonomy"라고 밝힘. 피라미드 그림을 권위 근거로 쓰지 않음
   - 우리 측정: Remember ↔ train Δbits, 전이 ↔ held-out Δbits (분류상 Understand보다 Apply × Procedural에 가까움)
- 기준을 뒤집는 틀 (중심 논지)
   - "정확도는 이 분해를 식별하지 못한다(non-identifying)." 정확도는 학습된 규칙성, 표본 특이적 정보, 디코딩 예산을 섞음
   - "정확도가 틀렸다"고 쓰지 않음

# 발견별 이론 연결

| 발견 | 연결 이론 |
|---|---|
| 정확한 짧은 풀이가 대부분 암기 | Mayer 2002 rote learning (파지 O, 전이 X), Geirhos 2020 shortcut learning, Voita & Titov 2020 (정확도 대신 MDL) |
| 암기가 많을수록 실제 향상이 작음 | Bransford et al. 2000 ("learn with understanding rather than merely memorize"), Xu & Raginsky 2017 / Steinke & Zakynthinou 2020 (격차 = 훈련셋 정보) |
| 같은 풀이도 student마다 암기/전이가 다름, ρ가 결정 | Expertise reversal effect (Kalyuga et al. 2003), McNamara et al. 1996 (텍스트 명시성 × 사전 지식), Ausubel 1968 [2차], Borkar et al. 2026 (증류 전 perplexity로 암기 예측), Prashanth et al. ICLR 2025 (예측 가능성이 기억 유형을 가름) |
| ρ 방향: 이미 예측 가능한 풀이일수록 암기 | MDL 해석 (새 규칙성이 적어 표본 특이 정보만 남음), expertise reversal (전문가에게 worked example은 중복) |
| 긴 CoT는 전이는 크지만 정확도가 떨어짐 | Soderstrom & Bjork 2015 (learning ≠ performance), Schaeffer et al. 2023 (불연속 지표), Li et al. 2025 (정확도로만 설명) |

# 예상 리뷰 공격과 방어

1. "train–test loss gap일 뿐" (Yeom et al. CSF 2018)
   - 격차를 반사실 기억으로 보는 것은 Feldman 2020, Zhang et al. 2023이 확립한 정의. 기여는 격차 자체가 아니라 teacher 스타일 축의 분해와 ρ에 의한 예측
2. "codelength 감소는 스타일 모방" (Gudibande et al. ICLR 2024, Chu et al. ICML 2025)
   - held-out 감소를 이해가 아니라 전이로 한정. 형식/내용 토큰 분리, 벤치마크·섭동 정확도와의 수렴 타당도로 보강
3. "쉬운 풀이일수록 아낄 bits가 적다"
   - token scaling과 train−held-out 차분이 기저를 상쇄. ρ 효과는 Borkar, Prashanth의 독립 결과와 일치
4. "암기는 나쁜 게 아니다" (Feldman STOC 2020)
   - 한계 절에서 먼저 인정. "이 설정에서 암기 비중이 높을수록 정확도 이득이 작았다"는 경험적 관찰로 한정
5. "bits와 정확도는 원래 같이 간다" (Huang et al. COLM 2024)
   - bits가 정당한 척도라는 근거로 인용하고, 기여는 둘이 어긋나는 조건을 규명한 것으로 대비

# 주요 서지 (확인됨)

- ML
   - Feldman, STOC 2020, https://dl.acm.org/doi/10.1145/3357713.3384290
   - Feldman & Zhang, NeurIPS 2020
   - Zhang, Ippolito, et al., Counterfactual Memorization in Neural Language Models, NeurIPS 2023, https://arxiv.org/abs/2112.12938
   - Tirumala et al., Memorization Without Overfitting, NeurIPS 2022
   - Schwarzschild et al., Adversarial Compression, NeurIPS 2024, https://arxiv.org/abs/2404.15146
   - Prashanth et al., Recite, Reconstruct, Recollect, ICLR 2025, https://arxiv.org/abs/2406.17746
   - Yeom et al., IEEE CSF 2018, https://arxiv.org/abs/1709.01604
   - Wu et al., Reasoning or Reciting?, NAACL 2024, https://aclanthology.org/2024.naacl-long.102/
   - Xie et al., On Memorization of LLMs in Logical Reasoning, IJCNLP-AACL 2025, https://aclanthology.org/2025.ijcnlp-long.148/
   - Chu et al., SFT Memorizes, RL Generalizes, ICML 2025, https://arxiv.org/abs/2501.17161
   - Gudibande et al., The False Promise of Imitating Proprietary Language Models, ICLR 2024
   - Borkar et al., COLM 2026 채택 (OpenReview nlyPykGtRd, 우리 bib은 아직 arXiv)
- 정보이론 / MDL
   - Grünwald, The MDL Principle, MIT Press 2007 (튜토리얼 https://arxiv.org/abs/math/0406077)
   - Delétang et al., Language Modeling Is Compression, ICLR 2024
   - Blier & Ollivier, NeurIPS 2018, https://arxiv.org/abs/1802.07044
   - Voita & Titov, Information-Theoretic Probing with MDL, EMNLP 2020, https://aclanthology.org/2020.emnlp-main.14/
   - Huang et al., Compression Represents Intelligence Linearly, COLM 2024
   - Xu & Raginsky, NIPS 2017, https://arxiv.org/abs/1705.07809
   - Steinke & Zakynthinou, COLT 2020, https://proceedings.mlr.press/v125/steinke20a.html
   - Lotfi et al., Non-Vacuous Generalization Bounds for LLMs, ICML 2024 (SubLoRA)
   - Stanton et al., Does Knowledge Distillation Really Work?, NeurIPS 2021
   - Geirhos et al., Shortcut learning, Nature MI 2020; McCoy et al., ACL 2019; Schaeffer et al., NeurIPS 2023
- 교육학 / 인지심리
   - Mayer (2002), Rote versus meaningful learning, Theory Into Practice 41(4), doi:10.1207/s15430421tip4104_4
   - Anderson & Krathwohl (2001); Krathwohl (2002) doi:10.1207/s15430421tip4104_2
   - Barnett & Ceci (2002), Psychological Bulletin, doi:10.1037/0033-2909.128.4.612
   - Salomon & Perkins (1989), doi:10.1207/s15326985ep2402_1
   - Bransford, Brown & Cocking (2000), How People Learn, doi:10.17226/9853
   - Kalyuga et al. (2003), expertise reversal, doi:10.1207/S15326985EP3801_4
   - Sweller & Cooper (1985), worked example, doi:10.1207/s1532690xci0201_3
   - McNamara et al. (1996), doi:10.1207/s1532690xci1401_1
   - Soderstrom & Bjork (2015), learning vs performance, doi:10.1177/1745691615569000
   - Chi et al. (1989) self-explanation, doi:10.1207/s1532690xci0201_3 계열; Chi & Wylie (2014) ICAP, doi:10.1080/00461520.2014.965823
   - [2차] Ausubel (1968), Vygotsky (1978) p.86, Perkins (1998), Bloom (1956) 원문 문구

# 남은 작업 후보

- ZPD가 예측하는 ρ와 전이의 역U자 관계 확인
- 형식 토큰 vs 내용 토큰 분리 (스타일 모방 공격 대응)
- held-out Δbits와 벤치마크 정확도의 상관 (수렴 타당도)
- Borkar et al. bib를 COLM 2026으로 교체 (OpenReview BibTeX 필요)
