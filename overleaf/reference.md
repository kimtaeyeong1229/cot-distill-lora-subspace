# References

- 인용 규칙: 원문 PDF가 `materials/references/<bibkey>.pdf`에 있는 문헌만 인용함. PDF가 없으면 인용 불가
- BibTeX는 `references.bib`에 있음. ACL Anthology 게재 논문은 Anthology 공식 BibTeX를 그대로 가져왔고(키만 변경), 나머지는 arXiv 초록 페이지 메타데이터로 만들었음
- 저자 검증: 원문 PDF 1–2쪽에서 모든 저자의 이름(given name 포함)과 순서를 대조했음
- 게재본 우선: 학회·저널 게재가 확인된 문헌은 게재본 PDF와 서지를 씀

## 서지 결정 규칙 (2026-09-14 사용자 승인)

- 판본: 게재본을 공개 출처로 확인할 수 있으면 게재본(ACL Anthology, 공식 proceedings, 저널, PDF 1쪽 "Published in" 표기, 사용자가 확인한 OpenReview). 확인할 수 없으면 arXiv
- 저자 표기: arXiv 기준. PDF로 확인되는 명백한 메타데이터 오류(가짜 저자 ":", 한 사람이 둘로 쪼개진 이름)만 고침. PDF에 원래 있는 중복은 유지
- 제목·연도: 인용 판본 기준. arXiv는 export와 같게 최신 판 연도
- 게재본과 arXiv 표기가 다르면: venue·쪽수·DOI는 게재본, 저자 표기는 arXiv
- bibkey는 본문에 인용한 뒤에는 바꾸지 않음 (예: hu2021lora는 ICLR 2022 판이지만 키 유지)

## 목록

| bibkey | 제목 | 저자 | 인용 판본 | arXiv | 게재본 | 검증 | 인용 위치 |
|---|---|---|---|---|---|---|---|
| hu2021lora | LoRA: Low-Rank Adaptation of Large Language Models | Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen (8) | ICLR 2022 | https://arxiv.org/abs/2106.09685 | https://openreview.net/forum?id=nZeVKeeFYf9 (사용자가 OpenReview에서 확인) | 저자 표기는 arXiv 기준 (OpenReview는 "yelong shen" 소문자) | Intro 문단 1 |
| guo2025deepseekr1 | DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning | Guo, Yang, Zhang, … Zhang (194) | Nature 645:633–638 (2025) | https://arxiv.org/abs/2501.12948 | https://doi.org/10.1038/s41586-025-09422-z (open access) | Nature 페이지 메타데이터 194명과 PDF 저자 목록의 이름·순서 일치 | Intro 문단 1 |
| li2025small | Small Models Struggle to Learn from Strong Reasoners | Li, Yue, Xu, Jiang, Niu, Lin, Ramasubramanian, Poovendran (8) | Findings of ACL 2025 | https://arxiv.org/abs/2502.12143 | https://aclanthology.org/2025.findings-acl.1301 | Anthology 저자 목록 = arXiv 저자 목록 = PDF | Intro 문단 1, 2 |
| jiang2025teach | Teach Small Models to Reason by Curriculum Distillation | Jiang, Lu, Lin, Han, Sun (5) | EMNLP 2025 main | 없음(arXiv 미게재) | https://aclanthology.org/2025.emnlp-main.376 | Anthology 저자 = PDF | Intro 문단 2 |
| dai2025capture | Capture the Key in Reasoning to Enhance CoT Distillation Generalization | Dai, Li, Zhou, Hu (4) | ACL 2025 long | https://arxiv.org/abs/2405.19737 (같은 연구의 2024년 preprint v1. 제목만 "Beyond Imitation: Learning Key Reasoning Steps from Dual Chain-of-Thoughts in Reasoning Distillation"이고, 저자 4명·순서·EDIT 방법·초록이 게재본과 같음. arXiv는 게재 후 갱신되지 않음) | https://aclanthology.org/2025.acl-long.21 | Anthology 저자 = PDF | Intro 문단 3 |
| yang2026reasoning | Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment | Yang, Lai, Zhao, … Huang (16) | ACL 2026 long | https://arxiv.org/abs/2601.14249 | https://aclanthology.org/2026.acl-long.1950 | Anthology 저자 목록 = arXiv 저자 목록 = PDF | Intro 문단 2 |
| lan2026midistillation | MI-Distillation: Selecting from Model-Interpolated Instruct-Reasoning Data Spectrum for Chain-of-Thought Distillation | Lan, Hu, Zheng, … (7) | arXiv | https://arxiv.org/abs/2608.29623 | arXiv 주석상 Findings of EMNLP 2026 accept, Anthology 미등재 | arXiv 저자 = PDF | Intro 문단 2, 4 |
| just2026signal | The Signal is in the Steps: Local Scoring for Reasoning Data Selection | Just, Ko, Jia (3) | arXiv (Preprint, v2 2026) | https://arxiv.org/abs/2510.03988 | 없음 | arXiv 저자 = PDF. 연도는 arXiv BibTeX export와 같게 최신 판(v2, 2026)을 따름 | Intro 문단 2 |
| panigrahi2025grace | In Good GRACEs: Principled Teacher Selection for Knowledge Distillation | Panigrahi, Liu, Malladi, Kakade, Goel (5) | ICLR 2026 | https://arxiv.org/abs/2511.02833 | https://openreview.net/forum?id=m276fke38H (사용자가 OpenReview BibTeX 제공) | 저자 표기는 arXiv 기준 (OpenReview는 "Sham M. Kakade") | Intro 문단 2 |
| cui2026whole | "The Whole Is Greater Than the Sum of Its Parts": A Compatibility-Aware Multi-Teacher CoT Distillation Framework | Cui, Guo, Yang, Lu, Zhou, Xu, Song, Zhao, Ren (9) | arXiv | https://arxiv.org/abs/2601.13992 | 게재본 찾지 못함 | arXiv 저자 = PDF | Intro 문단 3 |
| morris2025memorize | How much can language models memorize? | Morris, Sitawarin, Guo, Kokhlikyan, Suh, Rush, Chaudhuri, Mahloujifar (8) | ICML 2026 | https://arxiv.org/abs/2505.24832 | https://openreview.net/forum?id=bA6BgSbaUi (사용자가 OpenReview BibTeX 제공, ICML 2026 oral) | 제목은 게재본 기준 (arXiv는 "do", ICML은 "can"). 저자 표기는 arXiv 기준 (OpenReview는 "John Xavier Morris") | Intro 문단 3, 4 · Related Work 2.2 · Sec 3.2 |
| tan2026bits | How Many Bits Can an Adapter Write? Measuring the Capacity and Memorization of Parameter-Efficient Fine-Tuning | Tan, Du, Feng (3) | arXiv | https://arxiv.org/abs/2607.21351 | 없음 | arXiv 저자 = PDF | Intro 문단 3, 4 |
| kajitsuka2026capacity | Revisiting the Capacity Gap in Chain-of-Thought Distillation from a Practical Perspective | Kajitsuka, Honda, Takase (3) | arXiv | https://arxiv.org/abs/2604.08880 | 게재본 찾지 못함 | arXiv 저자 = PDF | Related Work 2.1 |
| li2026onpolicy | Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe | Li, Zuo, He, Zhang, Xiao, Qian, Yu, Gao, Yang, Liu, Ding (11) | arXiv (v2) | https://arxiv.org/abs/2604.13016 | 게재본 찾지 못함 | arXiv 저자 = PDF | Related Work 2.1 |
| borkar2026memorization | Memorization Dynamics in Knowledge Distillation for Language Models | Borkar, Chadha, Mireshghallah, Zhang, Veliche, Mitra, Smith, Xu, Garcia-Olano (9) | arXiv (v2) | https://arxiv.org/abs/2601.15394 | 게재본 찾지 못함 | arXiv 저자 = PDF | Related Work 2.3 |
| singh2025teacher | From Teacher to Student: Tracking Memorization Through Model Distillation | Singh (1) | L2M2 workshop @ ACL 2025 | https://arxiv.org/abs/2506.16170 | https://aclanthology.org/2025.l2m2-1.6 | Anthology 저자 = PDF | Related Work 2.3 |
| bossy2026mitigating | Mitigating Unintended Memorization with LoRA in Federated Learning for LLMs | Bossy, Vignoud, Rabbani, Troncoso Pastoriza, Jaggi (5) | TMLR 2026 | https://arxiv.org/abs/2502.05087 | https://openreview.net/forum?id=WKPyZnLIW4 (사용자가 OpenReview BibTeX 제공, ISSN 2835-8856). ICML 2025 workshop 판(XeHKCwAITc)은 비아카이브라 쓰지 않음. How Many Bits도 TMLR로 인용 | 저자 표기는 arXiv 기준 (OpenReview는 "Julien Tuấn Tú Vignoud") | Related Work 2.3 |
| carlini2019secret | The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks | Carlini, Liu, Erlingsson, Kos, Song (5) | USENIX Security 2019 | https://arxiv.org/abs/1802.08232 | https://www.usenix.org/conference/usenixsecurity19/presentation/carlini | USENIX 공식 BibTeX = PDF(게재본) | Related Work 2.2 |
| carlini2023quantifying | Quantifying Memorization Across Neural Language Models | Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang (6) | ICLR 2023 | https://arxiv.org/abs/2202.07646 | https://openreview.net/forum?id=TatRHT_1cK (사용자 확인) | 저자 표기는 arXiv 기준 (Tramer, 악센트 없음) | Related Work 2.2 |
| hendrycks2021math | Measuring Mathematical Problem Solving With the MATH Dataset | Hendrycks, Burns, Kadavath, Arora, Basart, Tang, Song, Steinhardt (8) | NeurIPS 2021 Datasets and Benchmarks | https://arxiv.org/abs/2103.03874 | https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html | NeurIPS 공식 BibTeX = 게재본 PDF | Sec 3.1 |
| lightman2023verify | Let's Verify Step by Step | Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe (10) | ICLR 2024 (MATH-500 출처) | https://arxiv.org/abs/2305.20050 | https://openreview.net/forum?id=v8L0pN6EOi (사용자 확인) | 저자 표기는 arXiv 기준 (OpenReview는 Yuri Burda, Harrison Edwards) | Sec 3.1 |
| cobbe2021training | Training Verifiers to Solve Math Word Problems | Cobbe, Kosaraju, … (12) | arXiv (GSM8K) | https://arxiv.org/abs/2110.14168 | 없음 | arXiv 저자 = PDF | Sec 3.3 |
| lewkowycz2022solving | Solving Quantitative Reasoning Problems with Language Models | Lewkowycz, Andreassen, … (14) | NeurIPS 2022 (Minerva) | https://arxiv.org/abs/2206.14858 | https://proceedings.neurips.cc/paper_files/paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html | NeurIPS 공식 BibTeX = 게재본 PDF | Sec 3.3 |
| he2024olympiadbench | OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems | He, Luo, … (14) | ACL 2024 long | https://arxiv.org/abs/2402.14008 | https://aclanthology.org/2024.acl-long.211 | Anthology 서지 사용, 5번째 저자는 arXiv·PDF 표기 "Zhen Leng Thai" (Anthology는 "Zhen Thai") | Sec 3.3 |
| rein2023gpqa | GPQA: A Graduate-Level Google-Proof Q&A Benchmark | Rein, Hou, … (8) | COLM 2024 | https://arxiv.org/abs/2311.12022 | https://openreview.net/forum?id=Ti67584b98 (사용자 확인) | arXiv 저자 = OpenReview = PDF | Sec 3.3 |
| qwen2025qwen25 | Qwen2.5 Technical Report | Qwen, Yang, Yang, … (팀명 + 42명) | arXiv | https://arxiv.org/abs/2412.15115 | 없음 | arXiv 목록에서 메타데이터 오류 ":"만 뺌, 팀명 "Qwen"은 유지 | Sec 3.3 |
| yang2024qwen25math | Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement | Yang, Zhang, … (16) | arXiv | https://arxiv.org/abs/2409.12122 | 없음 | arXiv 저자 = PDF | Sec 3.1 |
| yang2025qwen3 | Qwen3 Technical Report | Yang, Li, … (60) | arXiv | https://arxiv.org/abs/2505.09388 | 없음 | arXiv 저자 = PDF | Sec 3.1 |
| grattafiori2024llama3 | The Llama 3 Herd of Models | Grattafiori, Dubey, … (559) | arXiv | https://arxiv.org/abs/2407.21783 | 없음 | arXiv 목록에서 한 사람이 둘로 쪼개진 2건을 PDF대로 합침 (Guangyi (Jack) Zhang, Yu (Sid) Wang). PDF에 원래 두 번 나오는 Shaun Lindsay, Shuqiang Zhang은 유지 | Sec 3.1 (Llama-3.1), 3.3 (Llama-3.2, meta2024llama32와 함께) |
| olmo2025olmo2 | 2 OLMo 2 Furious | Team OLMo, Walsh, Soldaini, … (43) | arXiv v3 (32B 포함 전체판) | https://arxiv.org/abs/2501.00656 | 짧은 판이 COLM 2025 게재(arXiv 주석) | arXiv 저자 = PDF | Sec 3.3 |
| meta2024llama32 | Llama 3.2: Revolutionizing edge AI and vision with open, customizable models | Meta AI (기관) | 블로그 (2024-09-25) | 없음 | https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/ | 페이지를 PDF로 저장. Llama-3.2를 쓴 ACL 2026 논문 2편이 이 글을 인용 | Sec 3.3 |
| zhang2024aime24 | American Invitational Mathematics Examination (AIME) 2024 | Zhang, Math-AI Team | 데이터셋 (HF math-ai/aime24) | 없음 | https://huggingface.co/datasets/math-ai/aime24 | 우리가 평가에 쓰는 데이터(파일 크기 일치). 데이터셋 카드의 인용 형식을 따름, MI-Distillation과 같음. 카드 페이지를 PDF로 저장 | Sec 3.3 |
| maa2023amc | American Mathematics Competitions (AMC) 2023 | Mathematical Association of America (기관) | 웹페이지 | 없음 | https://maa.org/student-programs/amc/ | 데이터는 HF math-ai/amc23이지만 카드에 인용 형식이 없어 MI-Distillation처럼 MAA 대회 페이지를 인용. 페이지를 PDF로 저장 | Sec 3.3 |

## 참고

- guo2025deepseekr1은 저자 194명이라 참고문헌 지면을 많이 차지함. `and others` 금지 규칙 때문에 줄일 수 없으므로, 분량이 문제가 되면 인용 자체를 빼는 방향으로 판단함
- QwQ-32B는 논문 없이 블로그로만 공개되어 PDF가 없으므로 인용하지 않음(Introduction에서 이름도 뺐음)
- AMC23, AIME24는 MI-Distillation처럼 인용함 (AIME24는 데이터셋 카드 인용 형식, AMC23은 MAA 페이지)
- AdamW(Loshchilov & Hutter)는 인용하지 않음. AdamW를 쓴 ACL 2026 논문 6편 중 1편만 인용함
- 논문이 없는 모델·데이터(Llama 3.2, AIME24, AMC23)는 웹페이지를 PDF로 저장해 PDF 규칙을 지킴
- 참고문헌 목록에서 저자 20명이 넘는 항목은 ACL 스타일이 자동으로 "and N others"로 줄여 표시함(bib에는 전체 저자가 있음)
- 게재본 "찾지 못함"과 "확인불가"는 추가 확인 대상임. 게재가 확인되면 게재본 PDF와 서지로 교체함
