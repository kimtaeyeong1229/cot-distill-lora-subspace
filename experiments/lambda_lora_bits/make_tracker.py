#!/usr/bin/env python3
"""Build the experiment workbook: one side to enter values, one side to watch progress.

Sheets
  진행현황       stage counts and a student x pair completion matrix, driven by COUNTIFS
  분석_전환점    headline table: teacher transition lambda vs adapter divergence lambda, 18 rows
  분석_phi       phi(lambda0, lambda) and the seed baseline, lambdas as columns, 18 rows
  분석_정답률    held-out 300 and MATH-500 per lambda, averaged over seeds by formula, 18 rows
  입력_Teacher   18 interpolated teachers: generation stats and reasoning-format columns
  입력_Adapter   324 cells (3 pairs x 6 students x 6 lambdas x 3 seeds), one row per trained adapter

Only the "done" flags are filled from disk; measurement columns stay empty until runs land.
"""

import argparse
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

EXP = Path(__file__).resolve().parent
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
SEEDS = [0, 1, 2]

PAIRS = [
    ("A", "R1-Distill-Qwen-1.5B", "Qwen2.5-Math-1.5B-Instruct", "1.5B"),
    ("B", "Qwen3-4B-Thinking-2507", "Qwen3-4B-Instruct-2507", "4B"),
    ("D", "R1-Distill-Llama-8B", "Llama-3.1-8B-Instruct", "8B"),
]
STUDENTS = [
    ("Qwen2.5-0.5B", "qwen0.5b", "sweep"),
    ("Qwen2.5-1.5B", "qwen1.5b", "sweep_qwen1.5b"),
    ("Qwen2.5-3B", "qwen3b", "sweep_qwen3b"),
    ("Llama-3.2-1B", "llama1b", "sweep_llama"),
    ("Llama-3.2-3B", "llama3b", "sweep_llama3b"),
    ("OLMo-2-1B", "olmo1b", "sweep_olmo2"),
]
COT_DIR = {"A": "sweep/cot", "B": "pairB/cot", "D": "pairD/cot"}

HEAD = PatternFill("solid", fgColor="1F3864")
SUB = PatternFill("solid", fgColor="D9E2F3")
DONE = PatternFill("solid", fgColor="C6EFCE")
TODO = PatternFill("solid", fgColor="FFF2CC")
THIN = Border(*[Side(style="thin", color="BFBFBF")] * 4)


def student_dir(pair, key, tag):
    return EXP / (tag if pair == "A" else f"pair{pair}/{key}")


def adapter_path(pair, key, tag, lam, seed):
    return student_dir(pair, key, tag) / "runs" / f"lambda{lam}_seed{seed}" / "adapter" / "adapter_model.safetensors"


def header(ws, columns, widths, freeze="A2", height=32):
    ws.append(columns)
    for cell in ws[1]:
        cell.fill, cell.font = HEAD, Font(color="FFFFFF", bold=True, size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = freeze
    for i, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.row_dimensions[1].height = height


def note_block(ws, title, lines):
    ws.append([])
    ws.append([title])
    ws.cell(ws.max_row, 1).font = Font(bold=True)
    for line in lines:
        ws.append([line])


# ---------------------------------------------------------------- input sheets

def build_teacher(wb):
    ws = wb.create_sheet("입력_Teacher")
    header(ws,
           ["pair", "규모", "reasoning endpoint", "instruct endpoint", "lambda", "생성 완료",
            "teacher 정답률", "잘림률", "평균 토큰", "</think> 비율", "reflection density",
            "남은 문제", "공통 문제", "학습용", "held-out"],
           [6, 7, 26, 28, 8, 9, 12, 9, 10, 12, 16, 10, 10, 9, 9])
    for pair, reasoning, instruct, size in PAIRS:
        for lam in LAMBDAS:
            ok = (EXP / COT_DIR[pair] / f"lambda{lam}.jsonl").exists()
            ws.append([pair, size, reasoning, instruct, float(lam), "O" if ok else "",
                       None, None, None, None, None, None, None, None, 300])
            ws.cell(ws.max_row, 6).fill = DONE if ok else TODO
    for col in ("G", "H", "J"):
        for r in range(2, ws.max_row + 1):
            ws.cell(r, ws[f"{col}1"].column).number_format = "0.0%"
    note_block(ws, "채우는 법", [
        "teacher 정답률·잘림률·평균 토큰·</think> 비율·reflection density는 생성물(cot/lambda*.jsonl)에서 직접 셈",
        "</think> 비율이 급락하는 구간이 그 pair의 teacher 전환 지점이며, student 없이 결정됨",
        "공통 문제는 6개 lambda가 모두 정답을 낸 문제 수이고, 학습용 = 공통 문제 - 300",
    ])
    return ws


def build_adapter(wb):
    ws = wb.create_sheet("입력_Adapter")
    header(ws,
           ["id", "pair", "student", "lambda", "seed", "학습 완료",
            "‖ΔW‖_F", "유효 rank", "MLP 비중",
            "held300 정답률", "MATH500 pass@1", "MATH500 잘림률", "MATH500 비잘림 pass@1", "비고"],
           [20, 6, 14, 8, 6, 9, 10, 10, 10, 13, 14, 14, 18, 14])
    done = 0
    for pair, _, _, _ in PAIRS:
        for name, key, tag in STUDENTS:
            for lam in LAMBDAS:
                for seed in SEEDS:
                    ok = adapter_path(pair, key, tag, lam, seed).exists()
                    done += ok
                    ws.append([f"{pair}-{key}-L{lam}-s{seed}", pair, name, float(lam), seed,
                               "O" if ok else "", None, None, None, None, None, None, None, None])
                    ws.cell(ws.max_row, 6).fill = DONE if ok else TODO
    for col in (9, 10, 11, 12, 13):
        for r in range(2, ws.max_row + 1):
            ws.cell(r, col).number_format = "0.0%"
    return ws, done


# ------------------------------------------------------------- analysis sheets

def build_phi(wb):
    ws = wb.create_sheet("분석_phi")
    cols = ["pair", "student"] + [f"φ(λ0,{l})" for l in LAMBDAS] + [f"기준선 {l}" for l in LAMBDAS] \
        + ["무작위 수준", "분기 λ", "판정 근거"]
    header(ws, cols, [6, 14] + [10] * 12 + [11, 9, 30])
    for pair, _, _, _ in PAIRS:
        for name, _, _ in STUDENTS:
            ws.append([pair, name] + [None] * 12 + [None, None, None])
            for c in range(3, 17):
                ws.cell(ws.max_row, c).number_format = "0.000"
                ws.cell(ws.max_row, c).border = THIN
    # phi cells turn red once they drop below the seed baseline in the same column block
    for i in range(6):
        phi = get_column_letter(3 + i)
        base = get_column_letter(9 + i)
        ws.conditional_formatting.add(
            f"{phi}2:{phi}{ws.max_row}",
            CellIsRule(operator="lessThan", formula=[f"{base}2"], fill=PatternFill("solid", fgColor="F8CBAD")))
    note_block(ws, "읽는 법", [
        "φ(λ0, λ) = ‖U_0ᵀU_λ‖²_F / r,  r = 16.  ΔW = (α/r)·BA의 상위 16개 좌특이벡터, 층·모듈 168개를 ‖ΔW‖_F로 가중평균",
        "기준선 = 같은 λ·같은 데이터에서 seed만 바꾼 두 학습 사이의 φ (seed 0-1, 0-2, 1-2의 평균)",
        "φ가 그 λ의 기준선보다 낮으면 주황색으로 표시됨 → 학습 랜덤화로 설명되지 않는 방향 변화",
        "분기 λ = 기준선 아래로 처음 내려가는 λ",
        "무작위 수준 ≈ r / d_out (Qwen2.5-0.5B 기준 0.017)",
    ])
    return ws


def build_accuracy(wb):
    ws = wb.create_sheet("분석_정답률")
    cols = ["pair", "student", "학습 전 held300", "학습 전 MATH500"] \
        + [f"held300 {l}" for l in LAMBDAS] + [f"MATH500 {l}" for l in LAMBDAS] \
        + [f"잘림률 {l}" for l in LAMBDAS]
    header(ws, cols, [6, 14, 13, 13] + [11] * 18)
    src = "입력_Adapter"
    for pair, _, _, _ in PAIRS:
        for name, _, _ in STUDENTS:
            r = ws.max_row + 1
            row = [pair, name, None, None]
            for col_letter in ("J", "K", "L"):  # held300, MATH500 pass@1, truncation
                for lam in LAMBDAS:
                    row.append(f'=IFERROR(AVERAGEIFS(\'{src}\'!{col_letter}:{col_letter},'
                               f"'{src}'!B:B,$A{r},'{src}'!C:C,$B{r},'{src}'!D:D,{float(lam)}),\"\")")
            ws.append(row)
            for c in range(3, 23):
                ws.cell(r, c).number_format = "0.0%"
                ws.cell(r, c).border = THIN
    note_block(ws, "읽는 법", [
        "λ별 값은 입력_Adapter의 seed 3개를 AVERAGEIFS로 평균한 것이라, 값을 넣으면 자동으로 갱신됨",
        "held-out 300제는 6개 λ가 모두 맞힌 문제에서 뽑혀 난이도가 낮음 → 절대 성능이 아니라 λ 간 상대 비교로만 해석",
        "MATH-500은 4,000제에서 미리 제외했으므로 teacher가 푼 적도, 정답률로 걸러진 적도 없음",
        "잘림률을 함께 보아야 성능 변화가 추론 능력 때문인지 출력 길이 때문인지 구분됨",
        "소형 student 증류는 seed 간 편차가 매우 크므로 단일 seed 결과는 해석하지 않음 (Sumit et al. 2026)",
    ])
    return ws


def build_transition(wb):
    ws = wb.create_sheet("분석_전환점")
    header(ws,
           ["pair", "student", "teacher 전환 λ", "adapter 분기 λ", "일치",
            "λ0 → λ1 held300", "λ0 → λ1 MATH500", "비고"],
           [6, 14, 14, 14, 8, 16, 16, 34])
    for pair, _, _, _ in PAIRS:
        for name, _, _ in STUDENTS:
            r = ws.max_row + 1
            ws.append([pair, name,
                       f'=IFERROR(INDEX(분석_phi!A:A,0),"")' if False else None,
                       None,
                       f'=IF(OR($C{r}="",$D{r}=""),"",IF($C{r}=$D{r},"O","X"))',
                       None, None, None])
            for c in range(3, 8):
                ws.cell(r, c).border = THIN
    note_block(ws, "읽는 법", [
        "teacher 전환 λ는 입력_Teacher의 </think> 비율이 급락하는 λ이며, student와 무관하게 pair마다 하나로 정해짐",
        "adapter 분기 λ는 분석_phi에서 φ가 seed 기준선 아래로 처음 내려가는 λ",
        "두 값은 완전히 독립적으로 측정됨 — 하나는 teacher 생성물만, 다른 하나는 student 가중치만 사용",
        "adapter 분기가 teacher와 무관한 요인 때문이라면 3개 pair에서 같은 λ에 나타나야 하므로,",
        "  pair마다 전환 지점이 다르다는 점이 이 가설의 반증 조건이 됨",
    ])
    return ws


# --------------------------------------------------------------- progress sheet

def build_progress(wb):
    ws = wb.create_sheet("진행현황", 0)
    a, t = "입력_Adapter", "입력_Teacher"
    header(ws, ["단계", "필요", "완료", "남음", "비율", "비고"], [34, 9, 9, 9, 9, 46])
    rows = [
        ("teacher 생성 (3쌍 × 6λ)", 18, f"=COUNTIF('{t}'!F:F,\"O\")", "쌍마다 4,000제, 생성 설정 동일"),
        ("teacher 형식 측정 (</think> 비율)", 18, f"=COUNT('{t}'!J:J)", "생성물에서 직접 셈. student 불필요"),
        ("adapter 학습", 324, f"=COUNTIF('{a}'!F:F,\"O\")", "6 student × 6λ × 3쌍 × 3 seed"),
        ("adapter 가중치 측정 (‖ΔW‖, rank)", 324, f"=COUNT('{a}'!G:G)", "학습된 adapter를 읽기만 하면 됨"),
        ("held-out 300제 정답률", 324, f"=COUNT('{a}'!J:J)", "seed 3개 × 샘플링 4회"),
        ("MATH-500 정답률", 324, f"=COUNT('{a}'!K:K)", "pass@1 · 잘림률 · 비잘림 pass@1"),
        ("φ overlap (λ0 기준)", 108, "=COUNT(분석_phi!C:H)", "3쌍 × 6 student × 6λ"),
        ("seed 기준선", 108, "=COUNT(분석_phi!I:N)", "같은 λ·같은 데이터, seed 간 φ"),
    ]
    for label, need, formula, note in rows:
        r = ws.max_row + 1
        ws.append([label, need, formula, f"=B{r}-C{r}", f"=IF(B{r}=0,0,C{r}/B{r})", note])
        ws.cell(r, 5).number_format = "0%"
        for c in range(1, 7):
            ws.cell(r, c).border = THIN
    total = ws.max_row + 1
    ws.append(["합계", f"=SUM(B2:B{total - 1})", f"=SUM(C2:C{total - 1})",
               f"=B{total}-C{total}", f"=IF(B{total}=0,0,C{total}/B{total})", ""])
    for cell in ws[total]:
        cell.fill, cell.font, cell.border = SUB, Font(bold=True), THIN
    ws.cell(total, 5).number_format = "0%"

    ws.append([])
    ws.append(["adapter 학습 현황 (각 칸 최대 18 = 6λ × 3 seed)"])
    ws.cell(ws.max_row, 1).font = Font(bold=True)
    head_row = ws.max_row + 1
    ws.append(["student"] + [f"쌍 {p}" for p, _, _, _ in PAIRS] + ["합계 (/54)"])
    for cell in ws[head_row]:
        if cell.value:
            cell.fill, cell.font = SUB, Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
            cell.border = THIN
    for name, _, _ in STUDENTS:
        r = ws.max_row + 1
        ws.append([name] + [f"=COUNTIFS('{a}'!B:B,\"{p}\",'{a}'!C:C,$A{r},'{a}'!F:F,\"O\")"
                            for p, _, _, _ in PAIRS] + [f"=SUM(B{r}:D{r})"])
        for c in range(1, 6):
            ws.cell(r, c).border = THIN
            if c > 1:
                ws.cell(r, c).alignment = Alignment(horizontal="center")
    r = ws.max_row + 1
    ws.append(["합계 (/108)"] + [f"=SUM({get_column_letter(c)}{r - 6}:{get_column_letter(c)}{r - 1})"
                               for c in range(2, 5)] + [f"=SUM(B{r}:D{r})"])
    for cell in ws[r]:
        cell.fill, cell.font, cell.border = SUB, Font(bold=True), THIN

    note_block(ws, "시트 구성", [
        "입력_Teacher / 입력_Adapter : 실제 값을 넣는 곳. 한 줄이 하나의 teacher 또는 하나의 adapter",
        "분석_phi / 분석_정답률 / 분석_전환점 : 읽기 위한 곳. 분석_정답률은 입력_Adapter에서 자동 집계됨",
        "진행현황 : 이 시트. 다른 시트를 세기만 하므로 직접 고칠 것이 없음",
        "",
        "이 파일은 experiments/lambda_lora_bits/make_tracker.py로 다시 생성하면 학습 현황이 갱신됨",
        "다만 직접 입력한 값은 덮어쓰이므로, 값을 넣기 시작한 뒤에는 사본을 두고 쓸 것",
    ])
    return ws


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(EXP.parent / "tracker.xlsx"))
    args = ap.parse_args()

    wb = Workbook()
    wb.remove(wb.active)
    build_transition(wb)
    build_phi(wb)
    build_accuracy(wb)
    build_teacher(wb)
    _, n_adapter = build_adapter(wb)
    build_progress(wb)
    wb._sheets = [wb["진행현황"], wb["분석_전환점"], wb["분석_phi"], wb["분석_정답률"],
                  wb["입력_Teacher"], wb["입력_Adapter"]]
    wb.save(args.output)
    print(f"{args.output}  (adapter {n_adapter}/324)")


if __name__ == "__main__":
    main()
