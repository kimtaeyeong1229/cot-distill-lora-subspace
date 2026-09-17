#!/usr/bin/env python3
"""Count finished experiments against experiments/PLAN.md by looking at result files."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
PAIRS = ["A", "B", "C", "D"]
STUDENTS = [  # key, label, directory for pair A
    ("qwen0.5b", "Qwen2.5-0.5B", "sweep"),
    ("llama1b", "Llama-3.2-1B", "sweep_llama"),
    ("olmo1b", "OLMo-2-1B", "sweep_olmo2"),
    ("qwen1.5b", "Qwen2.5-1.5B", "sweep_qwen1.5b"),
    ("qwen3b", "Qwen2.5-3B", "sweep_qwen3b"),
    ("llama3b", "Llama-3.2-3B", "sweep_llama3b"),
]
OTHER_BENCH = ["gsm8k", "amc23", "aime24", "minervamath", "olympiadbench", "gpqa_diamond"]


def pair_dir(pair):
    return ROOT / ("sweep" if pair == "A" else f"pair{pair}")


def student_dir(pair, student):
    key, _, dir_a = student
    return ROOT / dir_a if pair == "A" else ROOT / f"pair{pair}" / key


def exists(path):
    return path.exists()


def math500(sdir, name):
    return exists(sdir / "eval_math500" / f"{name}.json") or exists(sdir / "eval_test" / f"{name}.json")


def best_lambda(sdir):
    accs = {}
    for lam in LAMBDAS:
        p = sdir / "eval_val" / f"lambda{lam}_seed0.json"
        if not p.exists():
            return None
        accs[lam] = json.loads(p.read_text())["summary"]["accuracy"]
    return max(accs, key=accs.get)


def main():
    totals = {}

    def tick(category, done, total):
        d, t = totals.get(category, (0, 0))
        totals[category] = (d + done, t + total)

    print("== 묶음별 teacher 준비 (보간 · 생성 · 공통 분할)")
    for pair in PAIRS:
        pd = pair_dir(pair)
        interp = sum(exists(pd / "teachers" / f"lambda{lam}" / "model.safetensors")
                     or exists(pd / "teachers" / f"lambda{lam}" / "model.safetensors.index.json") for lam in LAMBDAS)
        gen = sum(exists(pd / "cot" / f"lambda{lam}.jsonl") for lam in LAMBDAS)
        split = int(exists(pd / "splits" / "split_stats.json"))
        tick("1. teacher 보간", interp, 6)
        tick("2. CoT 생성", gen, 6)
        tick("3. 공통 분할", split, 1)
        mark = "✅" if interp == 6 and gen == 6 and split else ("🟡" if interp or gen else "⬜")
        print(f"  {mark} 묶음 {pair}: 보간 {interp}/6 · 생성 {gen}/6 · 분할 {split}/1")

    print("\n== student × 묶음 (학습 · bits · held-out · MATH-500 · 기타 6벤치)")
    header = "  " + f"{'student':<14}" + "".join(f"{'묶음 ' + p:^26}" for p in PAIRS)
    print(header)
    for student in STUDENTS:
        cells = []
        for pair in PAIRS:
            sd = student_dir(pair, student)
            train = sum(exists(sd / "runs" / f"lambda{lam}_seed0" / "adapter" / "adapter_model.safetensors") for lam in LAMBDAS)
            bits = sum(exists(sd / "bits" / f"lambda{lam}_seed0.json") for lam in LAMBDAS)
            val = sum(exists(sd / "eval_val" / f"lambda{lam}_seed0.json") for lam in LAMBDAS)
            m5 = sum(math500(sd, f"lambda{lam}_seed0") for lam in LAMBDAS)
            other = sum(exists(sd / f"eval_{bench}" / f"lambda{lam}_seed0.json") for bench in OTHER_BENCH for lam in LAMBDAS)
            tick("4. LoRA 학습", train, 6)
            tick("5. bits 측정", bits, 6)
            tick("6. held-out 평가", val, 6)
            tick("7. MATH-500 (λ별)", m5, 6)
            tick("8. 기타 6벤치 (λ별)", other, len(LAMBDAS) * len(OTHER_BENCH))
            cells.append(f"{train}/{bits}/{val}/{m5}/{other:>2}")
        print("  " + f"{student[1]:<14}" + "".join(f"{c:^26}" for c in cells))
    print("  (각 칸: 학습/6 · bits/6 · held-out/6 · MATH-500/6 · 기타 6벤치/36)")

    print("\n== 기본 student 평가 (teacher와 무관, student당 한 번)")
    for student in STUDENTS:
        sd = student_dir("A", student)
        val = int(exists(sd / "eval_val" / "base.json"))
        m5 = int(math500(sd, "base"))
        other = sum(exists(sd / f"eval_{bench}" / "base.json") for bench in OTHER_BENCH)
        tick("9. 기본 student 평가", val + m5 + other, 2 + len(OTHER_BENCH))
        print(f"  {student[1]:<14} held-out {val}/1 · MATH-500 {m5}/1 · 기타 {other}/6")

    print("\n== 전체 진행")
    done_all = total_all = 0
    for category, (d, t) in totals.items():
        done_all += d; total_all += t
        bar = "█" * round(20 * d / t) + "░" * (20 - round(20 * d / t))
        print(f"  {category:<28} {bar} {d:4d}/{t:<4d} ({d / t:5.1%})")
    print(f"  {'합계':<28} {'':20} {done_all:4d}/{total_all:<4d} ({done_all / total_all:5.1%})")


if __name__ == "__main__":
    main()
