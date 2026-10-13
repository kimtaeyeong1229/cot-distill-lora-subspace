#!/usr/bin/env python3
import gzip
import json
import re
import pandas as pd
from analyze_degeneration import repeat_share
from analyze_main_density import REFLECTIVE
from common import is_correct
from config import EXP as E, RESULTS

TAG, BENCH, PROBLEMS, SAMPLES = ("pairA/qwen0.5b", "eval_math500_7680", 200, 2)
HEAD, TAIL = (30, 40)
MATH = re.compile(
    "\\$\\$.*?\\$\\$|\\$.*?\\$|\\\\\\[.*?\\\\\\]|\\\\\\(.*?\\\\\\)|\\\\boxed\\{(?:[^{}]|\\{[^{}]*\\})*\\}",
    re.S,
)


def markers(text):
    return len(REFLECTIVE.findall(text)) / max(len(text.split()), 1) * 1000


def unedited(lam):
    recs = json.loads((E / TAG / BENCH / f"lambda{lam}_seed0.json").read_text())["records"][:PROBLEMS]
    return [
        {
            "text": r["response_0"],
            "tokens": r["tokens"][0],
            "truncated": bool(r["truncated"][0]),
            "correct": bool(r["correct"][0]),
        }
        for r in recs
    ]


def edited(name, answers):
    with gzip.open(E / TAG / "eval_main_s0" / f"{name}.samples.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for i, line in enumerate(f) if i % SAMPLES == 0][:PROBLEMS]
    return [
        {
            "text": r["text"],
            "tokens": int(r["tokens"]),
            "truncated": r["finish_reason"] not in ("stop", "forced"),
            "forced": r["finish_reason"] == "forced",
            "problem": r["problem"],
            "answer": answers[p],
        }
        for p, r in enumerate(rows)
    ]


def status(row):
    loop = repeat_share(row["text"]) >= 0.5
    if row["truncated"]:
        return "truncated, loop" if loop else "truncated"
    end = "correct" if row["correct"] else "wrong"
    done = f"cut, then answered ({end})" if row.get("forced") else f"answered ({end})"
    return f"{done}, loop" if loop else done


def tex(text):
    words = MATH.sub(" [math] ", text).split()
    shown = words if len(words) <= HEAD + TAIL else words[:HEAD] + ["…"] + words[-TAIL:]
    s = " ".join(shown)
    for a, b in (
        ("\\", "\x00"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("$", "\\$"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("^", "\\^{}"),
        ("~", "\\~{}"),
    ):
        s = s.replace(a, b)
    s = s.replace("\x00", "\\textbackslash{}").replace("…", "\\ldots{}").replace("**", "")
    for a, b in (
        ("÷", "/"),
        ("≈", "approx."),
        ("×", "x"),
        ("−", "-"),
        ("π", "pi"),
        ("√", "sqrt"),
        ("≤", "<="),
        ("≥", ">="),
        ("’", "'"),
        ("–", "-"),
        ("—", "-"),
        ("²", "^2"),
    ):
        s = s.replace(a, b)
    s = re.sub("[^\\x20-\\x7e]", "", s)
    return REFLECTIVE.sub(lambda m: "\\textbf{" + m.group(0) + "}", s)


def main():
    df = pd.read_parquet(E / "data" / "math500.parquet").to_dict("records")[:PROBLEMS]
    answers = [r["answer"] for r in df]
    base0, base1 = (unedited("0.0"), unedited("1.0"))
    rm, cut, graft = (edited(n, answers) for n in ("lambda0.0_rm_pre", "lambda0.0_budget", "graft2"))
    for rows in (rm, cut, graft):
        for p, r in enumerate(rows):
            r["correct"] = not r["truncated"] and is_correct(r["problem"], r["text"], r["answer"])

    def bad0(p):
        return base0[p]["truncated"] or repeat_share(base0[p]["text"]) >= 0.5

    rules = [
        (
            "lambda0 truncated or loops, removal correct, graft2 not finished",
            lambda p: bad0(p) and rm[p]["correct"] and graft[p]["truncated"],
        ),
        (
            "lambda0 truncated or loops, removal finishes, graft2 not finished",
            lambda p: bad0(p) and (not rm[p]["truncated"]) and graft[p]["truncated"],
        ),
        ("lambda0 truncated or loops, removal finishes", lambda p: bad0(p) and (not rm[p]["truncated"])),
    ]
    for rule, ok in rules:
        hit = next((p for p in range(PROBLEMS) if ok(p)), None)
        if hit is not None:
            break
    p = hit
    student = []
    for label, rows in (
        ("$\\lambda=0$ student", base0),
        ("Region removed", rm),
        ("Length-matched cut", cut),
        ("Graft, $\\beta=2$", graft),
        ("$\\lambda=1$ student", base1),
    ):
        r = rows[p]
        student.append(
            {
                "row": label,
                "tokens": r["tokens"],
                "markers_per_1k": markers(r["text"]),
                "status": status(r),
                "repeat_share": repeat_share(r["text"]),
                "text": r["text"],
            }
        )
    splits = {
        lam: [json.loads(l) for l in open(E / "pairA" / "splits" / f"lambda{lam}" / "train.jsonl")]
        for lam in ("0.0", "0.2", "0.4")
    }
    ids = [{r["unique_id"] for r in rows} for rows in splits.values()]
    uid = next((r["unique_id"] for r in splits["0.0"] if all((r["unique_id"] in s for s in ids))))
    teacher = []
    for lam, rows in splits.items():
        r = next((x for x in rows if x["unique_id"] == uid))
        teacher.append(
            {
                "row": f"Teacher $\\lambda={float(lam):g}$",
                "tokens": r["n_tokens"],
                "markers_per_1k": markers(r["response"]),
                "text": r["response"],
            }
        )
    out = {
        "rule": rule,
        "student_cell": "A/Qwen",
        "seed": 0,
        "problem_index": p,
        "problem_unique_id": df[p]["unique_id"],
        "problem": df[p]["problem"],
        "answer": answers[p],
        "students": student,
        "teacher_unique_id": uid,
        "teacher_problem": splits["0.0"][[x["unique_id"] for x in splits["0.0"]].index(uid)]["problem"],
        "teachers": teacher,
    }
    (RESULTS / "examples.json").write_text(json.dumps(out, indent=2))
    lines = [
        f"% Examples table, generated by make_examples_table.py from experiments/examples.json",
        f"% rule: {rule}; A/Qwen seed 0, MATH-500 problem index {p} ({df[p]['unique_id']}); teacher unique_id {uid}",
        "\\setlength{\\tabcolsep}{3pt}\\scriptsize",
        "\\begin{tabular}{@{}p{0.12\\textwidth}p{0.64\\textwidth}rrp{0.08\\textwidth}@{}}",
        "\\toprule",
        "Row & First 30 and last 40 words & Tokens & Markers & End \\\\",
        "\\midrule",
    ]
    for r in student:
        lines.append(
            f"{r['row']} & {tex(r['text'])} & {r['tokens']:,} & {r['markers_per_1k']:.1f} & {r['status']} \\\\"
        )
        lines.append("\\addlinespace")
    lines.append("\\midrule")
    for r in teacher:
        lines.append(
            f"{r['row']} & {tex(r['text'])} & {r['tokens']:,} & {r['markers_per_1k']:.1f} & training solution \\\\"
        )
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (E / "examples_table.tex").write_text("\n".join(lines) + "\n")
    print("rule:", rule, "| problem", p, df[p]["unique_id"], "| teacher", uid)
    for r in student + teacher:
        print(
            f"{r['row']:22s} tokens {r['tokens']:>6} markers {r['markers_per_1k']:5.1f} {r.get('status', '')}"
        )


if __name__ == "__main__":
    main()
