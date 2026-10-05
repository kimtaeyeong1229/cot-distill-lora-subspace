#!/usr/bin/env python3
"""One appendix table of example outputs, chosen by a fixed rule (no hand picking).

Students: cell A/Qwen, seed 0, the first 200 MATH-500 problems in file order. The problem is the first one
for which (1) the lambda0 student's first response hits the generation limit or loops, (2) the
region-removed adapter (lambda0.0_rm_pre) answers correctly, and (3) the beta = 2 graft does not finish.
If none qualifies, "correctly" is dropped first, then the graft condition; the rule used is recorded.
Rows: lambda0 student, region removed, length-matched cut (lambda0.0_budget), graft at beta = 2 (graft2),
lambda1 student, each the first response of seed 0 on that problem. Unedited students come from
eval_math500_7680/lambda{0.0,1.0}_seed0.json (response_0, its stored grade and truncation flag), edits from
eval_main_s0/<name>.samples.jsonl.gz (two samples per problem in problem order; sample 0 is used, graded with
common.is_correct, truncated if finish_reason is neither "stop" nor "forced"; the length-matched cut ends with
finish_reason "forced", the answer it writes after the cut).
Teachers: pair A at lambda 0, 0.2 and 0.4 on the first unique_id (in the lambda 0 file order) present in all
three sweep/splits/lambda*/train.jsonl, i.e. the training solutions as the students saw them (tags removed).

Each row shows the first 30 and the last 40 words, the token count, reflective markers per 1,000 words
(analyze_main_density.REFLECTIVE) and an end status. A response loops by analyze_degeneration's rule (at least
half of the word 10-grams in its last 300 words repeat). Math spans are shown as [math] and markers in bold.

Run inside the container from the repo root; writes experiments/examples.json and
experiments/lambda_lora_bits/examples_table.tex (copied to the manuscript as tables/tab25.tex).
"""

import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

E = Path("experiments/lambda_lora_bits")
sys.path.insert(0, str(E))
from analyze_degeneration import repeat_share  # noqa: E402
from analyze_main_density import REFLECTIVE  # noqa: E402
from common import is_correct  # noqa: E402

TAG, BENCH, PROBLEMS, SAMPLES = "sweep", "eval_math500_7680", 200, 2
HEAD, TAIL = 30, 40
MATH = re.compile(r"\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\\boxed\{(?:[^{}]|\{[^{}]*\})*\}", re.S)


def markers(text):
    return len(REFLECTIVE.findall(text)) / max(len(text.split()), 1) * 1000


def unedited(lam):
    recs = json.loads((E / TAG / BENCH / f"lambda{lam}_seed0.json").read_text())["records"][:PROBLEMS]
    return [{"text": r["response_0"], "tokens": r["tokens"][0], "truncated": bool(r["truncated"][0]),
             "correct": bool(r["correct"][0])} for r in recs]


def edited(name, answers):
    with gzip.open(E / TAG / "eval_main_s0" / f"{name}.samples.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for i, line in enumerate(f) if i % SAMPLES == 0][:PROBLEMS]
    return [{"text": r["text"], "tokens": int(r["tokens"]), "truncated": r["finish_reason"] not in ("stop", "forced"),
             "forced": r["finish_reason"] == "forced", "problem": r["problem"], "answer": answers[p]}
            for p, r in enumerate(rows)]


def status(row):
    loop = repeat_share(row["text"]) >= 0.5
    if row["truncated"]:
        return "truncated, loop" if loop else "truncated"
    end = "correct" if row["correct"] else "wrong"
    done = f"cut, then answered ({end})" if row.get("forced") else f"answered ({end})"
    return f"{done}, loop" if loop else done


def tex(text):
    """Head ... tail of a response, with math replaced, special characters escaped and markers in bold."""
    words = MATH.sub(" [math] ", text).split()
    shown = words if len(words) <= HEAD + TAIL else words[:HEAD] + ["\u2026"] + words[-TAIL:]
    s = " ".join(shown)
    for a, b in (("\\", "\x00"), ("{", "\\{"), ("}", "\\}"), ("$", "\\$"), ("&", "\\&"), ("%", "\\%"),
                 ("#", "\\#"), ("_", "\\_"), ("^", "\\^{}"), ("~", "\\~{}")):
        s = s.replace(a, b)
    s = s.replace("\x00", "\\textbackslash{}").replace("\u2026", "\\ldots{}").replace("**", "")
    for a, b in (("\u00f7", "/"), ("\u2248", "approx."), ("\u00d7", "x"), ("\u2212", "-"), ("\u03c0", "pi"),
                 ("\u221a", "sqrt"), ("\u2264", "<="), ("\u2265", ">="), ("\u2019", "'"), ("\u2013", "-"),
                 ("\u2014", "-"), ("\u00b2", "^2")):
        s = s.replace(a, b)
    s = re.sub(r"[^\x20-\x7e]", "", s)
    return REFLECTIVE.sub(lambda m: "\\textbf{" + m.group(0) + "}", s)


def main():
    df = pd.read_parquet(E / "data" / "math500.parquet").to_dict("records")[:PROBLEMS]
    answers = [r["answer"] for r in df]
    base0, base1 = unedited("0.0"), unedited("1.0")
    rm, cut, graft = (edited(n, answers) for n in ("lambda0.0_rm_pre", "lambda0.0_budget", "graft2"))
    for rows in (rm, cut, graft):
        for p, r in enumerate(rows):
            r["correct"] = (not r["truncated"]) and is_correct(r["problem"], r["text"], r["answer"])

    def bad0(p):
        return base0[p]["truncated"] or repeat_share(base0[p]["text"]) >= 0.5

    rules = [("lambda0 truncated or loops, removal correct, graft2 not finished",
              lambda p: bad0(p) and rm[p]["correct"] and graft[p]["truncated"]),
             ("lambda0 truncated or loops, removal finishes, graft2 not finished",
              lambda p: bad0(p) and not rm[p]["truncated"] and graft[p]["truncated"]),
             ("lambda0 truncated or loops, removal finishes",
              lambda p: bad0(p) and not rm[p]["truncated"])]
    for rule, ok in rules:
        hit = next((p for p in range(PROBLEMS) if ok(p)), None)
        if hit is not None:
            break
    p = hit
    student = []
    for label, rows in (("$\\lambda=0$ student", base0), ("Region removed", rm), ("Length-matched cut", cut),
                        ("Graft, $\\beta=2$", graft), ("$\\lambda=1$ student", base1)):
        r = rows[p]
        student.append({"row": label, "tokens": r["tokens"], "markers_per_1k": markers(r["text"]),
                        "status": status(r), "repeat_share": repeat_share(r["text"]), "text": r["text"]})

    splits = {lam: [json.loads(l) for l in open(E / "sweep" / "splits" / f"lambda{lam}" / "train.jsonl")]
              for lam in ("0.0", "0.2", "0.4")}
    ids = [{r["unique_id"] for r in rows} for rows in splits.values()]
    uid = next(r["unique_id"] for r in splits["0.0"] if all(r["unique_id"] in s for s in ids))
    teacher = []
    for lam, rows in splits.items():
        r = next(x for x in rows if x["unique_id"] == uid)
        teacher.append({"row": f"Teacher $\\lambda={float(lam):g}$", "tokens": r["n_tokens"],
                        "markers_per_1k": markers(r["response"]), "text": r["response"]})

    out = {"rule": rule, "student_cell": "A/Qwen", "seed": 0, "problem_index": p,
           "problem_unique_id": df[p]["unique_id"], "problem": df[p]["problem"], "answer": answers[p],
           "students": student, "teacher_unique_id": uid,
           "teacher_problem": splits["0.0"][[x["unique_id"] for x in splits["0.0"]].index(uid)]["problem"],
           "teachers": teacher}
    Path("experiments/examples.json").write_text(json.dumps(out, indent=2))

    lines = [f"% Examples table, generated by make_examples_table.py from experiments/examples.json",
             f"% rule: {rule}; A/Qwen seed 0, MATH-500 problem index {p} ({df[p]['unique_id']}); teacher unique_id {uid}",
             r"\setlength{\tabcolsep}{3pt}\scriptsize",
             r"\begin{tabular}{@{}p{0.12\textwidth}p{0.64\textwidth}rrp{0.08\textwidth}@{}}",
             r"\toprule", r"Row & First 30 and last 40 words & Tokens & Markers & End \\", r"\midrule"]
    for r in student:
        lines.append(f"{r['row']} & {tex(r['text'])} & {r['tokens']:,} & {r['markers_per_1k']:.1f} & {r['status']} \\\\")
        lines.append(r"\addlinespace")
    lines.append(r"\midrule")
    for r in teacher:
        lines.append(f"{r['row']} & {tex(r['text'])} & {r['tokens']:,} & {r['markers_per_1k']:.1f} & training solution \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (E / "examples_table.tex").write_text("\n".join(lines) + "\n")
    print("rule:", rule, "| problem", p, df[p]["unique_id"], "| teacher", uid)
    for r in student + teacher:
        print(f"{r['row']:22s} tokens {r['tokens']:>6} markers {r['markers_per_1k']:5.1f} {r.get('status', '')}")


if __name__ == "__main__":
    main()
