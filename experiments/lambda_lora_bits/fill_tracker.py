#!/usr/bin/env python3
"""Fill tracker.xlsx from whatever exists on disk: teacher generations, adapters, evaluations.

Teacher  reasoning-format columns, accuracy, truncation, length, split sizes
Adapter  Frobenius norm, effective rank, MLP share of the norm, held-out and MATH-500 accuracy
Phi      phi(lambda0, lambda) for seed 0, and the seed baseline (mean over the seed pairs at that lambda)

Safe to re-run: it rewrites only the cells it can compute and leaves the rest untouched.
"""

import argparse
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch
from openpyxl import load_workbook
from safetensors import safe_open

EXP = Path(__file__).resolve().parent
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
SEEDS = [0, 1, 2]
PAIRS = ["A", "B", "D"]
COT_DIR = {"A": "sweep/cot", "B": "pairB/cot", "D": "pairD/cot"}
SPLITS = {"A": "sweep/splits", "B": "pairB/splits", "D": "pairD/splits"}
STUDENTS = [
    ("Qwen2.5-0.5B", "qwen0.5b", "sweep"),
    ("Qwen2.5-1.5B", "qwen1.5b", "sweep_qwen1.5b"),
    ("Qwen2.5-3B", "qwen3b", "sweep_qwen3b"),
    ("Llama-3.2-1B", "llama1b", "sweep_llama"),
    ("Llama-3.2-3B", "llama3b", "sweep_llama3b"),
    ("OLMo-2-1B", "olmo1b", "sweep_olmo2"),
]
MLP = {"gate_proj", "up_proj", "down_proj"}
REFLECT = re.compile(r"\b(wait|Wait|hmm|Hmm|alternatively|Alternatively|but wait|let me re)\b")


def student_dir(pair, key, tag):
    return EXP / (tag if pair == "A" else f"pair{pair}/{key}")


# ------------------------------------------------------------------- teacher

def boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth, out = i + 7, 1, ""
    while j < len(text) and depth:
        c = text[j]
        depth += (c == "{") - (c == "}")
        if depth:
            out += c
        j += 1
    return out.strip()


def norm_answer(x):
    if x is None:
        return None
    x = re.sub(r"\\left|\\right", "", x)
    return x.replace(" ", "").replace("\\!", "").replace("\\,", "").replace("$", "").rstrip(".")


def teacher_stats(pair, lam):
    path = EXP / COT_DIR[pair] / f"lambda{lam}.jsonl"
    if not path.exists():
        return None
    n = correct = trunc = think = 0
    tokens, reflect = [], []
    for line in path.open():
        row = json.loads(line)
        n += 1
        body = row["cot_response"]
        tokens.append(row["generated_tokens"])
        trunc += row["finish_reason"] == "length"
        think += "</think>" in body
        reflect.append(len(REFLECT.findall(body)))
        correct += norm_answer(boxed(body)) == norm_answer(row["answer"])
    return dict(accuracy=correct / n, truncated=trunc / n, tokens=st.mean(tokens),
                think=think / n, reflect=st.mean(reflect))


def split_stats(pair):
    path = EXP / SPLITS[pair] / "split_stats.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ------------------------------------------------------------------- adapter

def load_delta(path, scale):
    parts = defaultdict(dict)
    with safe_open(path, "pt") as handle:
        for key in handle.keys():
            m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)", key)
            if m:
                parts[(int(m.group(1)), m.group(2))]["A" if "lora_A" in key else "B"] = handle.get_tensor(key).float()
    return {k: (v["B"] * scale, v["A"]) for k, v in parts.items() if len(v) == 2}


def factor(B, A):
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.T)
    U, S, _ = torch.linalg.svd(Rb @ Ra.T)
    return Qb @ U, S, float((S ** 2).sum().sqrt())


def adapter_stats(path):
    """(basis, singular values, Frobenius norm) per module, plus scalar summaries."""
    config = json.loads((path.parent / "adapter_config.json").read_text())
    mods = {k: factor(*v) for k, v in load_delta(path, config["lora_alpha"] / config["r"]).items()}
    weight = sum(v[2] for v in mods.values())
    fro = math.sqrt(sum(v[2] ** 2 for v in mods.values()))
    mlp = math.sqrt(sum(v[2] ** 2 for k, v in mods.items() if k[1] in MLP))

    def eff(S):
        p = (S ** 2) / (S ** 2).sum()
        return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))

    rank = sum(eff(v[1]) * v[2] for v in mods.values()) / weight
    return mods, dict(fro=fro, rank=rank, mlp_share=mlp / fro)


def overlap(x, y):
    keys = set(x) & set(y)
    weight = sum(x[k][2] for k in keys)
    return sum(float((x[k][0].T @ y[k][0]).pow(2).sum() / x[k][0].shape[1]) * x[k][2] for k in keys) / weight


def chance_level(mods):
    weight = sum(v[2] for v in mods.values())
    return sum(v[0].shape[1] / v[0].shape[0] * v[2] for v in mods.values()) / weight


# ---------------------------------------------------------------- evaluation

def read_eval(directory, condition):
    for name in (f"{condition}.json",):
        path = directory / name
        if path.exists():
            summary = json.loads(path.read_text())["summary"]
            return summary.get("accuracy"), summary.get("truncated_rate")
    return None, None


def accuracy_for(pair, key, tag, lam, seed):
    base = student_dir(pair, key, tag)
    condition = f"lambda{lam}_seed{seed}"
    held, _ = read_eval(base / "eval_val", condition)
    m500, trunc = None, None
    for folder in ("eval_math500", "eval_test"):
        m500, trunc = read_eval(base / folder, condition)
        if m500 is not None:
            break
    return held, m500, trunc


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", default=str(EXP.parent / "tracker.xlsx"))
    ap.add_argument("--skip_weights", action="store_true", help="Fill only the cheap columns.")
    args = ap.parse_args()

    wb = load_workbook(args.workbook)
    filled = defaultdict(int)

    # --- teacher sheet -----------------------------------------------------
    ws = wb["입력_Teacher"]
    shared = {p: split_stats(p) for p in PAIRS}
    for row in ws.iter_rows(min_row=2):
        pair, lam = row[0].value, row[4].value
        if pair not in PAIRS or lam is None:
            continue
        lam = f"{lam:.1f}"
        stats = teacher_stats(pair, lam)
        if stats:
            row[6].value, row[7].value, row[8].value = stats["accuracy"], stats["truncated"], round(stats["tokens"])
            row[9].value, row[10].value = stats["think"], round(stats["reflect"], 2)
            filled["teacher"] += 1
        s = shared.get(pair)
        if s:
            per = s["per_lambda"].get(lam)
            if per:
                row[11].value = per["kept"]
            row[12].value, row[13].value = s["shared"], s["n_train"]

    # --- adapter sheet -----------------------------------------------------
    ws = wb["입력_Adapter"]
    cache = {}
    for row in ws.iter_rows(min_row=2):
        pair, name, lam, seed = row[1].value, row[2].value, row[3].value, row[4].value
        if pair not in PAIRS or lam is None:
            continue
        key, tag = next((k, t) for n, k, t in STUDENTS if n == name)
        lam = f"{lam:.1f}"
        path = student_dir(pair, key, tag) / "runs" / f"lambda{lam}_seed{seed}" / "adapter" / "adapter_model.safetensors"
        if path.exists() and not args.skip_weights:
            mods, scalars = adapter_stats(path)
            cache[(pair, key, lam, seed)] = mods
            row[6].value = round(scalars["fro"], 3)
            row[7].value = round(scalars["rank"], 2)
            row[8].value = round(scalars["mlp_share"], 4)
            filled["adapter"] += 1
        held, m500, trunc = accuracy_for(pair, key, tag, lam, seed)
        for cell, value in ((row[9], held), (row[10], m500), (row[11], trunc)):
            if value is not None:
                cell.value = round(value, 4)
                filled["accuracy"] += 1

    # --- phi sheet (lambdas are columns: phi at C..H, seed baseline at I..N) ---
    ws = wb["분석_phi"]
    for row in ws.iter_rows(min_row=2):
        pair, name = row[0].value, row[1].value
        if pair not in PAIRS or not name:
            continue
        key, _ = next((k, t) for n, k, t in STUDENTS if n == name)
        ref = cache.get((pair, key, "0.0", 0))
        for col, lam in enumerate(LAMBDAS):
            cur = cache.get((pair, key, lam, 0))
            if ref and cur:
                row[2 + col].value = round(overlap(ref, cur), 4)
                row[14].value = round(chance_level(cur), 4)
                filled["phi"] += 1
            seeds = [cache[(pair, key, lam, s)] for s in SEEDS if (pair, key, lam, s) in cache]
            if len(seeds) >= 2:
                vals = [overlap(seeds[i], seeds[j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
                row[8 + col].value = round(st.mean(vals), 4)
                filled["baseline"] += 1

    wb.save(args.workbook)
    print(f"{args.workbook}: " + ", ".join(f"{k} {v}" for k, v in sorted(filled.items())))


if __name__ == "__main__":
    main()
