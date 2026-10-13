import argparse
import json, re, statistics as st
from collections import defaultdict
from pathlib import Path
import torch
from safetensors import safe_open
from config import (
    CELL_DIR,
    EXP as E,
    LAMBDAS as LAM,
    PAIR_KEY,
    PAIRS,
    RESULTS as RESULTS_DIR,
    SEEDS,
    STUDENT_SHORT,
    STUDENTS,
    SWITCH,
)

CELLS = [(PAIR_KEY[p], STUDENT_SHORT[s], CELL_DIR[f"{p}/{s}"], SWITCH[p]) for p in PAIRS for s in STUDENTS]


def basis(p):
    c = json.loads((p.parent / "adapter_config.json").read_text())
    s = c["lora_alpha"] / c["r"]
    parts = defaultdict(dict)
    with safe_open(p, "pt") as h:
        for k in h.keys():
            m = re.search("layers\\.(\\d+)\\.(?:self_attn|mlp)\\.(\\w+_proj)", k)
            if m:
                parts[int(m.group(1)), m.group(2)]["A" if "lora_A" in k else "B"] = h.get_tensor(k).float()
    out = {}
    for key, v in parts.items():
        if len(v) != 2:
            continue
        B, A = (v["B"] * s, v["A"])
        Qb, Rb = torch.linalg.qr(B)
        Qa, Ra = torch.linalg.qr(A.T)
        U, S, _ = torch.linalg.svd(Rb @ Ra.T)
        out[key] = (Qb @ U, float((S**2).sum().sqrt()), B.shape[0])
    return out


RESULTS = {}
for pair, stu, tag, br in CELLS:
    ads = {}
    for lam in LAM:
        for sd in SEEDS:
            p = E / tag / "runs" / f"lambda{lam}_seed{sd}" / "adapter" / "adapter_model.safetensors"
            if p.exists():
                ads[lam, sd] = basis(p)
    if len(ads) < 18:
        print(pair, stu, len(ads), "of 18, skipped", flush=True)
        continue
    i = LAM.index(br)
    pre = [(l, s) for l in LAM[:i] for s in SEEDS]
    post = [(l, s) for l in LAM[i:] for s in SEEDS]
    keys = set.intersection(*[set(a) for a in ads.values()])
    acc = {l: 0.0 for l in LAM}
    cw = w = 0.0
    for key in keys:
        wt = st.mean([ads[t][key][1] for t in ads])
        Us = {t: ads[t][key][0] for t in ads}
        d = ads[pre[0]][key][2]

        def consensus(members):
            M = torch.cat([Us[u] for u in members], dim=1)
            Q, S, _ = torch.linalg.svd(M, full_matrices=False)
            ev = S**2 / len(members)
            k = max(1, int((ev > 0.5).sum()))
            return (Q[:, :k], k)

        Vk, k = consensus(pre)
        score = {}
        for t in post:
            score[t] = float((Us[t].T @ Vk).pow(2).sum() / k)
        for t in pre:
            V2, k2 = consensus([u for u in pre if u != t])
            score[t] = float((Us[t].T @ V2).pow(2).sum() / k2)
        for l in LAM:
            acc[l] += st.mean((score[l, sd] for sd in SEEDS)) * wt
        cw += Us[pre[0]].shape[1] / d * wt
        w += wt
    RESULTS[f"{pair}/{stu}"] = dict(transition=br, coverage={l: acc[l] / w for l in LAM}, chance=cw / w)
    print(pair, stu, br, " ".join((f"{acc[l] / w:.3f}" for l in LAM)), flush=True)
out = RESULTS_DIR / "coverage_by_lambda.json"
out.write_text(json.dumps(RESULTS, indent=2))
print(out)
