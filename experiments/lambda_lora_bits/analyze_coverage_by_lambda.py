"""Per-lambda coverage of the consensus region (companion to analyze_consensus.py).

Is there a fixed set of directions that every pre-threshold adapter uses and every post-threshold one drops?

Pairwise overlap only says two runs share 2-3 of their 16 directions, too little to call any one region
"the" region. So take the mean of the rank-16 projectors U U^T over all pre-threshold adapters (every seed,
every lambda below the cliff): an eigenvalue near 1 marks a direction nearly all of them contain, and those
eigenvectors are the consensus region. Pre-threshold adapters are scored leave-one-out, so the tested
adapter never helped build the region it is scored against.
"""
import argparse
import json, re, statistics as st
from collections import defaultdict
from pathlib import Path
import torch
from safetensors import safe_open

E = Path("experiments/lambda_lora_bits"); LAM = ["0.0","0.2","0.4","0.6","0.8","1.0"]; SEEDS = [0,1,2]
# the transition each cell shows: the first large drop in retention (analyze_resample.py). Pair D drops in
# two steps of almost equal size, so all three of its cells use the first one, 0.6, where the teacher
# already emits </think> only 27% of the time -- the 0.6 adapters are not "before" the transition
CELLS = [
    ("A", "Qwen2.5-0.5B", "sweep", "0.4"),
    ("A", "Llama-3.2-1B", "sweep_llama", "0.4"),
    ("A", "OLMo-2-1B", "sweep_olmo2", "0.4"),
    ("B", "Qwen2.5-0.5B", "pairB/qwen0.5b", "0.6"),
    ("B", "Llama-3.2-1B", "pairB/llama1b", "0.6"),
    ("B", "OLMo-2-1B", "pairB/olmo1b", "0.6"),
    ("D", "Qwen2.5-0.5B", "pairD/qwen0.5b", "0.6"),
    ("D", "Llama-3.2-1B", "pairD/llama1b", "0.6"),
    ("D", "OLMo-2-1B", "pairD/olmo1b", "0.6"),
]
def basis(p):
    c = json.loads((p.parent/"adapter_config.json").read_text()); s = c["lora_alpha"]/c["r"]
    parts = defaultdict(dict)
    with safe_open(p, "pt") as h:
        for k in h.keys():
            m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)", k)
            if m: parts[(int(m.group(1)), m.group(2))]["A" if "lora_A" in k else "B"] = h.get_tensor(k).float()
    out = {}
    for key, v in parts.items():
        if len(v) != 2: continue
        B, A = v["B"]*s, v["A"]
        Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
        U, S, _ = torch.linalg.svd(Rb @ Ra.T)
        out[key] = (Qb @ U, float((S**2).sum().sqrt()), B.shape[0])
    return out


# Same region and scoring as analyze_consensus.py --side pre, but the coverage is reported per lambda, so a figure
# can show the region filling every pre-switch adapter and emptying at the switch. Pre-switch adapters are still
# scored leave-one-out; the cell-level means equal in_loo / out in consensus_pre.json.
RESULTS = {}
for pair, stu, tag, br in CELLS:
    ads = {}
    for lam in LAM:
        for sd in SEEDS:
            p = E/tag/"runs"/f"lambda{lam}_seed{sd}"/"adapter"/"adapter_model.safetensors"
            if p.exists(): ads[(lam, sd)] = basis(p)
    if len(ads) < 18:
        print(pair, stu, len(ads), "of 18, skipped", flush=True); continue
    i = LAM.index(br)
    pre = [(l, s) for l in LAM[:i] for s in SEEDS]
    post = [(l, s) for l in LAM[i:] for s in SEEDS]
    keys = set.intersection(*[set(a) for a in ads.values()])
    acc = {l: 0.0 for l in LAM}; cw = w = 0.0
    for key in keys:
        wt = st.mean([ads[t][key][1] for t in ads])
        Us = {t: ads[t][key][0] for t in ads}; d = ads[pre[0]][key][2]

        def consensus(members):
            M = torch.cat([Us[u] for u in members], dim=1)
            Q, S, _ = torch.linalg.svd(M, full_matrices=False)
            ev = (S ** 2) / len(members)
            k = max(1, int((ev > 0.5).sum()))
            return Q[:, :k], k

        Vk, k = consensus(pre)
        score = {}
        for t in post:
            score[t] = float((Us[t].T @ Vk).pow(2).sum() / k)
        for t in pre:
            V2, k2 = consensus([u for u in pre if u != t])
            score[t] = float((Us[t].T @ V2).pow(2).sum() / k2)
        for l in LAM:
            acc[l] += st.mean(score[(l, sd)] for sd in SEEDS) * wt
        cw += (Us[pre[0]].shape[1] / d) * wt; w += wt
    RESULTS[f"{pair}/{stu}"] = dict(transition=br, coverage={l: acc[l] / w for l in LAM}, chance=cw / w)
    print(pair, stu, br, " ".join(f"{acc[l]/w:.3f}" for l in LAM), flush=True)
out = E.parent / "coverage_by_lambda.json"
out.write_text(json.dumps(RESULTS, indent=2)); print(out)
