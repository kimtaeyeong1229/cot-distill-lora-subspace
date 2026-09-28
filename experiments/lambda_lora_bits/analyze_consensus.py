"""Is there a fixed set of directions that every pre-threshold adapter uses and every post-threshold one drops?

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

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--side", choices=["pre", "post"], default="pre",
                help="which side builds the consensus region; the other side is scored against it")
ARGS = ap.parse_args()
IN, OUT = ("문턱前", "문턱後") if ARGS.side == "pre" else ("문턱後", "문턱前")
print(f"{'칸':14s} {'절벽':>4s} | {'합의차원':>8s} {IN + '(LOO)':>11s} {OUT:>8s} {'무작위':>7s}", flush=True)
RESULTS = {}
for pair, stu, tag, br in CELLS:
    ads = {}
    for lam in LAM:
        for sd in SEEDS:
            p = E/tag/"runs"/f"lambda{lam}_seed{sd}"/"adapter"/"adapter_model.safetensors"
            if p.exists(): ads[(lam, sd)] = basis(p)
    if len(ads) < 18:
        print(f"{pair+'/'+stu:14s} {len(ads)}/18 건너뜀", flush=True); continue
    i = LAM.index(br)
    below, above = [(l, s) for l in LAM[:i] for s in SEEDS], [(l, s) for l in LAM[i:] for s in SEEDS]
    pre, post = (below, above) if ARGS.side == "pre" else (above, below)
    keys = set.intersection(*[set(a) for a in ads.values()])
    kw = pw = qw = cw = w = 0.0
    for key in keys:
        wt = st.mean([ads[t][key][1] for t in ads])
        Us = {t: ads[t][key][0] for t in ads}; d = ads[pre[0]][key][2]

        def consensus(members):
            """Eigenvectors of mean(U U^T) without forming a d x d matrix: stacking the bases gives M with
            M M^T = sum(U U^T), so M's left singular vectors are the eigenvectors and s^2/n the eigenvalues."""
            M = torch.cat([Us[u] for u in members], dim=1)
            Q, S, _ = torch.linalg.svd(M, full_matrices=False)
            ev = (S ** 2) / len(members)
            k = max(1, int((ev > 0.5).sum()))
            return Q[:, :k], k

        Vk, k = consensus(pre)
        qs = [float((Us[t].T @ Vk).pow(2).sum() / k) for t in post]
        ps = []
        for t in pre:
            V2, k2 = consensus([u for u in pre if u != t])
            ps.append(float((Us[t].T @ V2).pow(2).sum() / k2))
        # a random rank-r adapter covers r/d of any fixed k-dim region in expectation -- the score is
        # normalised by k, so the chance level is r/d, not k/d
        r_ = Us[pre[0]].shape[1]
        kw += k*wt; pw += st.mean(ps)*wt; qw += st.mean(qs)*wt; cw += (r_/d)*wt; w += wt
    print(f"{pair+'/'+stu:14s} {br:>4s} | {kw/w:8.1f} {pw/w:11.3f} {qw/w:8.3f} {cw/w:7.3f}", flush=True)
    RESULTS[f"{pair}/{stu}"] = dict(transition=br, dims=kw/w, in_loo=pw/w, out=qw/w, chance=cw/w,
                                    built_from=len(pre), side=ARGS.side)
out = E.parent / f"consensus_{ARGS.side}.json"
out.write_text(json.dumps(RESULTS, indent=2)); print(out)
