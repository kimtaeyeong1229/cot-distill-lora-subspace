# phi(lambda0, lambda) with the two adapters drawn from DIFFERENT seeds, so neither the
# pre- nor the post-threshold number can inherit a shared LoRA initialisation.
import json, re, statistics as st
from collections import defaultdict
from pathlib import Path
import torch
from safetensors import safe_open
from config import CELL_DIR, EXP as E, LAMBDAS as LAM, PAIR_KEY, PAIRS, RESULTS, SEEDS as S, STUDENTS
# (pair key, student label, cell directory); the output keys this file has always used, e.g. "D/Llama-1B"
LABEL = {"Qwen": "Qwen-0.5B", "Llama": "Llama-1B", "OLMo": "OLMo-1B"}
CELLS = [(PAIR_KEY[p], LABEL[s], CELL_DIR[f"{p}/{s}"]) for p in PAIRS for s in STUDENTS]
def basis(p):
    c=json.loads((p.parent/"adapter_config.json").read_text()); s=c["lora_alpha"]/c["r"]
    parts=defaultdict(dict)
    with safe_open(p,"pt") as h:
        for k in h.keys():
            m=re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)",k)
            if m: parts[(int(m.group(1)),m.group(2))]["A" if "lora_A" in k else "B"]=h.get_tensor(k).float()
    o={}
    for key,v in parts.items():
        if len(v)!=2: continue
        B,A=v["B"]*s,v["A"]
        Qb,Rb=torch.linalg.qr(B); Qa,Ra=torch.linalg.qr(A.T)
        U,Sv,_=torch.linalg.svd(Rb@Ra.T); o[key]=(Qb@U,float((Sv**2).sum().sqrt()))
    return o
def ov(x,y):
    ks=set(x)&set(y); w=sum(x[k][1] for k in ks)
    return sum(float((x[k][0].T@y[k][0]).pow(2).sum()/x[k][0].shape[1])*x[k][1] for k in ks)/w
out={}
for pair,stu,tag in CELLS:
    B={}
    for l in LAM:
        for s in S:
            p=E/tag/"runs"/f"lambda{l}_seed{s}"/"adapter"/"adapter_model.safetensors"
            if p.exists(): B[(l,s)]=basis(p)
    if len(B)<18: print(f"{pair}/{stu}: {len(B)}/18 skip",flush=True); continue
    print(f"\n=== {pair}/{stu}"); print(f"{'lam':5s} {'same-seed':>9s} {'cross-seed':>10s} {'seed null':>10s} {'verdict':>7s}")
    rows={}
    for l in LAM:
        same=[ov(B[("0.0",s)],B[(l,s)]) for s in S]
        xs=[ov(B[("0.0",i)],B[(l,j)]) for i in S for j in S if i!=j]
        null=[ov(B[(l,i)],B[(l,j)]) for i in S for j in S if i<j]
        v="below" if st.mean(xs)+st.stdev(xs)<st.mean(null)-st.stdev(null) else "above"
        print(f"{l:5s} {st.mean(same):9.3f} {st.mean(xs):9.3f} {st.mean(null):10.3f} {v:>5s}",flush=True)
        rows[l]=dict(same=st.mean(same),xseed=st.mean(xs),xseed_sd=st.stdev(xs),null=st.mean(null),null_sd=st.stdev(null),below=v=="below")
    out[f"{pair}/{stu}"]=rows
(RESULTS / "phi_xseed.json").write_text(json.dumps(out,indent=2))
