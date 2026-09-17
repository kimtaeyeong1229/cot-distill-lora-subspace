# phi(lambda0, lambda) with the two adapters drawn from DIFFERENT seeds, so neither the
# pre- nor the post-threshold number can inherit a shared LoRA initialisation.
import json, re, statistics as st
from collections import defaultdict
from pathlib import Path
import torch
from safetensors import safe_open
E=Path("experiments/lambda_lora_bits"); LAM=["0.0","0.2","0.4","0.6","0.8","1.0"]; S=[0,1,2]
CELLS=[("A","Qwen-0.5B","sweep"),("A","Llama-1B","sweep_llama"),("A","OLMo-1B","sweep_olmo2"),
       ("B","Qwen-0.5B","pairB/qwen0.5b"),("B","Llama-1B","pairB/llama1b"),("B","OLMo-1B","pairB/olmo1b"),
       ("D","Qwen-0.5B","pairD/qwen0.5b"),("D","Llama-1B","pairD/llama1b"),("D","OLMo-1B","pairD/olmo1b")]
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
    print(f"\n=== {pair}/{stu}"); print(f"{'lam':5s} {'같은시드':>9s} {'교차시드':>9s} {'seed기준선':>10s} {'판정':>5s}")
    rows={}
    for l in LAM:
        same=[ov(B[("0.0",s)],B[(l,s)]) for s in S]
        xs=[ov(B[("0.0",i)],B[(l,j)]) for i in S for j in S if i!=j]
        null=[ov(B[(l,i)],B[(l,j)]) for i in S for j in S if i<j]
        v="아래" if st.mean(xs)+st.stdev(xs)<st.mean(null)-st.stdev(null) else "위"
        print(f"{l:5s} {st.mean(same):9.3f} {st.mean(xs):9.3f} {st.mean(null):10.3f} {v:>5s}",flush=True)
        rows[l]=dict(same=st.mean(same),xseed=st.mean(xs),xseed_sd=st.stdev(xs),null=st.mean(null),null_sd=st.stdev(null),below=v=="아래")
    out[f"{pair}/{stu}"]=rows
Path("experiments/phi_xseed.json").write_text(json.dumps(out,indent=2))
