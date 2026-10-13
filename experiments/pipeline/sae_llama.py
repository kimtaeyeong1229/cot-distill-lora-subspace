#!/usr/bin/env python3
import argparse
import gzip
import json
import random
import re
from pathlib import Path
import pandas as pd
import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import build_prompt
from config import (
    CELL_DIR,
    EXP as E,
    LAMBDAS,
    MATH500_EVAL,
    MODELS,
    PAIR_DIR as COT,
    PAIRS,
    REFLECTIVE,
    SEEDS,
)

MODEL = str(MODELS / "Llama-3.2-1B-Instruct")
SAE_DIR = MODELS / "saes" / "llama-3.2-1b-131k"
OUT = E / "sae_llama"
CELLS = {p: (CELL_DIR[f"{p}/Llama"], MATH500_EVAL[f"{p}/Llama"]) for p in PAIRS}
EDITS = ["lambda0.0_rm_pre", "graft2", "graft_orth2"]
PROBLEMS, HEAD_TOKENS, CHUNK = (200, 400, 1024)
INJECT = ["Wait, ", "Hmm, ", "Let me double-check. ", "Alternatively, ", "Actually, ", "Hold on, "]


def mask(text):
    text = REFLECTIVE.sub("", text)
    text = re.sub("(^|\\n|[.!?]\\s)\\s*[,.]\\s*", "\\1", text)
    return re.sub("[ \\t]{2,}", " ", text)


def inject(text, rng):
    parts = re.split("(?<=[.!?])\\s+", text)
    return " ".join((rng.choice(INJECT) + p if i % 3 == 1 else p for i, p in enumerate(parts)))


class TopKSAE:

    def __init__(self, layer, device):
        w = load_file(str(SAE_DIR / f"layers.{layer}.mlp" / "sae.safetensors"), device=device)
        self.k = json.loads((SAE_DIR / f"layers.{layer}.mlp" / "cfg.json").read_text())["k"]
        self.W_enc, self.b_enc = (w["encoder.weight"].float(), w["encoder.bias"].float())
        self.W_dec, self.b_dec = (w["W_dec"].float(), w["b_dec"].float())
        self.n = self.W_enc.shape[0]

    def encode(self, x):
        pre = torch.relu((x.float() - self.b_dec) @ self.W_enc.T + self.b_enc)
        return pre.topk(self.k, dim=-1)

    def decode(self, vals, idx):
        return torch.einsum("tk,tkd->td", vals, self.W_dec[idx]) + self.b_dec


def mlp_module(model, layer):
    return [m for n, m in model.named_modules() if n.endswith(f"layers.{layer}.mlp")][0]


class Reader:

    def __init__(self, layers):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
        self.saes = {L: TopKSAE(L, "cuda") for L in layers}
        self.captured, self.replace = ({}, {})

    def attach(self, path):
        if path:
            self.model = PeftModel.from_pretrained(self.model, path).eval()
        self.hooks = []
        for L in self.saes:

            def hook(_m, _i, out, L=L):
                self.captured[L] = out[0]
                if L in self.replace:
                    return self.replace[L](out)

            self.hooks.append(mlp_module(self.model, L).register_forward_hook(hook))

    def detach(self):
        for h in self.hooks:
            h.remove()
        if isinstance(self.model, PeftModel):
            self.model = self.model.unload()

    def ids(self, problem, response, max_resp=7680):
        p = self.tok(build_prompt(self.tok, problem), add_special_tokens=False)["input_ids"]
        r = self.tok(response, add_special_tokens=False)["input_ids"][:max_resp]
        return (torch.tensor([p + r], device="cuda"), len(p))

    @torch.no_grad()
    def forward(self, problem, response, max_resp=7680, logits=False):
        ids, start = self.ids(problem, response, max_resp)
        out = self.model(input_ids=ids, logits_to_keep=0 if logits else 1).logits[0]
        return (ids[0], start, out)

    @torch.no_grad()
    def feature_means(self, L, start, feats=None):
        x = self.captured[L][start:]
        sae, total = (self.saes[L], torch.zeros(self.saes[L].n, device="cuda", dtype=torch.float64))
        head = torch.zeros_like(total)
        for c in range(0, x.shape[0], CHUNK):
            v, i = sae.encode(x[c : c + CHUNK])
            total.index_add_(0, i.flatten(), v.flatten().double())
            if c < HEAD_TOKENS:
                h = min(HEAD_TOKENS - c, v.shape[0])
                head.index_add_(0, i[:h].flatten(), v[:h].flatten().double())
        n = max(x.shape[0], 1)
        total, head = (total / n, head / max(min(n, HEAD_TOKENS), 1))
        return (total, head) if feats is None else (total[feats], head[feats])


def nll(logits, ids, start):
    lp = torch.log_softmax(logits[start - 1 : -1].float(), -1)
    return -lp.gather(1, ids[start:, None]).mean().item()


def cmd_check(a):
    rd = Reader(a.layers)
    rows = json.loads((E / "pairA/llama1b/eval_math500_7680/lambda0.0_seed0.json").read_text())["records"][
        : a.n
    ]
    probs = {r["unique_id"]: r for r in pd.read_parquet(E / "data/math500.parquet").to_dict("records")}
    out = {}
    for name, path in [
        ("instruct", None),
        ("A_l0_s0", E / "pairA/llama1b/runs/lambda0.0_seed0/adapter"),
        ("A_l1_s0", E / "pairA/llama1b/runs/lambda1.0_seed0/adapter"),
        ("B_l0_s0", E / "pairB/llama1b/runs/lambda0.0_seed0/adapter"),
        ("C_l0_s0", E / "pairC/llama1b/runs/lambda0.0_seed0/adapter"),
    ]:
        rd.attach(path)
        res = {}
        for L in a.layers:
            ss_res = ss_tot = 0.0
            clean, spliced = ([], [])
            for r in rows:
                prob = probs[r["unique_id"]]["problem"]
                rd.replace.pop(L, None)
                ids, start, logits = rd.forward(prob, r["response_0"], a.max_resp, logits=True)
                x = rd.captured[L][start:].float()
                recon = torch.cat(
                    [rd.saes[L].decode(*rd.saes[L].encode(x[c : c + CHUNK])) for c in range(0, len(x), CHUNK)]
                )
                ss_res += (x - recon).pow(2).sum().item()
                ss_tot += (x - x.mean(0)).pow(2).sum().item()
                clean.append(nll(logits, ids, start))
                sae = rd.saes[L]
                rd.replace[L] = lambda out, sae=sae: (
                    lambda y: torch.cat(
                        [sae.decode(*sae.encode(y[c : c + CHUNK])) for c in range(0, len(y), CHUNK)]
                    ).to(out.dtype)[None]
                )(out[0])
                ids, start, logits = rd.forward(prob, r["response_0"], a.max_resp, logits=True)
                spliced.append(nll(logits, ids, start))
                rd.replace.pop(L, None)
            res[L] = {
                "fvu": ss_res / ss_tot,
                "nll_clean": sum(clean) / len(clean),
                "nll_spliced": sum(spliced) / len(spliced),
            }
            print(
                f"{name:9s} layer {L:2d}  FVU {res[L]['fvu']:.3f}  NLL {res[L]['nll_clean']:.3f} -> {res[L]['nll_spliced']:.3f}",
                flush=True,
            )
        out[name] = res
        rd.detach()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(out, indent=2))


def cmd_select(a):
    rd = Reader([a.layer])
    rd.attach(None)
    rng = random.Random(0)
    sets = {k: [] for k in ("pos", "neg", "inj", "mask")}
    for pair, tag in COT.items():
        rows = {
            lam: [json.loads(l) for l in open(E / tag / "cot" / f"lambda{lam}.jsonl")]
            for lam in ("0.0", "1.0")
        }
        ids = rng.sample(range(len(rows["0.0"])), a.n)
        for i in ids:
            p0, p1 = (rows["0.0"][i], rows["1.0"][i])
            sets["pos"].append((p0["problem"], p0["cot_response"]))
            sets["neg"].append((p1["problem"], p1["cot_response"]))
            sets["inj"].append((p1["problem"], inject(p1["cot_response"], rng)))
            sets["mask"].append((p0["problem"], mask(p0["cot_response"])))
    means = {}
    for k, texts in sets.items():
        acc = torch.zeros(rd.saes[a.layer].n, device="cuda", dtype=torch.float64)
        for prob, resp in texts:
            _, start, _ = rd.forward(prob, resp, a.max_resp)
            acc += rd.feature_means(a.layer, start)[0]
        means[k] = acc / len(texts)
        print(f"{k}: {len(texts)} traces read", flush=True)
    pos, neg, inj, msk = (means[k] for k in ("pos", "neg", "inj", "mask"))
    diff = pos - neg
    cand = torch.nonzero((diff > 0) & (pos > 3 * neg + 1e-06)).flatten()
    cand = cand[diff[cand].argsort(descending=True)][: a.top]
    inj_r, mask_r = (((inj - neg) / diff)[cand], ((msk - neg) / diff)[cand])
    keep = cand[(inj_r <= 0.3) & (mask_r >= 0.5)]
    cue = cand[inj_r >= 0.5]
    feats = [
        {
            "feature": int(f),
            "pos": pos[f].item(),
            "neg": neg[f].item(),
            "inject_ratio": ((inj - neg) / diff)[f].item(),
            "mask_ratio": ((msk - neg) / diff)[f].item(),
        }
        for f in cand
    ]
    res = {
        "layer": a.layer,
        "traces_per_side": len(sets["pos"]),
        "candidates": feats,
        "selected": [int(f) for f in keep],
        "cue": [int(f) for f in cue],
        "scale": {int(f): pos[f].item() for f in torch.cat([keep, cue])},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"select_L{a.layer}.json").write_text(json.dumps(res, indent=2))
    print(
        f"layer {a.layer}: {len(cand)} candidates, {len(keep)} kept (behaviour), {len(cue)} cue-driven",
        flush=True,
    )


def responses(cell, cond, seed):
    tag, bench = CELLS[cell]
    if cond.startswith("base"):
        lam = cond[4:]
        probs = {
            r["unique_id"]: r["problem"]
            for r in pd.read_parquet(E / "data/math500.parquet").to_dict("records")
        }
        recs = json.loads((E / tag / bench / f"lambda{lam}_seed{seed}.json").read_text())["records"][
            :PROBLEMS
        ]
        return (
            E / tag / "runs" / f"lambda{lam}_seed{seed}" / "adapter",
            [(probs[r["unique_id"]], r["response_0"]) for r in recs],
        )
    recs = [
        json.loads(l) for l in gzip.open(E / tag / f"eval_main_s{seed}" / f"{cond}.samples.jsonl.gz", "rt")
    ]
    return (
        E / tag / f"runs_edit_s{seed}" / cond / "adapter",
        [(r["problem"], r["text"]) for i, r in enumerate(recs) if i % 2 == 0][:PROBLEMS],
    )


def cmd_score(a):
    sel = json.loads((OUT / f"select_L{a.layer}.json").read_text())
    groups = {"selected": sel["selected"], "cue": sel["cue"]}
    allf = torch.tensor(sorted(set(sel["selected"]) | set(sel["cue"])), device="cuda")
    scale = torch.tensor([sel["scale"][str(int(f))] for f in allf], device="cuda", dtype=torch.float64)
    pos = {int(f): j for j, f in enumerate(allf)}
    rd = Reader([a.layer])
    conds = a.conds or [f"base{l}" for l in LAMBDAS] + EDITS
    out_dir = OUT / a.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond in conds:
        for seed in SEEDS:
            dest = out_dir / f"{cond}_s{seed}.json"
            if dest.exists():
                continue
            path, texts = responses(a.cell, cond, seed)
            rd.attach(path)
            recs = []
            for prob, resp in texts:
                row = {"words": len(resp.split()), "markers": len(REFLECTIVE.findall(resp))}
                for variant, text in (("orig", resp), ("masked", mask(resp))):
                    _, start, _ = rd.forward(prob, text, a.max_resp)
                    tot, head = rd.feature_means(a.layer, start, allf)
                    tot, head = (tot / scale, head / scale)
                    for g, fs in groups.items():
                        j = [pos[f] for f in fs]
                        row[f"{variant}_{g}"] = tot[j].mean().item() if j else float("nan")
                        row[f"{variant}_{g}_head"] = head[j].mean().item() if j else float("nan")
                    row[f"{variant}_tokens"] = int(rd.captured[a.layer].shape[0] - start)
                    row[f"{variant}_per_feature"] = [round(v, 5) for v in tot.tolist()]
                    row[f"{variant}_per_feature_head"] = [round(v, 5) for v in head.tolist()]
                recs.append(row)
            rd.detach()
            dest.write_text(
                json.dumps(
                    {
                        "cell": a.cell,
                        "cond": cond,
                        "seed": seed,
                        "layer": a.layer,
                        "features": [int(f) for f in allf],
                        "records": recs,
                    }
                )
            )
            m = lambda k: sum((r[k] for r in recs)) / len(recs)
            print(
                f"{a.cell} {cond:18s} s{seed}  index {m('orig_selected'):.3f}  masked {m('masked_selected'):.3f}  cue {m('orig_cue'):.3f}  markers/1k {1000 * sum((r['markers'] for r in recs)) / max(sum((r['words'] for r in recs)), 1):.1f}",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--layers", type=int, nargs="+", default=[5, 8, 11])
    c.add_argument("--n", type=int, default=10)
    s = sub.add_parser("select")
    s.add_argument("--layer", type=int, required=True)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--top", type=int, default=100)
    r = sub.add_parser("score")
    r.add_argument("--layer", type=int, required=True)
    r.add_argument("--cell", choices=CELLS, required=True)
    r.add_argument(
        "--conds", nargs="+", help="conditions to score (default: all lambdas + the three pilot edits)"
    )
    for p in (c, s, r):
        p.add_argument("--max_resp", type=int, default=7680)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    {"check": cmd_check, "select": cmd_select, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
