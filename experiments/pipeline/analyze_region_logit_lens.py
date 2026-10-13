#!/usr/bin/env python3
import json
import statistics as st
from collections import defaultdict
import torch
from safetensors import safe_open
from ablate_subspace import LAMBDAS, consensus
from analyze_region_anatomy import PRE, RUNS, SEEDS, load
from config import MODELS as MODEL_DIR, RESULTS as X, STUDENT_MODEL as MODELS

CUE_WORDS = [
    "Wait",
    "Hmm",
    "Actually",
    "Alternatively",
    "But",
    "Let",
    "Hold",
    "Verify",
    "Check",
    "Double",
    "Reconsider",
]
CORE = {"Wait", "Hmm", "Alternatively", "Actually"}
WRITE = ("o_proj", "down_proj")
POST_NORM = {"o_proj": "post_attention_layernorm", "down_proj": "post_feedforward_layernorm"}
N_RANDOM, N_DRAWS, TOPK, SHOW = (1000, 10000, 15, 5)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def weights(name):
    path = MODEL_DIR / name
    config = json.loads((path / "config.json").read_text())
    tensors = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if k in ("lm_head.weight", "model.embed_tokens.weight", "model.norm.weight") or "post_" in k:
                    tensors[k] = h.get_tensor(k).float()
    tied = bool(config.get("tie_word_embeddings")) or "lm_head.weight" not in tensors
    W_U = tensors["model.embed_tokens.weight"] if tied else tensors["lm_head.weight"]
    post = {}
    if config["architectures"][0].startswith("Olmo2"):
        for k, v in tensors.items():
            for mod, norm in POST_NORM.items():
                if k.endswith(f"{norm}.weight"):
                    post[int(k.split(".")[2]), mod] = v
    return (W_U.to(DEV), tensors["model.norm.weight"].to(DEV), post, tied, config)


def cue_ids(tok):
    ids, labels = ([], [])
    for w in CUE_WORDS:
        variants = [" " + w.lower()] if w == "Reconsider" else [" " + w, " " + w.lower(), w]
        for v in variants:
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1 and e[0] not in ids:
                ids.append(e[0])
                labels.append((w, v))
    return (ids, labels)


def read(W_U, scale, D, cues, core_mask):
    logits = W_U @ (scale[:, None] * D)
    c = logits[cues]
    pct = torch.stack([(logits < c[i : i + 1]).float().mean(0) for i in range(len(cues))]) * 100
    mean_pct = pct.mean(0)
    sign = torch.where(mean_pct >= 50, 1.0, -1.0)
    score = torch.maximum(mean_pct, 100 - mean_pct)
    core = pct[core_mask].mean(0)
    core_score = torch.where(sign > 0, core, 100 - core)
    oriented = logits * sign[None, :]
    top = oriented.topk(100, dim=0).indices
    hits = torch.isin(top, torch.tensor(cues, device=top.device)).sum(0)
    oc = c * sign[None, :]
    in_top = torch.stack([(oriented > oc[i : i + 1]).sum(0) < 100 for i in range(len(cues))])
    pct_signed = torch.where(sign[None, :] > 0, pct, 100 - pct)
    return {
        "score": score.cpu(),
        "sign": sign.cpu(),
        "core_score": core_score.cpu(),
        "hits": hits.cpu(),
        "in_top": in_top.cpu(),
        "pct": pct_signed.cpu(),
        "logits": logits,
    }


def by_word_factory(labels, null_for):
    words = list(dict.fromkeys((w for w, _ in labels)))

    def by_word(g, ks):
        out = {}
        for name, rows in (
            ("region", g["region_in_top"]),
            ("post", g["post_in_top"]),
            ("random", [null_for(k)["in_top"] for k in ks]),
        ):
            m = torch.stack(rows).mean(0)
            out[name] = {w: float(sum((m[i] for i, (x, _) in enumerate(labels) if x == w))) for w in words}
        return out

    return by_word


def summ(v):
    return (
        {"mean": st.mean(v), "median": st.median(v), "min": min(v), "max": max(v), "n": len(v)} if v else None
    )


def main():
    from transformers import AutoTokenizer

    gen = torch.Generator().manual_seed(1234)
    out = {"cells": {}}
    for short, name in MODELS.items():
        W_U, g_final, post_w, tied, config = weights(name)
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR / name))
        cues, labels = cue_ids(tok)
        core_mask = torch.tensor([w in CORE for w, _ in labels])
        d = config["hidden_size"]
        R = torch.randn(d, N_RANDOM, generator=gen)
        R = (R / R.norm(dim=0, keepdim=True)).to(DEV)
        null_cache = {}

        def null_for(key):
            ck = key if post_w else None
            if ck not in null_cache:
                scale = g_final * (post_w[key].to(DEV) if post_w else 1.0)
                r = read(W_U, scale, R, cues, core_mask)
                null_cache[ck] = {
                    "score": r["score"],
                    "core_score": r["core_score"],
                    "hits": r["hits"],
                    "in_top": r["in_top"].float().mean(1),
                }
            return null_cache[ck]

        by_word = by_word_factory(labels, null_for)

        def decode(i):
            return tok.decode([int(i)])

        for pair, cells in RUNS.items():
            tag = cells[short]
            cell = f"{pair}/{short}"
            cliff = LAMBDAS[len(PRE[pair])]
            post_l = LAMBDAS[LAMBDAS.index(cliff) :]
            ads = {(l, s): load(tag, l, s) for l in PRE[pair] + post_l for s in SEEDS}
            keys = sorted((k for k in set.intersection(*[set(a) for a in ads.values()]) if k[1] in WRITE))
            layers = sorted({k[0] for k in keys})
            third = {
                l: ["early", "middle", "late"][min(2, 3 * i // len(layers))] for i, l in enumerate(layers)
            }
            dirs = []
            groups = defaultdict(
                lambda: {
                    "region": [],
                    "post": [],
                    "region_core": [],
                    "post_core": [],
                    "region_hits": [],
                    "post_hits": [],
                    "region_in_top": [],
                    "post_in_top": [],
                    "region_vs_null": [],
                    "post_vs_null": [],
                    "null_draw": [],
                }
            )
            examples = []
            for key in keys:
                V, k = consensus([ads[l, s][key][0] for l in PRE[pair] for s in SEEDS], threshold=0.5)
                Vp, _ = consensus([ads[l, s][key][0] for l in post_l for s in SEEDS], dims=k)
                scale = g_final * (post_w[key].to(DEV) if post_w else 1.0)
                null = null_for(key)
                rr = read(W_U, scale, V.to(DEV), cues, core_mask)
                rp = read(W_U, scale, Vp.to(DEV), cues, core_mask)
                energy = []
                for j in range(k):
                    v = V[:, j : j + 1]
                    e = []
                    for s in SEEDS:
                        _, sB, A = ads["0.0", s][key]
                        G = A @ A.T
                        vb = v.T @ sB
                        e.append(float(vb @ G @ vb.T) / float(torch.trace(sB.T @ sB @ G)))
                    energy.append(st.mean(e))
                g = groups[third[key[0]]]
                for j in range(k):
                    sc = float(rr["score"][j])
                    g["region"].append(sc)
                    g["region_core"].append(float(rr["core_score"][j]))
                    g["region_hits"].append(int(rr["hits"][j]))
                    g["region_in_top"].append(rr["in_top"][:, j].float())
                    g["post_in_top"].append(rp["in_top"][:, j].float())
                    g["region_vs_null"].append(float((null["score"] < sc).float().mean()) * 100)
                    g["post"].append(float(rp["score"][j]))
                    g["post_core"].append(float(rp["core_score"][j]))
                    g["post_hits"].append(int(rp["hits"][j]))
                    g["post_vs_null"].append(
                        float((null["score"] < float(rp["score"][j])).float().mean()) * 100
                    )
                    g["null_draw"].append(null["score"])
                    dirs.append(
                        {
                            "layer": key[0],
                            "module": key[1],
                            "third": third[key[0]],
                            "index": j,
                            "lambda0_energy_share": energy[j],
                            "cue_score": sc,
                            "cue_sign": "plus" if float(rr["sign"][j]) > 0 else "minus",
                            "core_score": float(rr["core_score"][j]),
                            "top100_hits": int(rr["hits"][j]),
                            "pct_of_random": g["region_vs_null"][-1],
                            "cue_pct": {
                                f"{w}|{v}": float(rr["pct"][i, j]) for i, (w, v) in enumerate(labels)
                            },
                        }
                    )
                    if third[key[0]] == "late":
                        lg = rr["logits"][:, j]
                        examples.append(
                            (
                                len(dirs) - 1,
                                {
                                    "plus": [decode(i) for i in lg.topk(TOPK).indices],
                                    "minus": [decode(i) for i in (-lg).topk(TOPK).indices],
                                },
                            )
                        )
                del rr["logits"], rp["logits"]
            res = {}
            for th, g in groups.items():
                n = len(g["region"])
                draws = torch.stack(
                    [nd[torch.randint(0, N_RANDOM, (N_DRAWS,), generator=gen)] for nd in g["null_draw"]]
                )
                null_means = draws.mean(0)
                allnull = torch.cat(g["null_draw"])
                q95 = float(torch.quantile(allnull, 0.95))
                res[th] = {
                    "directions": n,
                    "region_score": summ(g["region"]),
                    "post_score": summ(g["post"]),
                    "random_score": {"mean": float(allnull.mean()), "q95": q95, "n": N_RANDOM},
                    "region_core_score": summ(g["region_core"]),
                    "post_core_score": summ(g["post_core"]),
                    "region_top100_hits_mean": st.mean(g["region_hits"]),
                    "post_top100_hits_mean": st.mean(g["post_hits"]),
                    "random_top100_hits_mean": float(
                        torch.cat([null_for(k)["hits"].float() for k in keys if third[k[0]] == th]).mean()
                    ),
                    "top100_hits_by_word": by_word(g, [k for k in keys if third[k[0]] == th]),
                    "region_share_above_random_q95": sum((x > q95 for x in g["region"])) / n,
                    "post_share_above_random_q95": sum((x > q95 for x in g["post"])) / n,
                    "region_mean_pct_of_random": st.mean(g["region_vs_null"]),
                    "post_mean_pct_of_random": st.mean(g["post_vs_null"]),
                    "p_region_mean_vs_random": float((null_means >= st.mean(g["region"])).float().mean()),
                    "p_post_mean_vs_random": float((null_means >= st.mean(g["post"])).float().mean()),
                }
                late_dirs = [x for x in dirs if x["third"] == th]
                w = sum((x["lambda0_energy_share"] for x in late_dirs))
                res[th]["region_score_energy_weighted"] = (
                    sum((x["cue_score"] * x["lambda0_energy_share"] for x in late_dirs)) / w
                )
            ex = dict(examples)
            by_energy = sorted(ex, key=lambda i: -dirs[i]["lambda0_energy_share"])[:SHOW]
            by_cue = sorted(ex, key=lambda i: -dirs[i]["cue_score"])[:2]
            show = lambda i: {**{k: v for k, v in dirs[i].items() if k != "cue_pct"}, "top_tokens": ex[i]}
            out["cells"][cell] = {
                "cliff": cliff,
                "post_lambdas": post_l,
                "by_third": res,
                "strongest_late_by_lambda0_energy": [show(i) for i in by_energy],
                "highest_cue_score_late": [show(i) for i in by_cue],
                "directions": dirs,
            }
            L = res["late"]
            print(
                f"{cell:8s} late: region {L['region_score']['mean']:.1f} (core {L['region_core_score']['mean']:.1f}, hits {L['region_top100_hits_mean']:.2f}) post {L['post_score']['mean']:.1f} (core {L['post_core_score']['mean']:.1f}) random {L['random_score']['mean']:.1f} q95 {L['random_score']['q95']:.1f} | region>q95 {L['region_share_above_random_q95']:.2f} post>q95 {L['post_share_above_random_q95']:.2f} | pct-of-random region {L['region_mean_pct_of_random']:.0f} post {L['post_mean_pct_of_random']:.0f} | p region {L['p_region_mean_vs_random']:.4f} post {L['p_post_mean_vs_random']:.4f}",
                flush=True,
            )
            for i in by_energy[:2]:
                print(
                    f"    L{dirs[i]['layer']} {dirs[i]['module']} e={dirs[i]['lambda0_energy_share']:.3f} cue {dirs[i]['cue_score']:.1f}: + {ex[i]['plus'][:8]} | - {ex[i]['minus'][:8]}",
                    flush=True,
                )
        out.setdefault("models", {})[short] = {
            "name": name,
            "tied_unembedding": tied,
            "olmo_post_norm_applied": bool(post_w),
            "cue_tokens": [{"word": w, "variant": v, "id": i} for (w, v), i in zip(labels, cues)],
            "layers": config["num_hidden_layers"],
            "hidden": config["hidden_size"],
        }
        del W_U
        if DEV == "cuda":
            torch.cuda.empty_cache()
    late = {c: v["by_third"]["late"] for c, v in out["cells"].items()}
    out["summary_late"] = {
        "region_mean_score": summ([v["region_score"]["mean"] for v in late.values()]),
        "post_mean_score": summ([v["post_score"]["mean"] for v in late.values()]),
        "random_mean_score": summ([v["random_score"]["mean"] for v in late.values()]),
        "region_mean_pct_of_random": summ([v["region_mean_pct_of_random"] for v in late.values()]),
        "post_mean_pct_of_random": summ([v["post_mean_pct_of_random"] for v in late.values()]),
        "cells_region_p_below_0.05": sum((v["p_region_mean_vs_random"] < 0.05 for v in late.values())),
        "cells_post_p_below_0.05": sum((v["p_post_mean_vs_random"] < 0.05 for v in late.values())),
        "cells_region_above_post": sum(
            (v["region_score"]["mean"] > v["post_score"]["mean"] for v in late.values())
        ),
        "cells_region_core_above_post": sum(
            (v["region_core_score"]["mean"] > v["post_core_score"]["mean"] for v in late.values())
        ),
        "cells_region_hits_above_post": sum(
            (v["region_top100_hits_mean"] > v["post_top100_hits_mean"] for v in late.values())
        ),
        "region_top100_hits_mean": summ([v["region_top100_hits_mean"] for v in late.values()]),
        "post_top100_hits_mean": summ([v["post_top100_hits_mean"] for v in late.values()]),
        "random_top100_hits_mean": summ([v["random_top100_hits_mean"] for v in late.values()]),
    }
    out["definitions"] = {
        "readout": "W_U (g_final * [g_post *] v); norm statistics (RMS division) ignored",
        "score": "mean cue-token vocabulary percentile on the better sign, max(p, 100 - p)",
        "random": f"{N_RANDOM} Gaussian unit directions in the hidden space, generator seed 1234",
        "post": "post-switch consensus region, dimension-matched to the region per module (rm_post)",
        "region": "eigenvalue > 0.5 of the mean projector over all pre-switch adapters and seeds",
        "core_cues": sorted(CORE),
        "modules": list(WRITE),
        "thirds": "layers split into thirds as in analyze_region_anatomy.py; 'late' is the last third",
    }
    print(json.dumps(out["summary_late"], indent=1))
    (X / "region_logit_lens.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
