#!/usr/bin/env python3
import json
import math
import statistics as st
from pathlib import Path
from config import CELL_DIR, EXP as E, LAMBDAS as LAM, PAIR_DIR, PAIR_KEY, RESULTS as X, SEEDS, STUDENT_SHORT

PAIRS = {p: (PAIR_DIR[p], PAIR_KEY[p]) for p in PAIR_DIR}
STUDENTS = dict(STUDENT_SHORT)
RUNS = {p: {s: CELL_DIR[f"{p}/{s}"] for s in STUDENT_SHORT} for p in PAIR_DIR}
SWITCH_DROP, HELD_CHANGE, ON_LEVEL, OFF_LEVEL, MARKER_FLOOR = (50.0, 15.0, 50.0, 15.0, 2.0)


def classify(think, markers):
    labels = []
    for i in range(5):
        t0, t1 = (think[LAM[i]], think[LAM[i + 1]])
        if t0 - t1 >= SWITCH_DROP:
            labels.append("switch")
        elif labels and labels[-1] == "switch" and (t0 >= OFF_LEVEL) and (t0 - t1 >= OFF_LEVEL):
            labels.append("switch")
        elif (
            abs(t1 - t0) < HELD_CHANGE
            and min(t0, t1) >= ON_LEVEL
            and (markers[LAM[i + 1]] >= MARKER_FLOOR * markers["1.0"])
        ):
            labels.append("held")
        elif max(t0, t1) < OFF_LEVEL:
            labels.append("after")
        else:
            labels.append("unclassified")
    return labels


def two_seed_coverage():
    import torch
    from ablate_subspace import consensus, left_basis, read_adapter

    def load(tag, lam, seed):
        path = E / tag / "runs" / f"lambda{lam}_seed{seed}" / "adapter"
        config = json.loads((path / "adapter_config.json").read_text())
        scale = config["lora_alpha"] / config["r"]
        raw, parts = read_adapter(path / "adapter_model.safetensors")
        bases, norms = ({}, {})
        for key, names in parts.items():
            if len(names) == 2:
                B, A = (raw[names["B"]].float(), raw[names["A"]].float())
                bases[key] = left_basis(B, A, scale)
                norms[key] = float(torch.linalg.matrix_norm(scale * B @ A))
        return (bases, norms)

    out = {}
    for pair, cells in RUNS.items():
        for short, tag in cells.items():
            ads = {(l, s): load(tag, l, s) for l in LAM for s in SEEDS}
            keys = set.intersection(*[set(b) for b, _ in ads.values()])
            weight = {k: st.mean((n[k] for _, n in ads.values())) for k in keys}
            cov = {l: [] for l in LAM}
            local = {f"{LAM[i]}->{LAM[i + 1]}": [] for i in range(5)}
            chance = 0.0
            for held in SEEDS:
                members = [s for s in SEEDS if s != held]
                num = {l: 0.0 for l in LAM}
                den = 0.0
                for k in keys:
                    V, kv = consensus([ads["0.0", s][0][k] for s in members], threshold=0.75)
                    for l in LAM:
                        num[l] += weight[k] * float((ads[l, held][0][k].T @ V).pow(2).sum()) / kv
                    den += weight[k]
                    if held == SEEDS[0]:
                        U = ads["0.0", held][0][k]
                        chance += weight[k] * U.shape[1] / U.shape[0]
                for l in LAM:
                    cov[l].append(num[l] / den)
                for i in range(5):
                    a, b = (LAM[i], LAM[i + 1])
                    na = nb = 0.0
                    for k in keys:
                        V, kv = consensus([ads[a, s][0][k] for s in members], threshold=0.75)
                        na += weight[k] * float((ads[a, held][0][k].T @ V).pow(2).sum()) / kv
                        nb += weight[k] * float((ads[b, held][0][k].T @ V).pow(2).sum()) / kv
                    local[f"{a}->{b}"].append([na / den, nb / den])
            cell = f"{pair}/{short}"
            out[cell] = {
                "coverage": {l: st.mean(v) for l, v in cov.items()},
                "coverage_by_held_seed": cov,
                "chance": chance / den,
                "local_start_end": {
                    iv: [st.mean((x for x, _ in v)), st.mean((y for _, y in v))] for iv, v in local.items()
                },
                "local_change": {iv: st.mean((y - x for x, y in v)) for iv, v in local.items()},
            }
            print(
                f"two-seed region {cell:8s} " + " ".join((f"{out[cell]['coverage'][l]:.3f}" for l in LAM)),
                flush=True,
            )
    return out


PRE = {"A": ["0.0", "0.2"], "B": ["0.0", "0.2", "0.4"], "C": ["0.0", "0.2", "0.4"]}


def leave_lambda_out_coverage():
    import torch
    from ablate_subspace import consensus, left_basis, read_adapter

    def load(tag, lam, seed):
        path = E / tag / "runs" / f"lambda{lam}_seed{seed}" / "adapter"
        config = json.loads((path / "adapter_config.json").read_text())
        scale = config["lora_alpha"] / config["r"]
        raw, parts = read_adapter(path / "adapter_model.safetensors")
        bases, norms = ({}, {})
        for key, names in parts.items():
            if len(names) == 2:
                B, A = (raw[names["B"]].float(), raw[names["A"]].float())
                bases[key] = left_basis(B, A, scale)
                norms[key] = float(torch.linalg.matrix_norm(scale * B @ A))
        return (bases, norms)

    out = {}
    for pair, cells in RUNS.items():
        pre = PRE[pair]
        for short, tag in cells.items():
            ads = {(l, s): load(tag, l, s) for l in LAM for s in SEEDS}
            keys = set.intersection(*[set(b) for b, _ in ads.values()])
            weight = {k: st.mean((n[k] for _, n in ads.values())) for k in keys}
            den = sum(weight.values())
            score = {o: {l: 0.0 for l in LAM} for o in pre}
            dims = {o: 0.0 for o in pre}
            for k in keys:
                for o in pre:
                    V, kv = consensus([ads[l, s][0][k] for l in pre if l != o for s in SEEDS], threshold=0.5)
                    dims[o] += weight[k] * kv / den
                    for l in LAM:
                        score[o][l] += (
                            weight[k]
                            * st.mean((float((ads[l, s][0][k].T @ V).pow(2).sum()) / kv for s in SEEDS))
                            / den
                        )
            cov = {l: score[l][l] if l in pre else st.mean((score[o][l] for o in pre)) for l in LAM}
            out[f"{pair}/{short}"] = {"coverage": cov, "by_left_out_lambda": score, "region_dims": dims}
            print(
                f"leave-lambda-out {pair}/{short:6s} " + " ".join((f"{cov[l]:.3f}" for l in LAM)), flush=True
            )
    return out


def half_reference_retention():
    import torch
    from ablate_subspace import consensus, left_basis, read_adapter

    def load(tag, run):
        path = E / tag / "runs" / run / "adapter"
        config = json.loads((path / "adapter_config.json").read_text())
        scale = config["lora_alpha"] / config["r"]
        raw, parts = read_adapter(path / "adapter_model.safetensors")
        bases, norms = ({}, {})
        for key, names in parts.items():
            if len(names) == 2:
                B, A = (raw[names["B"]].float(), raw[names["A"]].float())
                bases[key] = left_basis(B, A, scale)
                norms[key] = float(torch.linalg.matrix_norm(scale * B @ A))
        return (bases, norms)

    halves = ["halfA", "halfA_seed1", "halfA_seed2", "halfB", "halfB_seed1", "halfB_seed2"]
    out = {}
    for pair, cells in RUNS.items():
        for short, tag in cells.items():
            full = {(l, s): load(tag, f"lambda{l}_seed{s}") for l in LAM for s in SEEDS}
            keys = set.intersection(*[set(b) for b, _ in full.values()])
            weight = {k: st.mean((n[k] for _, n in full.values())) for k in keys}
            den = sum(weight.values())
            res = {}
            for i in range(5):
                a, b = (LAM[i], LAM[i + 1])
                half = [load(tag, f"lambda{a}_{h}")[0] for h in halves]
                ref = nxt = 0.0
                for k in keys:
                    V, kv = consensus([full[a, s][0][k] for s in SEEDS], threshold=0.5)
                    ref += weight[k] * st.mean((float((h[k].T @ V).pow(2).sum()) / kv for h in half)) / den
                    nxt += (
                        weight[k]
                        * st.mean((float((full[b, s][0][k].T @ V).pow(2).sum()) / kv for s in SEEDS))
                        / den
                    )
                res[f"{a}->{b}"] = {"reference": ref, "next": nxt, "kept": nxt / ref}
            out[f"{pair}/{short}"] = res
            print(
                f"half reference {pair}/{short:6s} "
                + " ".join((f"{iv}:{v['kept']:.2f}" for iv, v in res.items())),
                flush=True,
            )
    return out


def summarize(values):
    return {"min": min(values), "max": max(values), "mean": st.mean(values), "n": len(values)}


def main():
    think_all = json.loads((X / "think_rate.json").read_text())
    teacher = json.loads((X / "teacher_density.json").read_text())
    res = json.loads((X / "phi_resample_s012.json").read_text())
    cov_all = json.loads((X / "coverage_by_lambda.json").read_text())
    seed_cov = two_seed_coverage()
    llo = leave_lambda_out_coverage()
    half_ref = half_reference_retention()
    out = {
        "criteria": {
            "switch_drop_points": SWITCH_DROP,
            "held_change_points": HELD_CHANGE,
            "held_min_think": ON_LEVEL,
            "after_max_think": OFF_LEVEL,
            "held_marker_floor_x_lambda1": MARKER_FLOOR,
            "switch_continuation": "directly after a switch, think rate falls from >= 15% by >= 15 points",
        },
        "pairs": {},
        "two_seed_region": seed_cov,
        "leave_lambda_out_region": llo,
        "half_reference": half_ref,
        "summary": {},
    }
    pooled = {
        c: {
            "coverage": [],
            "coverage_per_log_len": [],
            "R": [],
            "R_per_log_len": [],
            "seed_cov": [],
            "local": [],
            "llo": [],
            "kept": [],
            "log_len": [],
        }
        for c in ("held", "switch", "after")
    }
    for p, (split, key) in PAIRS.items():
        stats = json.loads((E / split / "splits" / "split_stats.json").read_text())
        n = stats["n_train"]
        tokens = {l: stats["per_lambda"][l]["train_tokens"] / n for l in LAM}
        think = {l: 100 * think_all[key]["rates"][l] for l in LAM}
        markers = {l: teacher[p]["teacher"][l] for l in LAM}
        labels = classify(think, markers)
        cov = {s: cov_all[f"{key}/{full}"]["coverage"] for s, full in STUDENTS.items()}
        R = {
            s: {l: res[f"{key}/{full}"]["rows"][l]["retention"] for l in LAM} for s, full in STUDENTS.items()
        }
        sc = {s: seed_cov[f"{p}/{s}"]["coverage"] for s in STUDENTS}
        lo = {s: llo[f"{p}/{s}"]["coverage"] for s in STUDENTS}
        intervals = []
        for i in range(5):
            a, b = (LAM[i], LAM[i + 1])
            d_len = math.log(tokens[a] / tokens[b])
            row = {
                "interval": f"{a}->{b}",
                "class": labels[i],
                "tokens": [tokens[a], tokens[b]],
                "log_len_drop": d_len,
                "think": [think[a], think[b]],
                "think_change": think[b] - think[a],
                "markers": [markers[a], markers[b]],
                "marker_change": markers[b] - markers[a],
                "marker_log_ratio": math.log(markers[a] / markers[b]),
                "coverage_change": {s: cov[s][b] - cov[s][a] for s in STUDENTS},
                "R_change": {s: R[s][b] - R[s][a] for s in STUDENTS},
                "two_seed_coverage_change": {s: sc[s][b] - sc[s][a] for s in STUDENTS},
                "leave_lambda_out_coverage_change": {s: lo[s][b] - lo[s][a] for s in STUDENTS},
                "kept_vs_half_reference": {s: half_ref[f"{p}/{s}"][f"{a}->{b}"]["kept"] for s in STUDENTS},
                "local_coverage_change": {
                    s: seed_cov[f"{p}/{s}"]["local_change"][f"{a}->{b}"] for s in STUDENTS
                },
            }
            row["coverage_change_per_log_len"] = {s: v / d_len for s, v in row["coverage_change"].items()}
            row["R_change_per_log_len"] = {s: v / d_len for s, v in row["R_change"].items()}
            intervals.append(row)
            c = labels[i]
            if c in pooled:
                pooled[c]["coverage"] += row["coverage_change"].values()
                pooled[c]["coverage_per_log_len"] += row["coverage_change_per_log_len"].values()
                pooled[c]["R"] += row["R_change"].values()
                pooled[c]["R_per_log_len"] += row["R_change_per_log_len"].values()
                pooled[c]["seed_cov"] += row["two_seed_coverage_change"].values()
                pooled[c]["local"] += row["local_coverage_change"].values()
                pooled[c]["llo"] += row["leave_lambda_out_coverage_change"].values()
                pooled[c]["kept"] += row["kept_vs_half_reference"].values()
                pooled[c]["log_len"].append(d_len)
            print(
                f"{p} {a}->{b} {c:7s} loglen {d_len:.2f} think {think[a]:5.1f}->{think[b]:5.1f} markers {markers[a]:5.2f}->{markers[b]:5.2f} | cov "
                + " ".join((f"{v:+.3f}" for v in row["coverage_change"].values()))
                + " | R "
                + " ".join((f"{v:+.3f}" for v in row["R_change"].values()))
                + " | 2-seed cov "
                + " ".join((f"{v:+.3f}" for v in row["two_seed_coverage_change"].values()))
                + " | local "
                + " ".join((f"{v:+.3f}" for v in row["local_coverage_change"].values()))
            )
        out["pairs"][p] = {
            "n_train": n,
            "tokens": tokens,
            "think_rate": think,
            "teacher_markers": markers,
            "coverage": cov,
            "R": R,
            "two_seed_coverage": sc,
            "leave_lambda_out_coverage": lo,
            "intervals": intervals,
        }
    for c, v in pooled.items():
        out["summary"][c] = {
            "intervals": sum((1 for p in out["pairs"].values() for r in p["intervals"] if r["class"] == c)),
            "log_len_drop": summarize(v["log_len"]),
            "coverage_change": summarize(v["coverage"]),
            "coverage_change_per_log_len": summarize(v["coverage_per_log_len"]),
            "R_change": summarize(v["R"]),
            "R_change_per_log_len": summarize(v["R_per_log_len"]),
            "two_seed_coverage_change": summarize(v["seed_cov"]),
            "local_coverage_change": summarize(v["local"]),
            "leave_lambda_out_coverage_change": summarize(v["llo"]),
            "kept_vs_half_reference": summarize(v["kept"]),
        }
        s = out["summary"][c]
        print(
            f"{c:7s} log-len drop {s['log_len_drop']['min']:.2f}..{s['log_len_drop']['max']:.2f} | coverage {s['coverage_change']['min']:+.3f}..{s['coverage_change']['max']:+.3f} (per log-len {s['coverage_change_per_log_len']['min']:+.2f}..{s['coverage_change_per_log_len']['max']:+.2f}) | R {s['R_change']['min']:+.3f}..{s['R_change']['max']:+.3f} | 2-seed cov {s['two_seed_coverage_change']['min']:+.3f}..{s['two_seed_coverage_change']['max']:+.3f} | local {s['local_coverage_change']['min']:+.3f}..{s['local_coverage_change']['max']:+.3f}"
        )
    (X / "length_confound.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
