#!/usr/bin/env python3
"""Summarize pilot_validity outputs into the quantities of overleaf/method_defensible.md."""

import json
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/lambda_lora_bits/pilot_validity")
CANARY_BITS = 64 * 5.169925001442312  # 64 chars over [a-z0-9]


def agg(samples, side, key="bits"):
    return sum(s[side][key] for s in samples)


def per_tok(samples, side, cat=None):
    k, t = ("bits", "tokens") if cat is None else (f"{cat}_bits", f"{cat}_tokens")
    return agg(samples, side, k) / max(1, agg(samples, side, t))


def W_tok(samples, cat=None):
    return per_tok(samples, "base", cat) - per_tok(samples, "tuned", cat)


lines = []
for lam in ("1.0", "0.0"):
    tpath = D / f"teacher_lambda{lam}.json"
    teacher = json.loads(tpath.read_text())["sets"] if tpath.exists() else {}
    for name in ("qwen0.5b", "llama1b"):
        p = D / f"student_{name}_lambda{lam}.json"
        if not p.exists():
            continue
        S = json.loads(p.read_text())["sets"]
        tr, he, mm, rs = S["train"], S["held"], S["mismatch"], S.get("resample", [])
        lines.append(f"\n## {name}  λ={lam}")
        n = lambda x: len(x)
        lines.append(f"- samples: train {n(tr)}, held {n(he)}, mismatch {n(mm)}, resample {n(rs)}")
        row = lambda lab, ss: (f"| {lab} | {agg(ss,'base')/len(ss):.0f} → {agg(ss,'tuned')/len(ss):.0f} | "
                               f"{per_tok(ss,'base'):.3f} → {per_tok(ss,'tuned'):.3f} | {W_tok(ss):+.3f} | "
                               f"{W_tok(ss,'math'):+.3f} | {W_tok(ss,'text'):+.3f} |")
        lines.append("| set | bits/sample (before → after) | bits/token | W/token | W/token math | W/token text |")
        lines.append("|---|---|---|---|---|---|")
        for lab, ss in (("train", tr), ("held", he), ("mismatch", mm), ("resample", rs)):
            if ss:
                lines.append(row(lab, ss))
        Ttr, Tho = agg(tr, "base", "tokens"), agg(he, "base", "tokens")
        Wtr, Who = agg(tr, "base") - agg(tr, "tuned"), agg(he, "base") - agg(he, "tuned")
        dW = Wtr - Ttr / Tho * Who
        lines.append(f"- ΔW (token-scaled, subset) = {dW:,.0f} bits, m = {dW / Wtr:.2f}")
        for cat in ("math", "text"):
            wt = agg(tr, "base", f"{cat}_bits") - agg(tr, "tuned", f"{cat}_bits")
            wh = agg(he, "base", f"{cat}_bits") - agg(he, "tuned", f"{cat}_bits")
            tt, th = agg(tr, "base", f"{cat}_tokens"), agg(he, "base", f"{cat}_tokens")
            d = wt - tt / th * wh
            lines.append(f"  - {cat}: token share train {tt / Ttr:.2f}; ΔW_{cat} = {d:,.0f} ({d / dW * 100 if dW else 0:.0f}% of ΔW)")
        wh, wm = W_tok(he), W_tok(mm)
        lines.append(f"- held-out gain per token: matched {wh:+.3f}, mismatched {wm:+.3f} → problem-conditional {wh - wm:+.3f}"
                     f" ({(wh - wm) / wh * 100 if wh > 0 else float('nan'):.0f}% of matched)")
        if rs:
            lines.append(f"- resample per token after training: train-original {per_tok(tr,'tuned'):.3f}, "
                         f"train-resample {per_tok(rs,'tuned'):.3f}, held {per_tok(he,'tuned'):.3f}")
        if teacher:
            for lab, ss in (("train", tr), ("held", he), ("resample", rs)):
                tset = {t["unique_id"]: t for t in teacher.get(lab, [])}
                pairs = [(s, tset[s["unique_id"]]) for s in ss if s["unique_id"] in tset]
                if not pairs:
                    continue
                fl = sum(t["floor_bits"] for _, t in pairs) / len(pairs)
                fl_lo = sum(t["floor_bits"] - t.get("artefact_bits", 0.0) for _, t in pairs) / len(pairs)
                tu = sum(s["tuned"]["bits"] for s, _ in pairs) / len(pairs)
                ba = sum(s["base"]["bits"] for s, _ in pairs) / len(pairs)
                below = sum(1 for s, t in pairs if s["tuned"]["bits"] < t["floor_bits"]) / len(pairs)
                miss = sum(t["nucleus_misses"] for _, t in pairs) / max(1, sum(t["tokens"] for _, t in pairs))
                lines.append(f"- generator floor [{lab}]: floor {fl:.0f}, student before {ba:.0f}, after {tu:.0f} bits/sample → "
                             f"M_G = {max(0.0, fl - tu):.1f} bits/sample (mean gap {fl - tu:+.1f}; without removed tokens {fl_lo - tu:+.1f}); samples below floor {below:.0%}; "
                             f"nucleus misses {miss:.4%}")

for rep in (1, 4, 16):
    p = D / f"canary_rep{rep}.json"
    if not p.exists():
        continue
    c = json.loads(p.read_text())["sets"]
    t, h = c["canary_train"], c["canary_held"]
    mb = lambda xs, k: sum(x[k] for x in xs) / len(xs)
    wt, wh = mb(t, "base") - mb(t, "tuned"), mb(h, "base") - mb(h, "tuned")
    lines.append(f"\n## canary x{rep}: base {mb(t,'base'):.0f}/{mb(h,'base'):.0f}, after train {mb(t,'tuned'):.0f}, held {mb(h,'tuned'):.0f}; "
                 f"W {wt:.0f}/{wh:.0f}; ΔW {wt - wh:.0f} bits/canary; strict (below {CANARY_BITS:.0f}) train {max(0, CANARY_BITS - mb(t,'tuned')):.0f}, "
                 f"held {max(0, CANARY_BITS - mb(h,'tuned')):.0f}")

print("\n".join(lines))
