#!/usr/bin/env python3
"""The Table 27 structure measures on the students' own outputs, per edit (review R2-W2, read only).

Texts: the first sample of each of the first 200 MATH-500 problems (the response whose text every Section 5
density uses), seeds 0-2, loaded as in analyze_removal_vs_untrained.py:
  lambda0       unedited lambda = 0 adapter             (<cell>/<bench>/lambda0.0_seed<s>.json response_0)
  rm_pre        lambda = 0 with the shared region removed  (<cell>/eval_main_s<s>/lambda0.0_rm_pre)
  rm_decisive   lambda = 0 with the same energy removed from other directions (lambda0.0_rm_decisive)
  graft2        lambda = 1 adapter with the region of the lambda = 0 adapter grafted at beta = 2 (graft2)
  ampV          lambda = 0 with the region amplified at fixed norm (ampV; the amplification arm, extra)
  lambda1       unedited lambda = 1 adapter             (<cell>/<bench>/lambda1.0_seed<s>.json response_0)
  untrained     untrained instruct student              (base_eval/<model>/eval_main_s<s>/base.samples.jsonl.gz)

Measures (analyze_teacher_structure.py definitions and regexes, applied per seed and then averaged over seeds;
words = whitespace tokens, "per 1k words" pools markers and words over the 200 responses):
  a_markers_per_1k                      shared REFLECTIVE pattern
  c_markers_per_1k.{hesitation,verification,alternative,other}   Table 27 bins of the REFLECTIVE matches
  c_attempts_per_solution               1 + ALT_APPROACH phrases per response; c_any_alternative_approach
  d_any_verification                    share with at least one VERIFY phrase
  e_opens_monologue                     share whose first word is Okay / Alright / Hmm / So (Table 27);
                                        opens_first_word gives Okay, Alright, Hmm, So and "Let me" separately
  e_monologue_sentence_share            share of sentences starting I / Let me / So / Okay / Hmm
  b_boxed_recap                         recap format without tags: at least two \\boxed and at least 30 words
                                        after the end of the first \\boxed (the tag-free proxy of Table 27, which
                                        agrees with the </think>-based recap on the teacher texts);
                                        b_boxed_recap_completed is the same share among responses that stopped
                                        before the generation limit, since a loop can repeat \\boxed;
                                        b_words_after_first_boxed is the pooled share of words after the first \\boxed
Every measure except the recap ones is also computed on the first 300 words of each response ("head"),
where a repetition loop cannot dilute or inflate the counts; the whole-response version is "whole".

Readings, per cell and measure (on seed means):
  kept_vs_untrained   (cond - untrained) / (lambda0 - untrained): share of the lambda = 0 adapter's excess over
                      the untrained student that a lambda = 0 edit keeps (rm_pre, rm_decisive, ampV)
  recovery_vs_lambda1 (graft2 - lambda1) / (lambda0 - lambda1): share of the lambda1 -> lambda0 gap the graft
                      restores
Both are left out (None) when the denominator is below a small floor.

Writes experiments/student_structure.json.
"""

import gzip
import json
import re
import statistics as st

from analyze_main_density import CELLS, REFLECTIVE
from analyze_teacher_structure import (ALT_APPROACH, BIN, BOXED, MIN_RECAP_WORDS, MONO, OPEN_MONO,
                                       VERIFY, first_boxed_end, sentences)
from config import EVAL_PROBLEMS as PROBLEMS, EVAL_SAMPLES as SAMPLES, EXP as E, HEAD_WORDS as HEAD, RESULTS, SEEDS
from config import STUDENT_MODEL as MODELS

EDITS = {"rm_pre": "lambda0.0_rm_pre", "rm_decisive": "lambda0.0_rm_decisive", "graft2": "graft2", "ampV": "ampV"}
CONDS = ["lambda0", "rm_pre", "rm_decisive", "graft2", "ampV", "lambda1", "untrained"]
FIRST_WORDS = ["Okay", "Alright", "Hmm", "So", "Let me"]
FLOOR = {"per_1k": 0.2, "share": 0.02, "attempts": 0.02}
WORD = re.compile(r"\S+")


def head(text, n=HEAD):
    """The text up to the end of its n-th whitespace word, keeping the original spacing and newlines."""
    for i, m in enumerate(WORD.finditer(text)):
        if i == n - 1:
            return text[:m.end()]
    return text


def first_word(text):
    t = text.lstrip()
    if t.startswith("Let me"):
        return "Let me"
    w = t.split(maxsplit=1)
    return w[0].rstrip(",.!") if w else ""


def measures(texts, completed, whole):
    n, total_words = len(texts), sum(len(t.split()) for t in texts)
    bins = {b: 0 for b in ("hesitation", "verification", "alternative", "other")}
    markers = mono = sents = verify = 0
    alts = []
    for t in texts:
        for m in REFLECTIVE.findall(t):
            bins[BIN[m.lower().replace("\n", " ")]] += 1
            markers += 1
        alts.append(len(ALT_APPROACH.findall(t)))
        verify += bool(VERIFY.search(t))
        ss = sentences(t)
        sents += len(ss)
        mono += sum(bool(MONO.match(s)) for s in ss)
    fw = [first_word(t) for t in texts]
    out = {"mean_words": total_words / n,
           "a_markers_per_1k": 1000 * markers / max(total_words, 1),
           "c_markers_per_1k": {b: 1000 * v / max(total_words, 1) for b, v in bins.items()},
           "c_attempts_per_solution": 1 + st.mean(alts),
           "c_any_alternative_approach": sum(a > 0 for a in alts) / n,
           "d_any_verification": verify / n,
           "e_monologue_sentence_share": mono / max(sents, 1),
           "e_opens_monologue": sum(w in OPEN_MONO for w in fw) / n,
           "opens_first_word": {w: sum(x == w for x in fw) / n for w in FIRST_WORDS}}
    if whole:
        recap, after, recap_done, n_done = 0, 0, 0, 0
        for t, done in zip(texts, completed):
            end = first_boxed_end(t)
            r = False
            if end >= 0:
                tail = len(t[end:].split())
                after += tail
                r = tail >= MIN_RECAP_WORDS and BOXED in t[end:]
            recap += r
            if done:
                n_done += 1
                recap_done += r
        out.update({"b_boxed_recap": recap / n, "b_words_after_first_boxed": after / max(total_words, 1),
                    "b_boxed_recap_completed": recap_done / n_done if n_done else None, "completed_first": n_done / n})
    return out


def load_samples(path):
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    assert len(rows) == PROBLEMS * SAMPLES, path
    firsts = [rows[p * SAMPLES] for p in range(PROBLEMS)]
    return [r["text"] for r in firsts], [r["finish_reason"] == "stop" for r in firsts]


def load_records(path):
    recs = json.loads(path.read_text())["records"][:PROBLEMS]
    return [r["response_0"] for r in recs], [not r["truncated"][0] for r in recs]


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        elif v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)):
            out[prefix + k] = v
    return out


def unflat(d):
    out = {}
    for k, v in d.items():
        node = out
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return out


def floor_of(key):
    if "per_1k" in key:
        return FLOOR["per_1k"]
    return FLOOR["attempts"] if "attempts" in key else FLOOR["share"]


def main():
    out = {"cells": {}}
    for cell, (tag, bench) in CELLS.items():
        short = cell.split("/")[1]
        per_seed = {c: [] for c in CONDS}
        for s in SEEDS:
            texts = {"lambda0": load_records(E / tag / bench / f"lambda0.0_seed{s}.json"),
                     "lambda1": load_records(E / tag / bench / f"lambda1.0_seed{s}.json"),
                     "untrained": load_samples(E / "base_eval" / MODELS[short] / f"eval_main_s{s}" / "base.samples.jsonl.gz")}
            for c, name in EDITS.items():
                texts[c] = load_samples(E / tag / f"eval_main_s{s}" / f"{name}.samples.jsonl.gz")
            for c in CONDS:
                t, done = texts[c]
                per_seed[c].append({"whole": measures(t, done, True),
                                    "head": measures([head(x) for x in t], done, False)})
        row = {}
        for c in CONDS:
            fl = [flat(x) for x in per_seed[c]]
            mean = {k: st.mean(f[k] for f in fl if f.get(k) is not None) if any(f.get(k) is not None for f in fl) else None
                    for k in fl[0]}
            row[c] = {"mean": unflat(mean), "seeds": [unflat(f) for f in fl]}
        m = {c: flat(row[c]["mean"]) for c in CONDS}
        kept, recov = {}, {}
        for k in m["lambda0"]:
            if any(m[c][k] is None for c in CONDS):
                for c in ("rm_pre", "rm_decisive", "ampV"):
                    kept.setdefault(c, {})[k] = None
                recov[k] = None
                continue
            gap_u = m["lambda0"][k] - m["untrained"][k]
            gap_1 = m["lambda0"][k] - m["lambda1"][k]
            for c in ("rm_pre", "rm_decisive", "ampV"):
                kept.setdefault(c, {})[k] = (m[c][k] - m["untrained"][k]) / gap_u if abs(gap_u) >= floor_of(k) else None
            recov[k] = (m["graft2"][k] - m["lambda1"][k]) / gap_1 if abs(gap_1) >= floor_of(k) else None
        row["kept_vs_untrained"] = {c: unflat(v) for c, v in kept.items()}
        row["recovery_vs_lambda1"] = {"graft2": unflat(recov)}
        out["cells"][cell] = row
        h = {c: m[c] for c in CONDS}
        print(f"{cell:8s} head: " + " | ".join(
            f"{c} hes {h[c]['head.c_markers_per_1k.hesitation']:5.1f} ver {h[c]['head.c_markers_per_1k.verification']:4.1f}"
            f" alt {h[c]['head.c_markers_per_1k.alternative']:4.1f} att {h[c]['head.c_attempts_per_solution']:.2f}"
            f" vany {h[c]['head.d_any_verification']:.2f} open {h[c]['head.e_opens_monologue']:.2f}"
            f" mono {h[c]['head.e_monologue_sentence_share']:.2f} recap {h[c]['whole.b_boxed_recap']:.2f}"
            for c in CONDS), flush=True)

    # nine-cell and per-pair summaries of the seed means and of the two readings
    cells = list(out["cells"])
    keys = list(flat(out["cells"][cells[0]]["lambda0"]["mean"]))
    groups = {"all": cells, "A": [c for c in cells if c[0] == "A"], "B": [c for c in cells if c[0] == "B"],
              "C": [c for c in cells if c[0] == "C"]}

    def summ(values):
        v = [x for x in values if x is not None]
        return {"mean": st.mean(v), "median": st.median(v), "min": min(v), "max": max(v), "n": len(v)} if v else None

    summary = {}
    for g, cs in groups.items():
        summary[g] = {
            "level": {c: unflat({k: summ([flat(out["cells"][x][c]["mean"])[k] for x in cs])["mean"]
                                 if summ([flat(out["cells"][x][c]["mean"])[k] for x in cs]) else None for k in keys})
                      for c in CONDS},
            "kept_vs_untrained": {c: unflat({k: summ([flat(out["cells"][x]["kept_vs_untrained"][c]).get(k) for x in cs])
                                             for k in keys}) for c in ("rm_pre", "rm_decisive", "ampV")},
            "recovery_vs_lambda1": {"graft2": unflat({k: summ([flat(out["cells"][x]["recovery_vs_lambda1"]["graft2"]).get(k)
                                                               for x in cs]) for k in keys})}}
    out["summary"] = summary
    out["definitions"] = {"texts": "first sample of each of the first 200 MATH-500 problems, seeds 0-2",
                          "head_words": HEAD, "min_recap_words": MIN_RECAP_WORDS,
                          "recap_measure": "b_boxed_recap: >= 2 \\boxed and >= 30 words after the end of the first \\boxed",
                          "opens_monologue_words": sorted(OPEN_MONO), "denominator_floor": FLOOR,
                          "conditions": {"lambda0": "unedited lambda0 adapter", "lambda1": "unedited lambda1 adapter",
                                         "untrained": "untrained instruct student", **EDITS}}
    (RESULTS / "student_structure.json").write_text(json.dumps(out, indent=2))
    s = summary["all"]
    for k in ("head.c_markers_per_1k.hesitation", "head.c_markers_per_1k.verification", "head.c_markers_per_1k.alternative",
              "head.c_attempts_per_solution", "head.d_any_verification", "whole.d_any_verification",
              "head.e_opens_monologue", "head.e_monologue_sentence_share", "whole.b_boxed_recap",
              "whole.b_boxed_recap_completed", "whole.b_words_after_first_boxed"):
        lv = {c: flat(s["level"][c])[k] for c in CONDS}
        kp = flat(s["kept_vs_untrained"]["rm_pre"]).get(k + ".median")
        kd = flat(s["kept_vs_untrained"]["rm_decisive"]).get(k + ".median")
        rg = flat(s["recovery_vs_lambda1"]["graft2"]).get(k + ".median")
        print(f"{k:40s} " + " ".join(f"{c}={v:.3f}" if v is not None else f"{c}=None" for c, v in lv.items())
              + f" | kept rm_pre {kp} rm_dec {kd} | graft2 rec {rg}")


if __name__ == "__main__":
    main()
