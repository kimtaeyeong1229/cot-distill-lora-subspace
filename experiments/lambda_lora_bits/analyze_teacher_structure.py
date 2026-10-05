#!/usr/bin/env python3
"""What does pair A's tag-held interval (lambda 0 -> 0.2) keep besides the think tag? Teacher texts only (read only).

Texts. "train": the teacher solutions the students were trained on (<pair>/splits/lambda<l>/train.jsonl, field
response = cot_response with the template tags removed by common.clean_response). Every lambda of a pair uses
the same problems (A 2,113, B 2,261, C 1,015), so differences across lambda are not problem mix. "raw": the
teacher's own generations (<pair>/cot/lambda<l>.jsonl, field cot_response, the files analyze_teacher_density.py
and measure_think_rate.py read), which still contain the closing </think>; they are joined to the training
problems by unique_id. Pair C is stored as pairD. Lambdas: A 0, 0.2, 0.4; B and C 0, 0.2, 0.4, 0.6.

Measures, per lambda (words = whitespace tokens; "per 1k words" pools markers and words over all solutions):
 (a) length: mean words and mean teacher tokens per training solution; reflective-marker density per 1k words
     with the shared REFLECTIVE pattern (analyze_main_density.py), on the training texts and, reused, the
     all-4,000-generation value from teacher_density.json.
 (b) closing answer section ("draft, then recap"):
     raw    tagged_recap = share of solutions that contain </think> followed by at least 30 words;
            words_after_tag = share of all words that come after </think> (pooled).
     train  boxed_recap = share of solutions with at least two \\boxed and at least 30 words after the end of the
            first \\boxed (an answer is reached, then the solution goes on and states it again);
            words_after_first_boxed = share of all words after the end of the first \\boxed (pooled).
            Its agreement with tagged_recap on the same problems is reported (proxy_vs_tag_agreement).
 (c) marker categories: every REFLECTIVE match is put in one bin --
            hesitation   wait, hmm, actually, hold on, on second thought
            verification let me check, recheck, verify, double-check / double check, is that right
            alternative  alternatively, reconsider, let me re(...)
            other        but let me
     reported per solution and per 1k words. Attempts proxy: 1 + the number of ALT_APPROACH phrases
     (alternatively, another way/approach/method, a different way/approach/method, let me try another/a
     different/again, start over), mean per solution, and the share of solutions with at least one.
 (d) verification step: share of solutions with at least one VERIFY phrase (let me check/verify/confirm/
     double-check/make sure, double-check(ing), verify(ing), recheck, check my/the work/answer/result,
     sanity check, is that right, plug(ging) ... back, substitut(e/ing) back).
 (e) first-person monologue: sentences are the pieces between [.!?] + whitespace or newlines that contain a
     letter; monologue_sentence_share = share whose first word is I (I'm, I'll, I've, I'd), "Let me", So,
     Okay or Hmm (pooled over sentences); opens_monologue = share of solutions whose first word is Okay,
     Alright, Hmm or So.

Interval reading: for each consecutive lambda step the ratio new / old of each measure; a measure is called
"held" when the ratio lies in [0.8, 1.25] and "falls" when it is below 0.8. Interval classes (held / switch)
are those of length_confound.json.

Run from the repo root; writes experiments/teacher_structure.json.
"""

import json
import re
import statistics as st
import sys
from pathlib import Path

E = Path("experiments/lambda_lora_bits")
sys.path.insert(0, str(E))
from analyze_main_density import REFLECTIVE  # noqa: E402

PAIRS = {"A": ("sweep", ["0.0", "0.2", "0.4"]), "B": ("pairB", ["0.0", "0.2", "0.4", "0.6"]),
         "C": ("pairD", ["0.0", "0.2", "0.4", "0.6"])}
MIN_RECAP_WORDS = 30
BIN = {"wait": "hesitation", "hmm": "hesitation", "actually": "hesitation", "hold on": "hesitation",
       "on second thought": "hesitation", "let me check": "verification", "recheck": "verification",
       "verify": "verification", "double-check": "verification", "double check": "verification",
       "is that right": "verification", "alternatively": "alternative", "reconsider": "alternative",
       "let me re": "alternative", "but let me": "other"}
ALT_APPROACH = re.compile(
    r"\b(alternatively|another (?:way|approach|method)|a different (?:way|approach|method)|"
    r"let me try (?:another|a different|again)|start over)\b", re.I)
VERIFY = re.compile(
    r"\b(let me (?:just )?(?:check|verify|confirm|double[- ]check|make sure)|double[- ]check(?:ing)?|"
    r"verif(?:y|ying|ied)|recheck|check(?:ing)? (?:my|the|this|our) (?:work|answer|result|solution|calculation)s?|"
    r"sanity check|is that right|plug(?:ging)? (?:\S+ ){0,3}back|substitut(?:e|ing) (?:\S+ ){0,3}back)\b", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MONO = re.compile(r"^(?:I|I'm|I'll|I've|I'd|Let me|So|Okay|Hmm+)\b")
OPEN_MONO = {"Okay", "Alright", "Hmm", "So"}
BOXED = "\\boxed"
SKIP = ("solutions", "joined", "b_proxy_vs_tag_agreement")       # counts and the proxy check are not measures


def first_boxed_end(text):
    """Index just past the closing brace of the first \\boxed{...}, or -1."""
    i = text.find(BOXED)
    if i < 0:
        return -1
    j = text.find("{", i)
    if j < 0:
        return -1
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    return -1


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if re.search(r"[A-Za-z]", s)]


def train_measures(rows):
    words = [len(r["response"].split()) for r in rows]
    n, total_words = len(rows), sum(words)
    bins = {b: 0 for b in ("hesitation", "verification", "alternative", "other")}
    markers = 0
    recap = after_boxed = 0
    alt_counts, verify_any, mono_sent, all_sent, opens = [], 0, 0, 0, 0
    for r in rows:
        t = r["response"]
        for m in REFLECTIVE.findall(t):
            bins[BIN[m.lower().replace("\n", " ")]] += 1
            markers += 1
        end = first_boxed_end(t)
        if end >= 0:
            tail = len(t[end:].split())
            after_boxed += tail
            recap += tail >= MIN_RECAP_WORDS and BOXED in t[end:]
        alt_counts.append(len(ALT_APPROACH.findall(t)))
        verify_any += bool(VERIFY.search(t))
        ss = sentences(t)
        all_sent += len(ss)
        mono_sent += sum(bool(MONO.match(s)) for s in ss)
        first = t.lstrip().split(maxsplit=1)
        opens += bool(first) and first[0].rstrip(",.!") in OPEN_MONO
    return {
        "solutions": n,
        "a_mean_words": total_words / n, "a_mean_teacher_tokens": st.mean(r["teacher_tokens"] for r in rows),
        "a_markers_per_1k": 1000 * markers / total_words,
        "b_boxed_recap": recap / n, "b_words_after_first_boxed": after_boxed / total_words,
        "c_markers_per_solution": {b: v / n for b, v in bins.items()},
        "c_markers_per_1k": {b: 1000 * v / total_words for b, v in bins.items()},
        "c_attempts_per_solution": 1 + st.mean(alt_counts),
        "c_any_alternative_approach": sum(c > 0 for c in alt_counts) / n,
        "d_any_verification": verify_any / n,
        "e_monologue_sentence_share": mono_sent / all_sent,
        "e_opens_monologue": opens / n,
    }


def raw_measures(raw, train_rows):
    ids = [r["unique_id"] for r in train_rows]
    have = [i for i in ids if i in raw]
    tagged = recap = after = total = 0
    agree = 0
    for i, tr in zip(ids, train_rows):
        if i not in raw:
            continue
        t = raw[i]
        k = t.find("</think>")
        has_recap = False
        if k >= 0:
            tagged += 1
            tail = len(t[k + len("</think>"):].split())
            after += tail
            has_recap = tail >= MIN_RECAP_WORDS
            recap += has_recap
        total += len(t.split())
        end = first_boxed_end(tr["response"])
        proxy = end >= 0 and len(tr["response"][end:].split()) >= MIN_RECAP_WORDS and BOXED in tr["response"][end:]
        agree += proxy == has_recap
    n = len(have)
    return {"joined": n, "b_think_tag": tagged / n, "b_tagged_recap": recap / n, "b_words_after_tag": after / total,
            "b_proxy_vs_tag_agreement": agree / n}


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and k not in SKIP:
            out[prefix + k] = v
    return out


def main():
    td = json.loads(Path("experiments/teacher_density.json").read_text())
    lc = json.loads(Path("experiments/length_confound.json").read_text())
    out = {}
    for pair, (tag, lams) in PAIRS.items():
        per = {}
        for lam in lams:
            rows = [json.loads(line) for line in open(E / tag / "splits" / f"lambda{lam}" / "train.jsonl")]
            raw = {}
            with open(E / tag / "cot" / f"lambda{lam}.jsonl") as f:
                for line in f:
                    r = json.loads(line)
                    raw[r["unique_id"]] = r["cot_response"]
            m = train_measures(rows)
            m["a_markers_per_1k_all_generations"] = td[pair]["teacher"][lam]   # reused from teacher_density.json
            m.update(raw_measures(raw, rows))
            per[lam] = m
            print(f"{pair} {lam}: words {m['a_mean_words']:6.0f} mk {m['a_markers_per_1k']:5.1f} tag {m['b_think_tag']:.2f}"
                  f" recap(tag) {m['b_tagged_recap']:.2f} recap(boxed) {m['b_boxed_recap']:.2f}"
                  f" after_boxed {m['b_words_after_first_boxed']:.2f} verify {m['d_any_verification']:.2f}"
                  f" attempts {m['c_attempts_per_solution']:.2f} mono {m['e_monologue_sentence_share']:.2f}"
                  f" open {m['e_opens_monologue']:.2f}", flush=True)
        classes = {r["interval"]: r["class"] for r in lc["pairs"][pair]["intervals"]}
        steps = {}
        for a, b in zip(lams, lams[1:]):
            fa, fb = flat(per[a]), flat(per[b])
            ratio = {k: (fb[k] / fa[k] if fa[k] else None) for k in fa}
            status = {k: (None if v is None else "held" if 0.8 <= v <= 1.25 else "falls" if v < 0.8 else "rises")
                      for k, v in ratio.items()}
            steps[f"{a}->{b}"] = {"class": classes.get(f"{a}->{b}"), "ratio": ratio, "status": status,
                                  "held": sorted(k for k, s in status.items() if s == "held"),
                                  "falls": sorted(k for k, s in status.items() if s == "falls"),
                                  "rises": sorted(k for k, s in status.items() if s == "rises")}
        out[pair] = {"per_lambda": per, "steps": steps}
        for s, v in steps.items():
            print(f"  {pair} {s} ({v['class']}): held {v['held']}\n     falls {v['falls']}\n     rises {v['rises']}")
    out["definitions"] = {"min_recap_words": MIN_RECAP_WORDS, "held_band": [0.8, 1.25],
                          "pairs": {p: {"dir": t, "lambdas": l} for p, (t, l) in PAIRS.items()}}
    Path("experiments/teacher_structure.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
