#!/usr/bin/env python3
import json
import re
import statistics as st
from analyze_main_density import REFLECTIVE
from config import EXP as E, RESULTS

PAIRS = {
    "A": ("pairA", ["0.0", "0.2", "0.4"]),
    "B": ("pairB", ["0.0", "0.2", "0.4", "0.6"]),
    "C": ("pairC", ["0.0", "0.2", "0.4", "0.6"]),
}
MIN_RECAP_WORDS = 30
BIN = {
    "wait": "hesitation",
    "hmm": "hesitation",
    "actually": "hesitation",
    "hold on": "hesitation",
    "on second thought": "hesitation",
    "let me check": "verification",
    "recheck": "verification",
    "verify": "verification",
    "double-check": "verification",
    "double check": "verification",
    "is that right": "verification",
    "alternatively": "alternative",
    "reconsider": "alternative",
    "let me re": "alternative",
    "but let me": "other",
}
ALT_APPROACH = re.compile(
    "\\b(alternatively|another (?:way|approach|method)|a different (?:way|approach|method)|let me try (?:another|a different|again)|start over)\\b",
    re.I,
)
VERIFY = re.compile(
    "\\b(let me (?:just )?(?:check|verify|confirm|double[- ]check|make sure)|double[- ]check(?:ing)?|verif(?:y|ying|ied)|recheck|check(?:ing)? (?:my|the|this|our) (?:work|answer|result|solution|calculation)s?|sanity check|is that right|plug(?:ging)? (?:\\S+ ){0,3}back|substitut(?:e|ing) (?:\\S+ ){0,3}back)\\b",
    re.I,
)
SENT_SPLIT = re.compile("(?<=[.!?])\\s+|\\n+")
MONO = re.compile("^(?:I|I'm|I'll|I've|I'd|Let me|So|Okay|Hmm+)\\b")
OPEN_MONO = {"Okay", "Alright", "Hmm", "So"}
BOXED = "\\boxed"
SKIP = ("solutions", "joined", "b_proxy_vs_tag_agreement")


def first_boxed_end(text):
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
    return [s.strip() for s in SENT_SPLIT.split(text) if re.search("[A-Za-z]", s)]


def train_measures(rows):
    words = [len(r["response"].split()) for r in rows]
    n, total_words = (len(rows), sum(words))
    bins = {b: 0 for b in ("hesitation", "verification", "alternative", "other")}
    markers = 0
    recap = after_boxed = 0
    alt_counts, verify_any, mono_sent, all_sent, opens = ([], 0, 0, 0, 0)
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
        mono_sent += sum((bool(MONO.match(s)) for s in ss))
        first = t.lstrip().split(maxsplit=1)
        opens += bool(first) and first[0].rstrip(",.!") in OPEN_MONO
    return {
        "solutions": n,
        "a_mean_words": total_words / n,
        "a_mean_teacher_tokens": st.mean((r["teacher_tokens"] for r in rows)),
        "a_markers_per_1k": 1000 * markers / total_words,
        "b_boxed_recap": recap / n,
        "b_words_after_first_boxed": after_boxed / total_words,
        "c_markers_per_solution": {b: v / n for b, v in bins.items()},
        "c_markers_per_1k": {b: 1000 * v / total_words for b, v in bins.items()},
        "c_attempts_per_solution": 1 + st.mean(alt_counts),
        "c_any_alternative_approach": sum((c > 0 for c in alt_counts)) / n,
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
            tail = len(t[k + len("</think>") :].split())
            after += tail
            has_recap = tail >= MIN_RECAP_WORDS
            recap += has_recap
        total += len(t.split())
        end = first_boxed_end(tr["response"])
        proxy = (
            end >= 0
            and len(tr["response"][end:].split()) >= MIN_RECAP_WORDS
            and (BOXED in tr["response"][end:])
        )
        agree += proxy == has_recap
    n = len(have)
    return {
        "joined": n,
        "b_think_tag": tagged / n,
        "b_tagged_recap": recap / n,
        "b_words_after_tag": after / total,
        "b_proxy_vs_tag_agreement": agree / n,
    }


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and (not isinstance(v, bool)) and (k not in SKIP):
            out[prefix + k] = v
    return out


def main():
    td = json.loads((RESULTS / "teacher_density.json").read_text())
    lc = json.loads((RESULTS / "length_confound.json").read_text())
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
            m["a_markers_per_1k_all_generations"] = td[pair]["teacher"][lam]
            m.update(raw_measures(raw, rows))
            per[lam] = m
            print(
                f"{pair} {lam}: words {m['a_mean_words']:6.0f} mk {m['a_markers_per_1k']:5.1f} tag {m['b_think_tag']:.2f} recap(tag) {m['b_tagged_recap']:.2f} recap(boxed) {m['b_boxed_recap']:.2f} after_boxed {m['b_words_after_first_boxed']:.2f} verify {m['d_any_verification']:.2f} attempts {m['c_attempts_per_solution']:.2f} mono {m['e_monologue_sentence_share']:.2f} open {m['e_opens_monologue']:.2f}",
                flush=True,
            )
        classes = {r["interval"]: r["class"] for r in lc["pairs"][pair]["intervals"]}
        steps = {}
        for a, b in zip(lams, lams[1:]):
            fa, fb = (flat(per[a]), flat(per[b]))
            ratio = {k: fb[k] / fa[k] if fa[k] else None for k in fa}
            status = {
                k: None if v is None else "held" if 0.8 <= v <= 1.25 else "falls" if v < 0.8 else "rises"
                for k, v in ratio.items()
            }
            steps[f"{a}->{b}"] = {
                "class": classes.get(f"{a}->{b}"),
                "ratio": ratio,
                "status": status,
                "held": sorted((k for k, s in status.items() if s == "held")),
                "falls": sorted((k for k, s in status.items() if s == "falls")),
                "rises": sorted((k for k, s in status.items() if s == "rises")),
            }
        out[pair] = {"per_lambda": per, "steps": steps}
        for s, v in steps.items():
            print(
                f"  {pair} {s} ({v['class']}): held {v['held']}\n     falls {v['falls']}\n     rises {v['rises']}"
            )
    out["definitions"] = {
        "min_recap_words": MIN_RECAP_WORDS,
        "held_band": [0.8, 1.25],
        "pairs": {p: {"dir": t, "lambdas": l} for p, (t, l) in PAIRS.items()},
    }
    (RESULTS / "teacher_structure.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
