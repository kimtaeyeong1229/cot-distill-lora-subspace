#!/usr/bin/env python3
"""Validity pilot for the transfer/memorization split (overleaf/method_defensible.md).

Modes
  student  per-token codelength under the base student and its adapter on a train subset, the held-out
           set, a mismatched held-out set (problem i paired with the rationale of problem j), and an
           optional teacher resample of train problems. Tokens are split into math (digits and LaTeX
           math spans) and text, so the drop can be attributed.
  teacher  generator codelength of the same rationales: -log2 p under the teacher sampler actually used
           for generation (temperature, top-p renormalised), scored on the raw generated text with the
           generation prompt. Its set mean is an in-expectation floor no model can undercut without
           sample-specific information (Morris et al. 2026; Achille & Soatto 2018).
  canary   base/adapter codelength of training vs held-out canaries (true content known).
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from common import backbone_and_head, build_prompt, clean_response, collate, load_tokenizer, read_jsonl, token_batches

LN2 = math.log(2)
MATH_SPAN = re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\$[^$]*?\$|\\boxed\{[^}]*\}", re.S)


def math_mask(text):
    mask = [False] * len(text)
    for m in MATH_SPAN.finditer(text):
        for i in range(m.start(), m.end()):
            mask[i] = True
    for i, ch in enumerate(text):
        if ch.isdigit():
            mask[i] = True
    return mask


def encode_with_categories(tok, problem, response, max_len):
    prompt_ids = tok(build_prompt(tok, problem), add_special_tokens=False)["input_ids"]
    enc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
    mask = math_mask(response)
    cats = []
    for a, b in enc["offset_mapping"]:
        span = mask[a:b] or [False]
        cats.append(1 if sum(span) * 2 >= len(span) else 0)  # 1 math, 0 text
    resp_ids = enc["input_ids"] + [tok.eos_token_id]
    cats.append(2)  # eos
    ids = (prompt_ids + resp_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + resp_ids)[:max_len]
    return (ids, labels), cats[: max(0, len(ids) - len(prompt_ids))]


def token_bits(model, encoded, pad_id, token_budget=8192, chunk=1024):
    """Per-token bits (list per sample) of labelled tokens."""
    backbone, head = backbone_and_head(model)
    weight = head.weight.float()
    device = weight.device
    out = [None] * len(encoded)
    with torch.inference_mode():
        for indices in token_batches([len(e[0]) for e in encoded], token_budget, 2 ** 24):
            ids, labels, mask = collate(encoded, indices, pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            for row, i in enumerate(indices):
                targets = labels[row, 1:]
                keep = targets != -100
                states, targets = hidden[row, :-1][keep], targets[keep]
                parts = []
                for s in range(0, targets.numel(), chunk):
                    logits = states[s:s + chunk].float() @ weight.t()
                    parts.append(F.cross_entropy(logits, targets[s:s + chunk], reduction="none"))
                out[i] = (torch.cat(parts) / LN2).cpu().tolist()
    return out


def summarize_tokens(bits, cats):
    d = {"tokens": len(bits), "bits": sum(bits)}
    for name, c in (("math", 1), ("text", 0), ("eos", 2)):
        d[f"{name}_bits"] = sum(b for b, k in zip(bits, cats) if k == c)
        d[f"{name}_tokens"] = sum(1 for k in cats if k == c)
    return d


def pick_train(rows, n, seed):
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def mismatch(rows, tok):
    """Pair each held-out problem with a similar-length rationale of a different problem."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["response"]))
    partner = {}
    for a, b in zip(order[0::2], order[1::2]):
        partner[a], partner[b] = b, a
    if len(order) % 2:
        last, prev = order[-1], order[-2]
        partner[last] = prev
    return [{"unique_id": f"{r['unique_id']}|{rows[partner[i]]['unique_id']}", "problem": r["problem"],
             "response": rows[partner[i]]["response"]} for i, r in enumerate(rows)]


def run_student(a):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(a.student)
    model = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    sets = {
        "train": pick_train(read_jsonl(Path(a.splits) / "train.jsonl"), a.n_train, a.seed),
        "held": read_jsonl(Path(a.splits) / "held.jsonl"),
    }
    sets["mismatch"] = mismatch(sets["held"], tok)
    if a.resample and Path(a.resample).exists():
        sets["resample"] = read_jsonl(a.resample)
    result = {"student": a.student, "adapter": a.adapter, "splits": a.splits, "sets": {}}
    for name, rows in sets.items():
        enc, cats = zip(*[encode_with_categories(tok, r["problem"], r["response"], a.max_len) for r in rows])
        with model.disable_adapter():
            base = token_bits(model, list(enc), tok.pad_token_id)
        tuned = token_bits(model, list(enc), tok.pad_token_id)
        samples = []
        for r, b, t, c in zip(rows, base, tuned, cats):
            samples.append({"unique_id": r["unique_id"], "base": summarize_tokens(b, c), "tuned": summarize_tokens(t, c),
                            "chars": len(r["response"])})
        result["sets"][name] = samples
        print(name, len(samples), sum(s["base"]["bits"] for s in samples) / len(samples),
              sum(s["tuned"]["bits"] for s in samples) / len(samples), flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result))


def run_teacher(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.teacher)
    model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    raw = {r["unique_id"]: r for r in read_jsonl(a.cot)}
    wanted = {"train": pick_train(read_jsonl(Path(a.splits) / "train.jsonl"), a.n_train, a.seed),
              "held": read_jsonl(Path(a.splits) / "held.jsonl")}
    if a.resample_raw and Path(a.resample_raw).exists():
        res = {r["unique_id"]: r for r in read_jsonl(a.resample_raw)}
        keep = {r["unique_id"] for r in read_jsonl(a.resample)} if a.resample and Path(a.resample).exists() else set(res)
        wanted["resample"] = [res[u] for u in res if u in keep]
        raw.update({f"resample:{u}": res[u] for u in res})
    result = {"teacher": a.teacher, "temperature": a.temperature, "top_p": a.top_p, "sets": {}}
    backbone, head = backbone_and_head(model)
    weight = head.weight.float()
    # tokens that clean_response() deletes from the student text (e.g. </think>): their generator bits are
    # reported separately so the floor can be taken on the same string the student is scored on
    artefact_ids = {i for t in ("</think>", "<think>") for i in tok(t, add_special_tokens=False)["input_ids"]
                    if len(tok(t, add_special_tokens=False)["input_ids"]) == 1}
    for name, rows in wanted.items():
        encoded, meta = [], []
        for r in rows:
            src = raw[f"resample:{r['unique_id']}"] if name == "resample" else raw[r["unique_id"]]
            text = src["cot_response"]
            p_ids = tok(build_prompt(tok, src["problem"]), add_special_tokens=False)["input_ids"]
            y_ids = tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            encoded.append((p_ids + y_ids, [-100] * len(p_ids) + y_ids))
            meta.append((r["unique_id"], src.get("generated_tokens"), len(y_ids), len(clean_response(text))))
        samples = [None] * len(encoded)
        with torch.inference_mode():
            for indices in token_batches([len(e[0]) for e in encoded], 8192, 2 ** 24):
                ids, labels, mask = collate(encoded, indices, tok.pad_token_id or 0, weight.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
                for row, i in enumerate(indices):
                    targets = labels[row, 1:]
                    keep = targets != -100
                    states, targets = hidden[row, :-1][keep], targets[keep]
                    floor, plain, t1, misses, art = 0.0, 0.0, 0.0, 0, 0.0
                    is_art = torch.tensor([int(t) in artefact_ids for t in targets.tolist()], device=targets.device)
                    for s in range(0, targets.numel(), 512):
                        logits = states[s:s + 512].float() @ weight.t()
                        tgt = targets[s:s + 512]
                        t1 += -F.log_softmax(logits, -1).gather(1, tgt[:, None]).sum().item() / LN2
                        lp = F.log_softmax(logits / a.temperature, -1)
                        lp_t = lp.gather(1, tgt[:, None]).squeeze(1)
                        probs = lp.exp()
                        sp, _ = probs.sort(-1, descending=True)
                        cum = sp.cumsum(-1)
                        kept = (cum - sp) < a.top_p
                        kept_mass = (sp * kept).sum(-1)
                        threshold = torch.where(kept, sp, torch.full_like(sp, 2.0)).min(-1).values
                        inside = lp_t.exp() >= threshold * (1 - 1e-6)
                        misses += int((~inside).sum())
                        plain += -lp_t.sum().item() / LN2
                        tok_floor = -(torch.where(inside, lp_t - kept_mass.log(), lp_t)) / LN2
                        floor += tok_floor.sum().item()
                        art += tok_floor[is_art[s:s + 512]].sum().item()
                    uid, gen, ntok, nchar = meta[i]
                    samples[i] = {"unique_id": uid, "floor_bits": floor, "tempered_bits": plain, "t1_bits": t1,
                                  "nucleus_misses": misses, "tokens": ntok, "generated_tokens": gen,
                                  "artefact_bits": art}
        result["sets"][name] = samples
        print(name, len(samples), sum(s["floor_bits"] for s in samples) / len(samples),
              sum(s["nucleus_misses"] for s in samples), flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result))


def run_canary(a):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(a.student)
    model = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    result = {"adapter": a.adapter, "sets": {}}
    for name in ("canary_train", "canary_held"):
        rows = read_jsonl(Path(a.splits) / f"{name}.jsonl")
        enc, cats = zip(*[encode_with_categories(tok, r["problem"], r["response"], a.max_len) for r in rows])
        with model.disable_adapter():
            base = token_bits(model, list(enc), tok.pad_token_id)
        tuned = token_bits(model, list(enc), tok.pad_token_id)
        result["sets"][name] = [{"base": sum(b), "tuned": sum(t)} for b, t in zip(base, tuned)]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["student", "teacher", "canary"])
    p.add_argument("--student")
    p.add_argument("--adapter")
    p.add_argument("--teacher")
    p.add_argument("--cot", help="raw generation jsonl (cot_response) for teacher mode")
    p.add_argument("--splits", required=True)
    p.add_argument("--resample", help="cleaned, correct resample rows (student format)")
    p.add_argument("--resample_raw", help="raw resample generation jsonl")
    p.add_argument("--n_train", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_len", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    {"student": run_student, "teacher": run_teacher, "canary": run_canary}[a.mode](a)


if __name__ == "__main__":
    main()
