#!/usr/bin/env python3
"""Generator codelength of teacher rationales under the sampler that produced them.

bits(y|x) = -log2 of the sampled text's probability under temperature T and top-p, renormalised per step,
with the prompt the generator used. Generation stops on any id in generation_config.eos_token_id, so the
end of a rationale is the event "some stop id", scored as -log2 of their summed (kept, renormalised)
probability. Replaces the teacher mode of pilot_validity.py, which appended a single eos id and
over-charged teachers that stop on <|im_end|>.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import backbone_and_head, build_prompt, read_jsonl  # noqa: E402

LN2 = math.log(2)


def step_logprobs(logits, T, top_p):
    """Renormalised log-probs after temperature and top-p (vLLM order); -inf outside the nucleus."""
    lp = F.log_softmax(logits.float() / T, -1)
    probs = lp.exp()
    sp, si = probs.sort(-1, descending=True)
    keep_sorted = (sp.cumsum(-1) - sp) < top_p
    keep = torch.zeros_like(keep_sorted).scatter(-1, si, keep_sorted)
    mass = (probs * keep).sum(-1, keepdim=True)
    return torch.where(keep, lp - mass.log(), torch.full_like(lp, float("-inf"))), lp - lp.logsumexp(-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--cot", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", default="held,train")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.teacher)
    stop = json.loads((Path(a.teacher) / "generation_config.json").read_text()).get("eos_token_id")
    stop = stop if isinstance(stop, list) else [stop]
    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=dtype).to(a.device).eval()
    raw = {r["unique_id"]: r for r in read_jsonl(a.cot)}
    rows = {"held": read_jsonl(Path(a.splits) / "held.jsonl")}
    train = read_jsonl(Path(a.splits) / "train.jsonl")
    rows["train"] = random.Random(a.seed).sample(train, min(a.n_train, len(train)))
    out = {"teacher": a.teacher, "stop_ids": stop, "temperature": a.temperature, "top_p": a.top_p, "sets": {}}
    done = json.loads(Path(a.output).read_text())["sets"] if Path(a.output).exists() else {}
    for name in a.sets.split(","):
        if name in done:
            out["sets"][name] = done[name]
            continue
        samples = []
        for k, r in enumerate(rows[name]):
            src = raw[r["unique_id"]]
            p = tok(build_prompt(tok, src["problem"]), add_special_tokens=False)["input_ids"]
            y = tok(src["cot_response"], add_special_tokens=False)["input_ids"]
            ids = torch.tensor([p + y], device=a.device)
            backbone, head = backbone_and_head(model)
            with torch.inference_mode():
                hidden = backbone(input_ids=ids).last_hidden_state[0, len(p) - 1:]  # predicts y[0..] then the stop event
            weight = head.weight.float()
            body, raw_t, misses = 0.0, 0.0, 0
            for s in range(0, len(y), 256):
                chunk = hidden[s:min(s + 256, len(y))].float() @ weight.t()
                lp_keep, lp_temp = step_logprobs(chunk, a.temperature, a.top_p)
                tgt = torch.tensor(y[s:s + 256], device=a.device)[:, None]
                kept = lp_keep.gather(1, tgt).squeeze(1)
                temp = lp_temp.gather(1, tgt).squeeze(1)
                misses += int(torch.isinf(kept).sum())
                body += -torch.where(torch.isinf(kept), temp, kept).sum().item() / LN2
                raw_t += -temp.sum().item() / LN2
            lp_keep, lp_temp = step_logprobs((hidden[len(y):len(y) + 1].float() @ weight.t()), a.temperature, a.top_p)
            stop_lp = lp_keep[0, stop].logsumexp(0)
            end = -(stop_lp if torch.isfinite(stop_lp) else lp_temp[0, stop].logsumexp(0)).item() / LN2
            samples.append({"unique_id": r["unique_id"], "body_bits": body, "end_bits": end, "floor_bits": body + end,
                            "tempered_bits": raw_t + end, "nucleus_misses": misses, "tokens": len(y) + 1})
            if k % 50 == 0:
                print(name, k, round(body, 1), round(end, 2), misses, flush=True)
        out["sets"][name] = samples
        Path(a.output).write_text(json.dumps(out))
        print(name, "mean floor", sum(s["floor_bits"] for s in samples) / len(samples), flush=True)
    Path(a.output).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
