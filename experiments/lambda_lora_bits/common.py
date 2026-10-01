"""Shared helpers for the lambda-sweep LoRA-bits experiment."""

import json
import math
import os
import select
import signal
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
MI_EVALUATION = ROOT / "code" / "MI-Distillation" / "evaluation"

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# 151643 ends R1-style turns and 151645 (<|im_end|>) ends ChatML turns.
STOP_TOKEN_IDS = [151643, 151645]

# Template artefacts that can leak into decoded teacher text. Ids 151648/151649 are
# <think>/</think> in the R1 tokenizer but <|box_start|>/<|box_end|> in Qwen's, so
# both spellings are removed to keep every lambda's CoT in the same plain-text form.
_ARTEFACTS = [
    "<think>", "</think>", "<|box_start|>", "<|box_end|>",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|object_ref_start|>", "<|object_ref_end|>",
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>",
    "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>", "<|eom_id|>", "<|start_header_id|>", "<|end_header_id|>",
]


def clean_response(text: str) -> str:
    for token in _ARTEFACTS:
        text = text.replace(token, "")
    return text.strip()


def read_jsonl(path) -> List[Dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokenizer(path: str):
    """Load a tokenizer, giving it a pad token when the family ships without one (e.g. Llama)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prompt(tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": f"{problem.strip()}\n{INSTRUCTION}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode(tokenizer, problem: str, response: str, max_len: int) -> Tuple[List[int], List[int]]:
    """Prompt + response + EOS; only response and EOS tokens carry labels."""
    prompt_ids = tokenizer(build_prompt(tokenizer, problem), add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + response_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_len]
    return input_ids, labels


# --- grading (reuses the MI-Distillation grader) ------------------------------

_grader = None


def _load_grader():
    global _grader
    if _grader is None:
        if str(MI_EVALUATION) not in sys.path:
            sys.path.insert(0, str(MI_EVALUATION))
        from data_processing.answer_extraction import extract_math_answer, strip_string
        from eval.eval_script import eval_math

        _grader = (extract_math_answer, strip_string, eval_math)
    return _grader


class _GraderTimeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _GraderTimeout()


def is_correct(problem: str, response: str, answer: str, timeout: int = 5) -> bool:
    """Grade in a forked child that is killed after the timeout.

    SIGALRM only fires between Python bytecodes, so a response that sends a regex or sympy into a long
    C-level loop hung the whole evaluation (2026-10-01, an OLMo merge run stuck for 40 minutes). The child
    runs the same grading with the same alarm; a child that does not answer in time counts as wrong,
    which is what the alarm was meant to do, so no result that previously finished changes.
    """
    _load_grader()                       # import in the parent, so the child never takes the import lock
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            ok = _grade(problem, response, answer, timeout)
        except BaseException:
            ok = False
        os.write(w, b"1" if ok else b"0")
        os._exit(0)
    os.close(w)
    try:
        ready, _, _ = select.select([r], [], [], timeout + 5)
        if ready:
            return os.read(r, 1) == b"1"
        os.kill(pid, signal.SIGKILL)
        print(f"[is_correct] grading killed after {timeout + 5}s", file=sys.stderr, flush=True)
        return False
    finally:
        os.close(r)
        os.waitpid(pid, 0)


def _grade(problem: str, response: str, answer: str, timeout: int = 5) -> bool:
    extract_math_answer, strip_string, eval_math = _load_grader()
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout)
    try:
        prediction = extract_math_answer(problem, response, task="math")
        item = {"prediction": prediction, "answer": [strip_string(answer)]}
        return bool(eval_math(item, pred_key="prediction"))
    except Exception:  # sympy failures and timeouts both count as wrong
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# --- batching and likelihoods -----------------------------------------------

def token_batches(lengths: Sequence[int], token_budget: int, quad_budget: int = 8192 ** 2) -> List[List[int]]:
    """Group indices (shortest first) so each batch fits both a padded-token and an attention budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, current, width = [], [], 0
    for i in order:
        new_width = max(width, lengths[i])
        size = len(current) + 1
        if current and (new_width * size > token_budget or new_width * new_width * size > quad_budget):
            batches.append(current)
            current, new_width = [], lengths[i]
        current.append(i)
        width = new_width
    if current:
        batches.append(current)
    return batches


def collate(encoded: Sequence[Tuple[List[int], List[int]]], indices: Sequence[int], pad_id: int, device):
    import torch

    width = max(len(encoded[i][0]) for i in indices)
    ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    labels = torch.full((len(indices), width), -100, dtype=torch.long)
    mask = torch.zeros((len(indices), width), dtype=torch.long)
    for row, i in enumerate(indices):
        x, y = encoded[i]
        ids[row, : len(x)] = torch.tensor(x)
        labels[row, : len(y)] = torch.tensor(y)
        mask[row, : len(x)] = 1
    # A batch without padding can use the fused causal kernel.
    attention_mask = None if bool(mask.all()) else mask.to(device)
    return ids.to(device), labels.to(device), attention_mask


def backbone_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head


def label_count(labels: Sequence[int]) -> int:
    return sum(1 for y in labels[1:] if y != -100)


def sample_bits(
    model, encoded, pad_id: int, token_budget: int = 8192, chunk: int = 1024, quad_budget: int = 2 ** 24
) -> List[float]:
    """Codelength in bits of each sample's labelled tokens: -sum log2 p(y_t | y_<t).

    Budgets are tighter than training's: batches here are length-sorted and dense, and the
    first sweep run hit repeated allocator OOM retries at 16k tokens / 2048-token logit chunks.
    """
    import torch
    import torch.nn.functional as F

    backbone, head = backbone_and_head(model)
    device = head.weight.device
    weight = head.weight.float()
    bits = [0.0] * len(encoded)
    with torch.inference_mode():
        for indices in token_batches([len(e[0]) for e in encoded], token_budget, quad_budget):
            ids, labels, mask = collate(encoded, indices, pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            for row, i in enumerate(indices):
                targets = labels[row, 1:]
                keep = targets != -100
                states, targets = hidden[row, :-1][keep], targets[keep]
                nats = 0.0
                for start in range(0, targets.numel(), chunk):
                    logits = states[start:start + chunk].float() @ weight.t()
                    nats += F.cross_entropy(logits, targets[start:start + chunk], reduction="sum").item()
                bits[i] = nats / math.log(2)
    return bits


def _chunk_nll(states, targets, weight):
    import torch.nn.functional as F

    return F.cross_entropy(states.float() @ weight.float().t(), targets, reduction="sum")


def batch_nll_sum(model, ids, labels, mask, chunk: int = 2048):
    """Summed next-token NLL (nats) over labelled tokens, with chunked, checkpointed logits."""
    import torch
    from torch.utils.checkpoint import checkpoint

    backbone, head = backbone_and_head(model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
    targets = labels[:, 1:]
    keep = targets != -100
    states, targets = hidden[:, :-1][keep], targets[keep]
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk):
        total = total + checkpoint(
            _chunk_nll, states[start:start + chunk], targets[start:start + chunk], head.weight, use_reentrant=False
        )
    return total
