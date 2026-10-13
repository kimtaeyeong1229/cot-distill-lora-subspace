import json
import os
import select
import signal
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
MI_EVALUATION = ROOT / "code" / "MI-Distillation" / "evaluation"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
STOP_TOKEN_IDS = [151643, 151645]
_ARTEFACTS = [
    "<think>",
    "</think>",
    "<|box_start|>",
    "<|box_end|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
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
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prompt(tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": f"{problem.strip()}\n{INSTRUCTION}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode(tokenizer, problem: str, response: str, max_len: int) -> Tuple[List[int], List[int]]:
    prompt_ids = tokenizer(build_prompt(tokenizer, problem), add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + response_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_len]
    return (input_ids, labels)


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
    _load_grader()
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
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def token_batches(lengths: Sequence[int], token_budget: int, quad_budget: int = 8192**2) -> List[List[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, current, width = ([], [], 0)
    for i in order:
        new_width = max(width, lengths[i])
        size = len(current) + 1
        if current and (new_width * size > token_budget or new_width * new_width * size > quad_budget):
            batches.append(current)
            current, new_width = ([], lengths[i])
        current.append(i)
        width = new_width
    if current:
        batches.append(current)
    return batches


def collate(encoded: Sequence[Tuple[List[int], List[int]]], indices: Sequence[int], pad_id: int, device):
    import torch

    width = max((len(encoded[i][0]) for i in indices))
    ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    labels = torch.full((len(indices), width), -100, dtype=torch.long)
    mask = torch.zeros((len(indices), width), dtype=torch.long)
    for row, i in enumerate(indices):
        x, y = encoded[i]
        ids[row, : len(x)] = torch.tensor(x)
        labels[row, : len(y)] = torch.tensor(y)
        mask[row, : len(x)] = 1
    attention_mask = None if bool(mask.all()) else mask.to(device)
    return (ids.to(device), labels.to(device), attention_mask)


def backbone_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return (base.model, base.lm_head)


def label_count(labels: Sequence[int]) -> int:
    return sum((1 for y in labels[1:] if y != -100))


def _chunk_nll(states, targets, weight):
    import torch.nn.functional as F

    return F.cross_entropy(states.float() @ weight.float().t(), targets, reduction="sum")


def batch_nll_sum(model, ids, labels, mask, chunk: int = 2048):
    import torch
    from torch.utils.checkpoint import checkpoint

    backbone, head = backbone_and_head(model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
    targets = labels[:, 1:]
    keep = targets != -100
    states, targets = (hidden[:, :-1][keep], targets[keep])
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk):
        total = total + checkpoint(
            _chunk_nll,
            states[start : start + chunk],
            targets[start : start + chunk],
            head.weight,
            use_reentrant=False,
        )
    return total
