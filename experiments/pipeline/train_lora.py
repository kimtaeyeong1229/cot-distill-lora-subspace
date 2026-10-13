#!/usr/bin/env python3
import argparse
import json
import math
import random
import time
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM
from common import batch_nll_sum, collate, encode, label_count, load_tokenizer, read_jsonl, token_batches

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--token_budget", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    if (output / "adapter" / "adapter_model.safetensors").exists():
        print(f"{output} already trained; skipping")
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    random.seed(args.seed)
    tokenizer = load_tokenizer(args.base)
    rows = read_jsonl(args.train)
    encoded = [encode(tokenizer, row["problem"], row["response"], args.max_len) for row in rows]
    lengths = [len(ids) for ids, _ in encoded]
    counts = [label_count(labels) for _, labels in encoded]
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.cuda()
    torch.manual_seed(args.seed)
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.save_pretrained(output / "adapter_init")
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(rows) / args.batch_size) * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = LambdaLR(optimizer, schedule)
    order_rng = random.Random(args.seed)
    step, tokens_seen, began = (0, 0, time.time())
    with (output / "train_log.jsonl").open("w") as log:
        for epoch in range(args.epochs):
            order = list(range(len(rows)))
            order_rng.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                batch = order[start : start + args.batch_size]
                n_labels = sum((counts[i] for i in batch))
                nll_total = 0.0
                for micro in token_batches([lengths[i] for i in batch], args.token_budget):
                    indices = [batch[j] for j in micro]
                    ids, labels, mask = collate(encoded, indices, tokenizer.pad_token_id, "cuda")
                    nll = batch_nll_sum(model, ids, labels, mask)
                    (nll / n_labels).backward()
                    nll_total += nll.item()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                tokens_seen += n_labels
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": nll_total / n_labels,
                    "grad_norm": grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens": tokens_seen,
                    "elapsed_min": (time.time() - began) / 60,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step % 10 == 0 or step == total_steps:
                    print(json.dumps(record))
    model.save_pretrained(output / "adapter")
    summary = {
        "steps": step,
        "train_samples": len(rows),
        "train_label_tokens_per_epoch": sum(counts),
        "trainable_params": sum((p.numel() for p in params)),
        "minutes": (time.time() - began) / 60,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
