#!/usr/bin/env python3
import argparse
import io
import re
import urllib.request
from pathlib import Path
import pandas as pd

MATH_TRAIN_URL = (
    "https://huggingface.co/api/datasets/DigitalLearningGmbH/MATH-lighteval/parquet/default/train/0.parquet"
)
MATH500_URL = "https://huggingface.co/api/datasets/HuggingFaceH4/MATH-500/parquet/default/test/0.parquet"


def fetch_parquet(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url) as response:
        return pd.read_parquet(io.BytesIO(response.read()))


def last_boxed(text: str):
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : pos].strip()
    return None


def normalize(text: str) -> str:
    return re.sub("\\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="experiments/pipeline/data")
    parser.add_argument("--num_problems", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = fetch_parquet(MATH_TRAIN_URL)
    math500 = fetch_parquet(MATH500_URL)
    print(f"MATH train: {len(train)}  MATH-500: {len(math500)}")
    train["answer"] = train["solution"].map(last_boxed)
    no_answer = train["answer"].isna().sum()
    train = train[train["answer"].notna() & (train["answer"] != "")]
    test_problems = set(math500["problem"].map(normalize))
    overlap = train["problem"].map(normalize).isin(test_problems)
    train = train[~overlap]
    print(f"Dropped {no_answer} without a boxed answer, {int(overlap.sum())} overlapping MATH-500")
    train = train.reset_index(drop=True)
    train.insert(0, "unique_id", range(len(train)))
    pool = train.sample(n=min(args.num_problems, len(train)), random_state=args.seed).sort_values("unique_id")
    pool[["unique_id", "problem", "answer", "level", "type", "solution"]].to_parquet(out_dir / "pool.parquet")
    math500.to_parquet(out_dir / "math500.parquet")
    print(f"Pool: {len(pool)} problems -> {out_dir / 'pool.parquet'}")
    print(pool["level"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
