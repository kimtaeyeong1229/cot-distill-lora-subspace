"""Constants shared by the paper's training, evaluation, analysis and figure scripts.

Paths. Every script resolves data and results from REPO_ROOT (default: the repository that holds this
file), so it can be run from any working directory:

  REPO_ROOT    repository root; per-cell runs and evaluations live in REPO_ROOT/experiments/lambda_lora_bits
  RESULTS_DIR  where the result JSONs are read and written (default REPO_ROOT/experiments)

Naming. The paper's teacher pairs are A, B and C. Their directories are sweep*/, pairB/ and pairD/, and
some result files key pair C as "D" (the directory name). PAIR_DIR and PAIR_KEY translate.
A "cell" is one (pair, student), named like "C/Llama" in the result files.

This module imports only the standard library, so CPU-only analysis scripts can use it.
"""

import os
import re
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
EXP = REPO_ROOT / "experiments" / "lambda_lora_bits"           # per-cell runs/ and eval_*/ directories
RESULTS = Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "experiments"))   # experiments/*.json
MODELS = REPO_ROOT / "models"                                  # student and teacher checkpoints, SAE

# --- design -----------------------------------------------------------------------------------------
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]          # weight on the instruct endpoint
SEEDS = [0, 1, 2]                                              # adapter (training) seeds

PAIRS = ["A", "B", "C"]
PAIR_DIR = {"A": "sweep", "B": "pairB", "C": "pairD"}          # teacher CoT (cot/) and splits (splits/)
PAIR_KEY = {"A": "A", "B": "B", "C": "D"}                      # pair key in think_rate / phi_resample / consensus JSONs

STUDENTS = ["Qwen", "Llama", "OLMo"]
STUDENT_MODEL = {"Qwen": "Qwen2.5-0.5B-Instruct", "Llama": "Llama-3.2-1B-Instruct", "OLMo": "OLMo-2-1B-Instruct"}
STUDENT_SHORT = {"Qwen": "Qwen2.5-0.5B", "Llama": "Llama-3.2-1B", "OLMo": "OLMo-2-1B"}

CELL_DIR = {
    "A/Qwen": "sweep", "A/Llama": "sweep_llama", "A/OLMo": "sweep_olmo2",
    "B/Qwen": "pairB/qwen0.5b", "B/Llama": "pairB/llama1b", "B/OLMo": "pairB/olmo1b",
    "C/Qwen": "pairD/qwen0.5b", "C/Llama": "pairD/llama1b", "C/OLMo": "pairD/olmo1b",
}
CELL_NAMES = list(CELL_DIR)

# The switch lambda each pair's regions are built at: adapters below it are "pre-switch". Pair C drops in
# two steps of almost equal size, so its cells use the first one (0.6).
SWITCH = {"A": "0.4", "B": "0.6", "C": "0.6"}

# Directory of the unedited MATH-500 evaluation per cell. Pair A Qwen and Llama were re-run at the shared
# generation limit (drivers/run_math500.sh); every other cell's first run already used it.
MATH500_EVAL = {cell: "eval_math500_7680" if cell in ("A/Qwen", "A/Llama") else "eval_math500" for cell in CELL_NAMES}

# --- generation -------------------------------------------------------------------------------------
MAX_MODEL_LEN, MAX_NEW_TOKENS = 8192, 7680
OLMO_MAX_MODEL_LEN, OLMO_MAX_NEW_TOKENS = 4096, 3840          # OLMo-2-1B has a 4,096-token context


def generation_limits(student: str):
    """(max_model_len, max_new_tokens) for a student name or path."""
    return (OLMO_MAX_MODEL_LEN, OLMO_MAX_NEW_TOKENS) if "OLMo" in student else (MAX_MODEL_LEN, MAX_NEW_TOKENS)


# The shared student setting: every student number uses the first 200 MATH-500 problems x 2 samples.
EVAL_PROBLEMS, EVAL_SAMPLES = 200, 2
HEAD_WORDS = 300                                               # "early-response" markers: first 300 words

# --- reflective markers -----------------------------------------------------------------------------
# Hedging and self-correction markers, counted per 1,000 whitespace words. The training responses have the
# think tags stripped, so this density (not the tag) separates a long-CoT student from a short-answer one.
REFLECTIVE = re.compile(
    r"\b(wait|hmm|actually|but let me|let me check|recheck|verify|reconsider|alternatively|"
    r"on second thought|let me re|double-check|double check|is that right|hold on)\b", re.I)
