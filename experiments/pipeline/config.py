import os
import re
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
EXP = REPO_ROOT / "experiments" / "pipeline"
RESULTS = Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "experiments"))
MODELS = REPO_ROOT / "models"
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
SEEDS = [0, 1, 2]
PAIRS = ["A", "B", "C"]
PAIR_DIR = {"A": "pairA", "B": "pairB", "C": "pairC"}
PAIR_KEY = {"A": "A", "B": "B", "C": "C"}
STUDENTS = ["Qwen", "Llama", "OLMo"]
STUDENT_MODEL = {
    "Qwen": "Qwen2.5-0.5B-Instruct",
    "Llama": "Llama-3.2-1B-Instruct",
    "OLMo": "OLMo-2-1B-Instruct",
}
STUDENT_SHORT = {"Qwen": "Qwen2.5-0.5B", "Llama": "Llama-3.2-1B", "OLMo": "OLMo-2-1B"}
CELL_DIR = {
    "A/Qwen": "pairA/qwen0.5b",
    "A/Llama": "pairA/llama1b",
    "A/OLMo": "pairA/olmo1b",
    "B/Qwen": "pairB/qwen0.5b",
    "B/Llama": "pairB/llama1b",
    "B/OLMo": "pairB/olmo1b",
    "C/Qwen": "pairC/qwen0.5b",
    "C/Llama": "pairC/llama1b",
    "C/OLMo": "pairC/olmo1b",
}
CELL_NAMES = list(CELL_DIR)
SWITCH = {"A": "0.4", "B": "0.6", "C": "0.6"}
MATH500_EVAL = {
    cell: "eval_math500_7680" if cell in ("A/Qwen", "A/Llama") else "eval_math500" for cell in CELL_NAMES
}
MAX_MODEL_LEN, MAX_NEW_TOKENS = (8192, 7680)
OLMO_MAX_MODEL_LEN, OLMO_MAX_NEW_TOKENS = (4096, 3840)


def generation_limits(student: str):
    return (OLMO_MAX_MODEL_LEN, OLMO_MAX_NEW_TOKENS) if "OLMo" in student else (MAX_MODEL_LEN, MAX_NEW_TOKENS)


EVAL_PROBLEMS, EVAL_SAMPLES = (200, 2)
HEAD_WORDS = 300
REFLECTIVE = re.compile(
    "\\b(wait|hmm|actually|but let me|let me check|recheck|verify|reconsider|alternatively|on second thought|let me re|double-check|double check|is that right|hold on)\\b",
    re.I,
)
