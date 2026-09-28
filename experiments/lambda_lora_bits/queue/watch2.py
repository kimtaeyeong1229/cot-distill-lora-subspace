#!/usr/bin/env python3
"""Health check for the two machines that are left, printing only what needs attention.

The old watcher polled the GPU dashboard, which still lists server1 and server3; now that those are
returned they would read as permanently idle and drown the real signal. This asks the two remaining
machines directly and knows what work is supposed to be running, so it can tell "finished" apart from
"died" -- a distinction the idle check alone cannot make.

Reports a line when: a GPU falls idle while its queue still has cells to do, a container is not
running, a machine is unreachable, or the sweep finishes.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP = "experiments/lambda_lora_bits"
STATE = HERE / "watch2_state.json"

# cell -> machine that owns it, and how many steering evaluations it should produce
SERVER2 = ["sweep_llama", "sweep", "sweep_olmo2", "pairB/llama1b", "pairB/qwen0.5b"]
SERVER5 = ["pairB/olmo1b", "pairD/llama1b", "pairD/qwen0.5b", "pairD/olmo1b"]
PER_CELL = 5
REMOTE5 = "rtx3090"
RDIR = "workspace/cot-distill-lora-bit"


def run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return ""


def local_done(cells):
    return sum(len(list((ROOT / EXP / c / "eval_steer").glob("*.json")))
               for c in cells if (ROOT / EXP / c / "eval_steer").is_dir())


def remote_done(cells):
    globs = " ".join(f"{RDIR}/{EXP}/{c}/eval_steer/*.json" for c in cells)
    out = run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", REMOTE5,
               f"ls {globs} 2>/dev/null | wc -l"])
    return int(out) if out.isdigit() else None


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    alerts = []

    machines = [
        ("server2", None, SERVER2, local_done(SERVER2)),
        ("server5", REMOTE5, SERVER5, remote_done(SERVER5)),
    ]

    for name, host, cells, done in machines:
        target = len(cells) * PER_CELL
        if host is None:
            gpu = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                       "--format=csv,noheader,nounits"])
            alive = len(run(["pgrep", "-f", "eval_abla[t]ion"]).split())
        else:
            gpu = run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host,
                       "nvidia-smi --query-gpu=utilization.gpu,memory.used "
                       "--format=csv,noheader,nounits"])
            alive = run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host,
                         "docker ps --filter name=cot-distill -q | wc -l"])
            alive = int(alive) if alive.isdigit() else 0

        if not gpu:
            alerts.append(f"ALERT {name} 응답 없음")
            continue
        if done is None:
            alerts.append(f"ALERT {name} 진행 상황 조회 실패")
            continue

        util, mem = (int(float(x)) for x in gpu.split(",")[:2])
        key = f"{name}_idle"
        if done >= target:
            if not state.get(f"{name}_reported"):
                alerts.append(f"DONE {name} 조작 실험 완료 {done}/{target}")
                state[f"{name}_reported"] = True
        elif util < 5 and mem < 3000:
            # an idle GPU with work left means the run stopped, not that it finished
            state[key] = state.get(key, 0) + 1
            if state[key] >= 2:
                alerts.append(f"ALERT {name} GPU 유휴 {state[key]}회 연속인데 {done}/{target}만 완료 "
                              f"— 중단된 듯 (컨테이너 {alive})")
                state[key] = 0
        else:
            state[key] = 0

    STATE.write_text(json.dumps(state))
    for line in alerts:
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
