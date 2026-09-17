#!/usr/bin/env python3
"""One health check across the three machines; prints a line only when something needs attention.

Reads the GPU dashboard at :8765 for utilisation, and each machine's queue/<host>.failed for a
paused queue. Idle counts persist in watch_state.json so a single quiet poll is not reported —
only a GPU that is idle twice in a row, which is what a stalled queue looks like.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "watch_state.json"
API = "http://100.93.247.70:8765/api/state"
REMOTE = {"server1": "leap@100.71.184.57", "server5": "rtx3090", "server3": "leap@100.126.236.97"}
REMOTE_QUEUE = "workspace/cot-distill-lora-bit/experiments/lambda_lora_bits/queue"


def run(cmd, timeout):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return ""


def main():
    alerts = []
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    idle = state.get("idle", {})

    body = run(["curl", "-s", "--max-time", "20", API], 30)
    try:
        hosts = json.loads(body)["hosts"]
    except (ValueError, KeyError, TypeError):
        hosts = []
        alerts.append("WARN GPU 대시보드 응답 없음 (:8765) — 대시보드만의 문제일 수 있으니 큐는 별도 확인")

    for host in hosts:
        hid = host.get("id", "?")
        if host.get("status") != "online":
            alerts.append(f"ALERT {hid} 접속 끊김 (status={host.get('status')})")
            continue
        gpu = (host.get("gpus") or [{}])[0]
        util, mem = gpu.get("util") or 0.0, gpu.get("mem_used") or 0.0
        if util < 5 and mem < 2000:
            idle[hid] = idle.get(hid, 0) + 1
            if idle[hid] >= 2:
                alerts.append(f"ALERT {hid} GPU 유휴 {idle[hid]}회 연속 (util={util:.0f}% mem={mem:.0f}MiB) — 큐 확인 필요")
                idle[hid] = 0
        else:
            idle[hid] = 0

    # local queue (server2)
    failed = HERE / "server2.failed"
    if failed.exists() and failed.stat().st_size:
        alerts.append(f"ALERT server2 큐 정지 — 실패: {failed.read_text().split()}")
        failed.write_text("")

    for name, target in REMOTE.items():
        out = run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", target,
                   f"cd {REMOTE_QUEUE} && if [ -s {name}.failed ]; then cat {name}.failed; : > {name}.failed; fi"], 30)
        if out:
            alerts.append(f"ALERT {name} 큐 정지 — 실패: {out.split()}")

    STATE.write_text(json.dumps({"idle": idle}))
    for line in alerts:
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
