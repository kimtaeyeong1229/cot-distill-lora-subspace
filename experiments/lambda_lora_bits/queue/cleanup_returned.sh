#!/usr/bin/env bash
# Remove this project's footprint from the two machines being returned, once both keepers have it.
#
# Gate first: server2 was already verified file-by-file (every hash on server1 and server3 exists
# here), and this waits for the server5 snapshot to finish and checks its size before touching
# anything. Deletion is irreversible and one of the two machines is shared, so the scope is written
# out explicitly rather than swept with wildcards.
#
# Deliberately left alone:
#   server3 ~/.cache/huggingface   236 GB of leapshared/*GR00T17 and lerobot/* -- the lab's robotics
#                                  project, not ours
#   docker build cache             74 GB / 136 GB, produced by other people's builds; our image was
#                                  pulled, not built, so `docker system prune` would destroy their
#                                  work and reclaim nothing of ours
#   every other container/image    autobio_benchmark, vla-bench, qwen38-bench (llama.cpp),
#                                  openarmsciedu-*, vla-serving
set -uo pipefail
cd "$(dirname "$0")/../../.."

DEST=tykim@100.115.125.36
SNAP=workspace/backup/cot-distill-lora-bit/2026-09-21
IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel
CONTAINER=cot-distill-lora-bit
REPO=workspace/cot-distill-lora-bit
declare -A RETURN=( [server1]=leap@100.71.184.57 [server3]=leap@100.126.236.97 )

expected=$(du -sm --exclude=teachers --exclude=teachers_r1tok --exclude=models --exclude=cache \
                  --exclude=__pycache__ --exclude=hf_upload . | cut -f1)

echo "[$(date '+%F %T')] server5 스냅샷 완료 대기 (목표 ${expected}MB)"
while pgrep -f "rsyn[c].*2026-09-21" > /dev/null; do sleep 60; done

got=$(ssh -o ConnectTimeout=20 "${DEST}" "du -sm ${SNAP} 2>/dev/null | cut -f1")
files=$(ssh -o ConnectTimeout=20 "${DEST}" "find ${SNAP} -type f 2>/dev/null | wc -l")
here=$(find . -type f \( -path './models/*' -o -path '*/teachers/*' -o -path '*/teachers_r1tok/*' \
        -o -path '*/cache/*' -o -name '*.pyc' -o -path './experiments/hf_upload/*' \) -prune \
        -o -type f -print | wc -l)
echo "  server5: ${got}MB / ${files}개 파일   (server2 기준 ${expected}MB / ${here}개)"

if [[ -z "${got}" ]] || (( got * 100 < expected * 95 )); then
  echo "  중단: server5 스냅샷이 기대치의 95%에 못 미침. 아무것도 지우지 않음."
  exit 1
fi
echo "  통과 — 삭제 진행"

for name in "${!RETURN[@]}"; do
  host="${RETURN[$name]}"
  echo
  echo "[$(date '+%F %T')] === ${name}"
  ssh -o ConnectTimeout=30 "${host}" "
    before=\$(df -h ~ | tail -1 | awk '{print \$4}')
    docker rm -f ${CONTAINER} >/dev/null 2>&1 && echo '  컨테이너 제거' || echo '  컨테이너 없음'
    docker rmi ${IMAGE} >/dev/null 2>&1 && echo '  이미지 제거' || echo '  이미지 제거 실패 또는 없음'
    if [[ -d ${REPO} ]]; then
      keys=\$(find ${REPO} -name '*.key' 2>/dev/null | wc -l)
      sudo rm -rf ${REPO} 2>/dev/null || rm -rf ${REPO}
      [[ -d ${REPO} ]] && echo '  ★ 저장소 삭제 실패 (권한)' || echo \"  저장소 삭제 (비밀키 \${keys}개 포함)\"
    else
      echo '  저장소 없음'
    fi
    echo \"  디스크 여유: \${before} → \$(df -h ~ | tail -1 | awk '{print \$4}')\"
    echo '  남은 컨테이너:'; docker ps -a --format '    {{.Names}}' 2>/dev/null | head -8
  "
done
echo
echo "[$(date '+%F %T')] 정리 완료"
