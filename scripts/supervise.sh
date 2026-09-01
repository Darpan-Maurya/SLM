#!/usr/bin/env bash
# Relaunch training until it finishes, so a preemption is a pause and not a failure.
set -uo pipefail

CONFIG="${1:?usage: supervise.sh <config.yaml> [key=value ...]}"
shift || true

PYTHON="${PYTHON:-python}"
MAX_RESTARTS="${MAX_RESTARTS:-500}"
BACKOFF="${BACKOFF:-15}"
RUN_NAME="$($PYTHON - "$CONFIG" "$@" <<'PY'
import sys
sys.path.insert(0, ".")
from slm.config import load_config
cfgs = [a for a in sys.argv[1:] if a.endswith((".yaml", ".yml"))]
over = [a for a in sys.argv[1:] if a not in cfgs]
c = load_config(cfgs, over)
print(f"{c.train.out_dir}/{c.train.run_name}")
PY
)"

echo "[supervise] run dir: $RUN_NAME"
echo "[supervise] every launch resumes from the newest valid checkpoint"

attempt=0
while (( attempt < MAX_RESTARTS )); do
    if [[ -f "$RUN_NAME/DONE" ]]; then
        echo "[supervise] DONE marker present - training complete"
        exit 0
    fi

    attempt=$(( attempt + 1 ))
    echo "[supervise] === launch $attempt at $(date -u +%FT%TZ) ==="
    $PYTHON scripts/train.py "$CONFIG" "$@"
    code=$?

    if [[ -f "$RUN_NAME/DONE" ]]; then
        echo "[supervise] finished cleanly after $attempt launch(es)"
        exit 0
    fi
    if [[ -f "$RUN_NAME/STOP" ]]; then
        echo "[supervise] STOP file present - not relaunching"
        exit 0
    fi
    if (( code != 0 )); then
        echo "[supervise] exit code $code; retrying in ${BACKOFF}s"
    else
        echo "[supervise] interrupted before the budget was reached; resuming in ${BACKOFF}s"
    fi
    sleep "$BACKOFF"
done

echo "[supervise] gave up after $MAX_RESTARTS launches" >&2
exit 1
