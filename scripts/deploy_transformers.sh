#!/usr/bin/env bash
# Deploy independent Alice/Bob Transformers servers for Gemma 2 on a T4.
set -u

ALICE_PORT="${ALICE_PORT:-8100}"
BOB_PORT="${BOB_PORT:-8101}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DTYPE="${DTYPE:-half}"
PID_DIR="logs/transformers_pids"

stop_one() {
  local pid_file="$1"
  if [ -f "${pid_file}" ]; then
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      local command
      command="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
      if [[ "${command}" == *"scripts/transformers_openai_server.py"* ]]; then
        kill "${pid}" 2>/dev/null || true
        for _ in $(seq 1 30); do
          kill -0 "${pid}" 2>/dev/null || break
          sleep 1
        done
        kill -9 "${pid}" 2>/dev/null || true
      else
        echo "[deploy] stale PID file ignored: ${pid_file} -> ${pid}"
      fi
    fi
    rm -f "${pid_file}"
  fi
}

stop_all() {
  stop_one "${PID_DIR}/alice.pid"
  stop_one "${PID_DIR}/bob.pid"
  echo "[deploy] stopped Alice and Bob Transformers servers"
}

if [ "${1:-}" = "stop" ]; then
  stop_all
  exit 0
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[deploy] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ "${1:-}" = "start-bob" ]; then
  BOB_MODEL="${2:-google/gemma-2-2b-it}"
  mkdir -p logs "${PID_DIR}"
  stop_one "${PID_DIR}/bob.pid"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup "${PYTHON_BIN}" \
    scripts/transformers_openai_server.py \
    --model "${BOB_MODEL}" --served-model-name bob \
    --host 127.0.0.1 --port "${BOB_PORT}" --device cuda:0 --dtype "${DTYPE}" \
    > logs/transformers_bob.log 2>&1 &
  echo $! > "${PID_DIR}/bob.pid"
  echo "[deploy] Bob is loading: tail -f logs/transformers_bob.log"
  exit 0
fi

ALICE_MODEL="${1:-google/gemma-2-2b-it}"
BOB_MODEL="${2:-google/gemma-2-2b-it}"
mkdir -p logs "${PID_DIR}"
stop_all

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup "${PYTHON_BIN}" \
  scripts/transformers_openai_server.py \
  --model "${ALICE_MODEL}" --served-model-name alice \
  --host 127.0.0.1 --port "${ALICE_PORT}" --device cuda:0 --dtype "${DTYPE}" \
  > logs/transformers_alice.log 2>&1 &
echo $! > "${PID_DIR}/alice.pid"

echo "[deploy] Alice is loading: tail -f logs/transformers_alice.log"
echo "[deploy] After Alice is ready, start Bob with:"
echo "  PYTHON_BIN=${PYTHON_BIN} scripts/deploy_transformers.sh start-bob '${BOB_MODEL}'"

if [ "${START_BOTH:-0}" = "1" ]; then
  while ! grep -q '^\[ready\]' logs/transformers_alice.log 2>/dev/null; do
    kill -0 "$(cat "${PID_DIR}/alice.pid")" 2>/dev/null || {
      echo "[deploy] Alice exited; inspect logs/transformers_alice.log" >&2
      exit 1
    }
    sleep 5
  done
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup "${PYTHON_BIN}" \
    scripts/transformers_openai_server.py \
    --model "${BOB_MODEL}" --served-model-name bob \
    --host 127.0.0.1 --port "${BOB_PORT}" --device cuda:0 --dtype "${DTYPE}" \
    > logs/transformers_bob.log 2>&1 &
  echo $! > "${PID_DIR}/bob.pid"
  echo "[deploy] Bob is loading: tail -f logs/transformers_bob.log"
fi
