#!/usr/bin/env bash
# Deploy the two vLLM endpoints used by the OPTIMA Qwen pipeline.
#
#   scripts/deploy_vllm.sh                       # base model on both endpoints
#   scripts/deploy_vllm.sh <alice_model> <bob_model>
#   scripts/deploy_vllm.sh stop                  # kill both endpoints
#
# Python code never starts/stops vLLM: run this BEFORE sft_script.py, and
# re-run it between iterations when checkpoints change (or use
# scripts/run_pipeline.sh which handles the full loop).
#
# Env overrides:
#   ALICE_PORT=8100  BOB_PORT=8101  GPU_ID=0  MEM_UTIL=0.35  VLLM_ENV=vllm_env
set -u

ALICE_PORT="${ALICE_PORT:-8100}"
BOB_PORT="${BOB_PORT:-8101}"
GPU_ID="${GPU_ID:-0}"
MEM_UTIL="${MEM_UTIL:-0.35}"    # two endpoints share one GPU; keep headroom for
                                # the in-process reward scorer (frozen 0.5B)
VLLM_ENV="${VLLM_ENV:-vllm_env}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

ALICE_MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
BOB_MODEL="${2:-Qwen/Qwen2.5-0.5B-Instruct}"

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${VLLM_ENV}"

stop_all() {
  pkill -f "vllm serve" 2>/dev/null || true
  echo "[deploy] stopped all 'vllm serve' processes"
}

if [ "${1:-}" = "stop" ]; then
  stop_all
  exit 0
fi

stop_all
mkdir -p logs

CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup vllm serve "${ALICE_MODEL}" \
  --host 127.0.0.1 --port "${ALICE_PORT}" \
  --served-model-name alice \
  --gpu-memory-utilization "${MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  > logs/vllm_alice.log 2>&1 &

CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup vllm serve "${BOB_MODEL}" \
  --host 127.0.0.1 --port "${BOB_PORT}" \
  --served-model-name bob \
  --gpu-memory-utilization "${MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  > logs/vllm_bob.log 2>&1 &

echo "[deploy] alice: ${ALICE_MODEL} -> http://127.0.0.1:${ALICE_PORT}/v1/chat/completions (logs/vllm_alice.log)"
echo "[deploy] bob:   ${BOB_MODEL} -> http://127.0.0.1:${BOB_PORT}/v1/chat/completions (logs/vllm_bob.log)"
