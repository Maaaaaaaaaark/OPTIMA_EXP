#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/content/optima-train/bin/python}"
DEPLOY="scripts/deploy_transformers.sh"
BASE_MODEL="google/gemma-2-2b-it"
ALICE_SFT="checkpoints/gemma2-2b-arc-colab-medium/alice/iteration_0"
BOB_SFT="checkpoints/gemma2-2b-arc-colab-medium/bob/iteration_0"
BASE_CFG="configs/gemma2-2b/arc_baseline_fixed_eval50.yaml"
POST_CFG="configs/gemma2-2b/arc_post_isft_fixed_eval50.yaml"

mkdir -p logs

cleanup() {
  PYTHON_BIN="${PYTHON_BIN}" "${DEPLOY}" stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_endpoint() {
  local url="$1"
  local label="$2"
  for _ in $(seq 1 120); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "[compare] ${label} ready"
      return 0
    fi
    sleep 5
  done
  echo "[compare] timed out waiting for ${label}" >&2
  tail -n 80 logs/transformers_alice.log 2>/dev/null || true
  tail -n 80 logs/transformers_bob.log 2>/dev/null || true
  return 1
}

deploy_pair() {
  local alice_model="$1"
  local bob_model="$2"
  START_BOTH=1 PYTHON_BIN="${PYTHON_BIN}" DTYPE=half \
    "${DEPLOY}" "${alice_model}" "${bob_model}"
  wait_endpoint "http://127.0.0.1:8100/v1/models" "Alice"
  wait_endpoint "http://127.0.0.1:8101/v1/models" "Bob"
}

test -f "${ALICE_SFT}/config.json"
test -f "${BOB_SFT}/config.json"

echo "[compare] 1/3 fixed baseline evaluation"
deploy_pair "${BASE_MODEL}" "${BASE_MODEL}"
"${PYTHON_BIN}" sft_script.py --config "${BASE_CFG}" --iteration 0 --no_train --overwrite \
  2>&1 | tee logs/arc_baseline_fixed_eval50.log

echo "[compare] 2/3 fixed post-iSFT evaluation"
deploy_pair "${ALICE_SFT}" "${BOB_SFT}"
"${PYTHON_BIN}" sft_script.py --config "${POST_CFG}" --iteration 0 --no_train --overwrite \
  2>&1 | tee logs/arc_post_isft_fixed_eval50.log

echo "[compare] 3/3 comparison"
"${PYTHON_BIN}" scripts/compare_arc_eval_runs.py \
  --baseline runs/gemma2-2b-arc-baseline-fixed-eval50/iteration_0/cleaned/iteration_0.jsonl \
  --post runs/gemma2-2b-arc-post-isft-fixed-eval50/iteration_0/cleaned/iteration_0.jsonl \
  --output runs/arc_debate_isft_comparison_fixed.json \
  2>&1 | tee logs/arc_debate_isft_comparison_fixed.txt

echo "[compare] all done"
