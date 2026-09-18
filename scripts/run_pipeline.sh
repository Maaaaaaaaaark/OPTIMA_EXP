#!/usr/bin/env bash
# Optional full-loop wrapper: runs iSFT iteration by iteration, re-deploying
# the selected inference backend with freshly trained checkpoints.
#
#   scripts/run_pipeline.sh                          # hotpot_qa config, all iterations
#   scripts/run_pipeline.sh configs/qwen0.5b/arc.yaml
#   scripts/run_pipeline.sh configs/qwen0.5b/hotpot_qa.yaml 2   # first 2 iterations
#
# Each sft_script.py call re-runs iterations 0..i: generation/resume skips
# completed tasks, already-trained checkpoints are skipped, so only the new
# iteration does real work.
set -u

CONFIG="${1:-configs/qwen0.5b/hotpot_qa.yaml}"
N_ITER="${2:-}"
DEPLOY_BACKEND="${DEPLOY_BACKEND:-vllm}"
PIPELINE_PYTHON="${PYTHON_BIN:-python}"
BASE_MODEL="$("${PIPELINE_PYTHON}" -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['base_model_path'])")"

RUN_NAME="$("${PIPELINE_PYTHON}" -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['run_name'])")"
TOTAL="$("${PIPELINE_PYTHON}" -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['iteration_times'])")"
FROM_INITIAL="$("${PIPELINE_PYTHON}" -c "import yaml; print(bool(yaml.safe_load(open('${CONFIG}')).get('from_initial', False)))")"
CKPT_ROOT="checkpoints/${RUN_NAME}"

if [ "${DEPLOY_BACKEND}" = "transformers" ]; then
  DEPLOY_SCRIPT="scripts/deploy_transformers.sh"
else
  DEPLOY_SCRIPT="scripts/deploy_vllm.sh"
fi

if [ -n "${N_ITER}" ]; then
  TOTAL="${N_ITER}"
fi

for ((i = 0; i < TOTAL; i++)); do
  ALICE="${BASE_MODEL}"
  BOB="${BASE_MODEL}"
  if [ "${i}" -gt 0 ] && [ "${FROM_INITIAL}" != "True" ]; then
    ALICE="${CKPT_ROOT}/alice/iteration_$((i - 1))"
    BOB="${CKPT_ROOT}/bob/iteration_$((i - 1))"
  fi
  echo "===== iteration ${i}: alice=${ALICE} bob=${BOB} ====="
  if [ "${DEPLOY_BACKEND}" = "transformers" ]; then
    START_BOTH=1 "${DEPLOY_SCRIPT}" "${ALICE}" "${BOB}"
  else
    "${DEPLOY_SCRIPT}" "${ALICE}" "${BOB}"
  fi
  "${PIPELINE_PYTHON}" sft_script.py --config "${CONFIG}" --iteration "${i}"
  "${DEPLOY_SCRIPT}" stop
done

echo "[pipeline] done"
