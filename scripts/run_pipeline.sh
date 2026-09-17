#!/usr/bin/env bash
# Optional full-loop wrapper: runs iSFT iteration by iteration, re-deploying
# vLLM with the freshly trained checkpoints between iterations.
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
BASE_MODEL="$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['base_model_path'])")"

RUN_NAME="$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['run_name'])")"
TOTAL="$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['iteration_times'])")"
FROM_INITIAL="$(python -c "import yaml; print(bool(yaml.safe_load(open('${CONFIG}')).get('from_initial', False)))")"
CKPT_ROOT="checkpoints/${RUN_NAME}"

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
  scripts/deploy_vllm.sh "${ALICE}" "${BOB}"
  python sft_script.py --config "${CONFIG}" --iteration "${i}"
  scripts/deploy_vllm.sh stop
done

echo "[pipeline] done"
