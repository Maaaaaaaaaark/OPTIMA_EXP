# OPTIMA single-RTX-3090 runbook

This workflow uses exactly one selected physical GPU. Alice and Bob remain
independent models and independent checkpoints, but their expensive stages run
sequentially:

1. serve Alice and Bob for rollout generation;
2. stop both servers and load the frozen reward model;
3. train Alice, then train Bob;
4. redeploy the two new checkpoints for the next rollout/evaluation.

The default gate requires 21,000 MiB free on the selected 24,576 MiB RTX 3090.
It waits when the GPU is busy and never kills unrelated users' jobs.

## Preflight

Run from the repository root with the training environment activated:

```bash
cd /data/yuheng/OPTIMA_EXP

nvidia-smi
python -m py_compile \
  scripts/run_optima_single_gpu.py \
  scripts/run_isft_cycle_single_gpu.py \
  scripts/run_idpo_cycle_single_gpu.py \
  scripts/summarize_isft_cycle.py \
  scripts/summarize_idpo_cycle.py

python -m pytest \
  tests/test_conversation_flow.py \
  tests/test_idpo_pairs.py \
  tests/test_generate_resume.py \
  tests/test_reward_unchanged.py -q
```

Replace `/data/yuheng/OPTIMA_EXP` and `python` if the server uses different
paths. The Hugging Face account must already have Gemma 2 access.

## One-command experiment

First choose a GPU that is assigned to you and is idle. The example uses
physical GPU 2.

Information Exchange (HotpotQA):

```bash
nohup python -u scripts/run_optima_single_gpu.py \
  --task information \
  --gpu-id 2 \
  --reset \
  > logs/optima_information_gpu2.log 2>&1 &
echo $! > logs/optima_information_gpu2.pid
```

Debate (ARC), after the Information run completes:

```bash
nohup python -u scripts/run_optima_single_gpu.py \
  --task debate \
  --gpu-id 2 \
  --reset \
  > logs/optima_debate_gpu2.log 2>&1 &
echo $! > logs/optima_debate_gpu2.pid
```

Do not launch both commands simultaneously when only one GPU is allocated.
To run both experiments serially without watching them, use:

```bash
nohup python -u scripts/run_optima_single_gpu.py \
  --task both \
  --gpu-id 2 \
  --reset \
  > logs/optima_both_gpu2.log 2>&1 &
echo $! > logs/optima_both_gpu2.pid
```

## Resume and monitoring

The pipeline is resumable. If SSH disconnects or a stage fails, rerun the same
command **without** `--reset`; completed task rows and checkpoints are reused.

```bash
tail -f logs/optima_information_gpu2.log
ps -fp "$(cat logs/optima_information_gpu2.pid)"
nvidia-smi -i 2
```

To run only one half:

```bash
python -u scripts/run_optima_single_gpu.py \
  --task information --gpu-id 2 --stage isft

python -u scripts/run_optima_single_gpu.py \
  --task information --gpu-id 2 --stage idpo
```

## Outputs

Information Exchange:

- iSFT report: `runs/gemma2-2b-hotpotqa-3090-isft/comparison/isft_cycle_summary.md`
- iDPO report: `runs/gemma2-2b-hotpotqa-3090-idpo/comparison/idpo_cycle_summary.md`
- iSFT checkpoints: `checkpoints/gemma2-2b-hotpotqa-3090-isft/{alice,bob}/iteration_0`
- iDPO checkpoints: `checkpoints/gemma2-2b-hotpotqa-3090-idpo/{alice,bob}/iteration_0`

Debate:

- iSFT report: `runs/gemma2-2b-arc-3090-isft/comparison/isft_cycle_summary.md`
- iDPO report: `runs/gemma2-2b-arc-3090-idpo/comparison/idpo_cycle_summary.md`
- iSFT checkpoints: `checkpoints/gemma2-2b-arc-3090-isft/{alice,bob}/iteration_0`
- iDPO checkpoints: `checkpoints/gemma2-2b-arc-3090-idpo/{alice,bob}/iteration_0`

The iteration-0/iteration-1 training-rollout table is descriptive because the
questions differ. The fixed validation table is the valid before/after model
comparison.

## Scope of this iDPO implementation

The included iDPO stage compares multiple candidate replies from the same
conversation state and trains Alice and Bob on separate chosen/rejected pairs.
It is a bounded single-GPU approximation with two search states per task, not
the paper's full author-scale multi-node MCTS search.
