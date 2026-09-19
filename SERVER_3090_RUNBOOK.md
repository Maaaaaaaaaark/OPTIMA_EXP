# One-round OPTIMA reproduction on one RTX 3090

This workflow uses exactly one selected physical GPU. Alice and Bob remain
independent models, inference endpoints, training datasets, and checkpoints.
Their expensive stages run sequentially:

1. serve Alice and Bob for rollout generation;
2. use the same frozen Gemma base model for the author's reward formula;
3. stop both servers before training;
4. train Alice, then train Bob;
5. redeploy the two new checkpoints for iteration-1 generation and fixed
   validation evaluation.

The pipeline contains three arms: iSFT, standalone iDPO, and the paper's
iSFT-DPO hybrid. Each arm trains only iteration 0 and then generates iteration
1 without a second training step. Standalone iDPO starts from the corresponding
iSFT iteration-0 checkpoints, as in the author's recipes.

The iDPO implementation follows the repository rules: eight MCTS searches,
three rollouts per expansion, top-10 softmax node selection, online frozen
trajectory-reward backpropagation, same-parent best/worst pairs, reward/gap
thresholds, top-50% pair retention, and RPO (`rpo_alpha=1`). The hybrid uses
standard DPO, matching the author code.

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
  scripts/run_hybrid_cycle_single_gpu.py \
  scripts/summarize_isft_cycle.py \
  scripts/summarize_idpo_cycle.py \
  scripts/summarize_hybrid_cycle.py \
  scripts/summarize_all_methods.py

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

To run only one method:

```bash
python -u scripts/run_optima_single_gpu.py \
  --task information --gpu-id 2 --stage isft

python -u scripts/run_optima_single_gpu.py \
  --task information --gpu-id 2 --stage idpo

python -u scripts/run_optima_single_gpu.py \
  --task information --gpu-id 2 --stage hybrid
```

## Outputs

Information Exchange:

- iSFT report: `runs/gemma2-2b-hotpotqa-3090-isft/comparison/isft_cycle_summary.md`
- iDPO report: `runs/gemma2-2b-hotpotqa-3090-idpo/comparison/idpo_cycle_summary.md`
- hybrid report: `runs/gemma2-2b-hotpotqa-3090-hybrid/comparison/hybrid_cycle_summary.md`
- all-method English report: `reports/information_exchange/all_methods_summary.md`
- CSV: `reports/information_exchange/all_methods_summary.csv`
- iSFT checkpoints: `checkpoints/gemma2-2b-hotpotqa-3090-isft/{alice,bob}/iteration_0`
- iDPO checkpoints: `checkpoints/gemma2-2b-hotpotqa-3090-idpo/{alice,bob}/iteration_0`
- hybrid checkpoints: `checkpoints/gemma2-2b-hotpotqa-3090-hybrid/{alice,bob}/dpo/iteration_0`

Debate:

- iSFT report: `runs/gemma2-2b-arc-3090-isft/comparison/isft_cycle_summary.md`
- iDPO report: `runs/gemma2-2b-arc-3090-idpo/comparison/idpo_cycle_summary.md`
- hybrid report: `runs/gemma2-2b-arc-3090-hybrid/comparison/hybrid_cycle_summary.md`
- all-method English report: `reports/debate/all_methods_summary.md`
- CSV: `reports/debate/all_methods_summary.csv`
- iSFT checkpoints: `checkpoints/gemma2-2b-arc-3090-isft/{alice,bob}/iteration_0`
- iDPO checkpoints: `checkpoints/gemma2-2b-arc-3090-idpo/{alice,bob}/iteration_0`
- hybrid checkpoints: `checkpoints/gemma2-2b-arc-3090-hybrid/{alice,bob}/dpo/iteration_0`

The iteration-0/iteration-1 training-rollout table is descriptive because the
questions differ. The fixed validation table is the valid before/after model
comparison.

## Reproduction scope

Algorithmic rules, reward coefficients, MCTS settings, selection thresholds,
and task-specific training hyperparameters follow the author repository. The
intentional changes are:

- `google/gemma-2-2b-it` replaces Llama 3 8B;
- Alice and Bob are separate policies instead of one shared policy;
- only iteration 0 is trained, followed by iteration-1 generation;
- LoRA is used so each independent Gemma policy can train on one 24 GB GPU;
- each arm uses 50 training tasks as a compute-controlled run. The author's
  recipes use 2,000-10,000 iSFT tasks and 10,000-13,000 MCTS tasks.

Therefore this is author-rule-faithful but not model-, scale-, or
parameterization-identical. The generated English reports record this scope.
