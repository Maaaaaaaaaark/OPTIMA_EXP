# Dual-agent iDPO on Colab (Gemma 2 2B)

This pipeline keeps Alice and Bob as independent policies, checkpoints and
training datasets.  It uses the existing frozen OPTIMA reward implementation.

## What one iteration does

1. At the same Alice state, sample several candidate replies and roll each one
   to the end of the dialogue.
2. Fix one Alice reply; at that exact Bob state, sample several Bob replies and
   roll each one to the end.
3. Stop the two inference services and score every rollout with the frozen
   base model and the existing task/token/loss reward.
4. For each state, create a pair only when the best candidate clears
   `dpo.min_value` and beats the worst distinct candidate by more than
   `dpo.min_reward_gap`.
5. Train and merge Alice LoRA, release it, then train and merge Bob LoRA.

This is a bounded, T4-compatible version of the paper's MCTS preference
construction.  It preserves same-state comparisons and reward thresholds but
searches two states per task instead of the paper's large multi-node search.

## Preflight

```bash
cd /content/OPTIMA_EXP
/content/optima-train/bin/python -m py_compile \
  idpo_script.py train/idpo.py train/dpo_generate.py train/dpo_trainer.py \
  scripts/check_idpo.py utils/run_config.py

/content/optima-train/bin/python -m pytest \
  tests/test_idpo_pairs.py tests/test_reward_unchanged.py \
  tests/test_generate_resume.py tests/test_run_lock.py -q
```

## Run safely in three stages

First deploy the independent iSFT Alice and Bob checkpoints with
`scripts/deploy_transformers.sh`, as for the previous evaluation.  Then:

```bash
# 1. Generate rollouts while both inference services are running.
/content/optima-train/bin/python idpo_script.py \
  --config configs/gemma2-2b/hotpot_qa_colab_idpo.yaml \
  --iteration 0 --stage generate --overwrite

# 2. Stop inference automatically, score, and create preference datasets.
/content/optima-train/bin/python idpo_script.py \
  --config configs/gemma2-2b/hotpot_qa_colab_idpo.yaml \
  --iteration 0 --stage score

# 3. Audit before any parameter update.
/content/optima-train/bin/python scripts/check_idpo.py \
  --config configs/gemma2-2b/hotpot_qa_colab_idpo.yaml \
  --iteration 0 --show 5

# 4. Train Alice and Bob sequentially only if both datasets are non-empty.
/content/optima-train/bin/python idpo_script.py \
  --config configs/gemma2-2b/hotpot_qa_colab_idpo.yaml \
  --iteration 0 --stage train
```

For Debate, replace the config path with
`configs/gemma2-2b/arc_colab_idpo.yaml` after ARC iSFT iteration 0 has produced
both checkpoints.

Do not use `--overwrite` for `score` or `train`; it is intentionally rejected.
