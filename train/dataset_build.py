"""Trajectory selection and speaker-aware SFT dataset construction.

Replicates the old pipeline's data_clean + process_dataloader_for_sft with
the paper-faithful defaults:
- per task, pick the highest-reward trajectory (argmax)
- globally sort by reward, keep the trim window (default top 70%), then
  filter by the >= episilon threshold
- name-mixing utterances get the -10 reward penalty (old data_clean rule)

The SFT datasets are speaker-routed: Alice's dataset uses the recorded
Alice system prompt with Alice turns as assistant and Bob turns as user;
Bob's dataset is the mirror image. Rows are templated up front, overlong
samples are filtered (SFTTrainer tail-truncation would otherwise cut the
last assistant turn), and the result is split train/test and save_to_disk.
"""
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import random
import re

import numpy as np
from datasets import Dataset, DatasetDict

from utils.run_config import RunConfig

NAME_PENALTY = -10.0


def apply_name_penalty(result: Dict[str, Any], enabled: bool = True) -> None:
    """Old data_clean rule: an utterance mixing both names (or repeating one
    name) marks a broken format and subtracts 10 from the reward."""
    penalty = 0.0
    if not enabled:
        result["name_penalty"] = penalty
        return
    for sentence in result.get("conversation", []):
        try:
            text = str(sentence)
        except Exception:
            continue
        if re.search(r"Alice:", text) and re.search(r"Bob:", text):
            penalty = NAME_PENALTY
            break
        if len(re.findall(r"Alice:", text)) >= 2 or len(re.findall(r"Bob:", text)) >= 2:
            penalty = NAME_PENALTY
            break
    result["name_penalty"] = penalty


def select_trajectories(cfg: RunConfig, iteration: int) -> List[Dict[str, Any]]:
    """Read rewarded jsonl, apply name penalties, select per task the best
    trajectory, trim to the configured window and the episilon threshold,
    and write the cleaned jsonl with selected/rank markers.

    Returns the list of selected result dicts."""
    rewarded_path = cfg.rewarded_path(iteration)
    cleaned_path = cfg.cleaned_path(iteration)
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)

    tasks: List[Dict[str, Any]] = []
    with open(rewarded_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    best_per_task: List[Dict[str, Any]] = []
    for task in tasks:
        results = task["results"]
        for result in results:
            apply_name_penalty(result, enabled=cfg.require_name_prefix)
            if result.get("reward") is None:
                result["reward"] = 0.0
        effective = [
            r["reward"] + (r.get("name_penalty") or 0.0) for r in results
        ]
        best_idx = int(np.argmax(effective))
        best = results[best_idx]
        best["_effective_reward"] = effective[best_idx]
        best_per_task.append(best)

    # global sort by effective reward, then the trim window, then threshold
    best_per_task.sort(key=lambda r: r["_effective_reward"], reverse=True)
    n = len(best_per_task)
    low = int(cfg.selection_trim_low * n)
    high = int(cfg.selection_trim_high * n)
    window = best_per_task[low:high]
    selected: List[Dict[str, Any]] = []
    rank = 0
    for result in best_per_task:
        result["selected"] = False
        result["rank"] = None
        if result in window and result["_effective_reward"] >= cfg.episilon:
            result["selected"] = True
            result["rank"] = rank
            rank += 1
            selected.append(result)
    for result in best_per_task:
        result.pop("_effective_reward", None)

    rows = sorted(tasks, key=lambda d: d.get("task_id", -1))
    with open(cleaned_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[select] {len(selected)}/{n} trajectories selected "
        f"(trim [{low},{high}), episilon {cfg.episilon})"
    )
    return selected


def _templated_row(
    tokenizer,
    result: Dict[str, Any],
    speaker: str,
    system_prompt: str,
    max_seq_length: int,
) -> Optional[Dict[str, Any]]:
    """One SFT row from the perspective of `speaker`: own turns = assistant,
    partner turns = user, using the RECORDED system prompt (not a
    re-rendered one). Returns None when the templated text is too long."""
    partner = "Bob" if speaker == "Alice" else "Alice"
    messages = [{"role": "system", "content": system_prompt}]
    for turn in result.get("turns", []):
        if turn.get("speaker") == speaker:
            messages.append({"role": "assistant", "content": turn["content"]})
        elif turn.get("speaker") == partner:
            messages.append({"role": "user", "content": turn["content"]})
        else:  # malformed turn: keep it as user content rather than dropping
            messages.append({"role": "user", "content": turn.get("content", "")})
    if not any(m["role"] == "assistant" for m in messages):
        return None
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if len(tokenizer.encode(text)) > max_seq_length:
        return None
    return {
        "text": text,
        "task_id": result.get("task_id"),
        "trajectory_id": result.get("trajectory_id"),
        "reward": result.get("reward"),
        "correct_score": result.get("correct_score"),
    }


def build_speaker_datasets(
    cfg: RunConfig,
    tokenizer,
    iteration: int,
    selected: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """Build and save the Alice/Bob datasets from the selected trajectories."""
    max_seq_length = cfg.sft.max_seq_length

    def build_one(speaker: str, system_key: str) -> int:
        rows = []
        limit = max_seq_length if cfg.sft.filter_long_samples else 10**9
        for result in selected:
            row = _templated_row(
                tokenizer, result, speaker, result.get(system_key, ""), limit
            )
            if row is not None:
                rows.append(row)
        dataset = Dataset.from_list(rows)
        n = len(rows)
        split = int(cfg.sft.train_ratio * n)
        train = dataset.select(range(split))
        test = dataset.select(range(split, n))
        return DatasetDict({"train": train, "test": test}), n

    alice_ds, n_alice = build_one("Alice", "system_first")
    bob_ds, n_bob = build_one("Bob", "system_second")

    alice_path = cfg.alice_dataset_path(iteration)
    bob_path = cfg.bob_dataset_path(iteration)
    alice_ds.save_to_disk(alice_path)
    bob_ds.save_to_disk(bob_path)
    print(
        f"[dataset] alice: {n_alice} rows -> {alice_path}\n"
        f"[dataset] bob:   {n_bob} rows -> {bob_path}"
    )
    return n_alice, n_bob
