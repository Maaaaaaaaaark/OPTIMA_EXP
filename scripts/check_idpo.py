"""Small audit report for a generated dual-agent iDPO iteration."""
from argparse import ArgumentParser
from collections import Counter
import json
import os
import sys

from datasets import load_from_disk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.run_config import load_run_config


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()
    cfg = load_run_config(args.config)
    path = cfg.dpo_rewarded_path(args.iteration)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    states = [state for row in rows for state in row.get("states", [])]
    branches = [branch for state in states for branch in state.get("branches", [])]
    pairs = []
    if os.path.exists(cfg.dpo_pairs_path(args.iteration)):
        pairs = [
            json.loads(line)
            for line in open(cfg.dpo_pairs_path(args.iteration), encoding="utf-8")
            if line.strip()
        ]
    print("=" * 74)
    print(f"iDPO audit — {cfg.run_name} / iteration {args.iteration}")
    print("=" * 74)
    print(f"tasks: {len(rows)} / expected {cfg.sample_count}")
    print(f"states: {len(states)} (Alice root + Bob-after-Alice expected per task)")
    print(f"rollout branches: {len(branches)}")
    print(f"terminations: {dict(Counter(b.get('termination_reason') for b in branches))}")
    print(f"preference pairs: {len(pairs)} / {dict(Counter(p.get('speaker') for p in pairs))}")
    if pairs:
        distances = [float(pair["distance"]) for pair in pairs]
        chosen = [float(pair["chosen_value"]) for pair in pairs]
        print(f"mean chosen value: {sum(chosen) / len(chosen):.4f}")
        print(f"mean reward gap: {sum(distances) / len(distances):.4f}")
        print(
            "threshold check: ",
            all(v > cfg.dpo.min_value for v in chosen)
            and all(v > cfg.dpo.min_reward_gap for v in distances),
        )
    for speaker, dataset_path in (
        ("Alice", cfg.alice_dpo_dataset_path(args.iteration)),
        ("Bob", cfg.bob_dpo_dataset_path(args.iteration)),
    ):
        if os.path.exists(dataset_path):
            ds = load_from_disk(dataset_path)
            print(f"{speaker} dataset: train {len(ds['train'])} / test {len(ds['test'])}")
        else:
            print(f"{speaker} dataset: missing")
    for pair in pairs[: args.show]:
        print("-" * 74)
        print(
            f"task {pair['task_id']} {pair['speaker']} {pair['state_id']} | "
            f"chosen={pair['chosen_value']:.4f}, rejected={pair['rejected_value']:.4f}, "
            f"gap={pair['distance']:.4f}"
        )
        print(f"chosen:   {pair['chosen'][:300]}")
        print(f"rejected: {pair['rejected'][:300]}")


if __name__ == "__main__":
    main()
