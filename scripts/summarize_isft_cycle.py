"""Create JSON/Markdown summaries for a two-stage iSFT cycle."""
from argparse import ArgumentParser
from collections import Counter
import csv
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.run_config import load_run_config


def load_results(path):
    results = []
    if not os.path.exists(path):
        return results
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                results.extend(json.loads(line).get("results", []))
    return results


def duplicated(result):
    seen = set()
    for text in result.get("conversation", []):
        if text in ("large", "error"):
            continue
        normalized = re.sub(r"^\s*(Alice|Bob)\s*:\s*", "", str(text)).strip().lower()
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


def summarize(path):
    results = load_results(path)
    n = len(results)
    tokens = [int(result.get("token_count", 0)) for result in results]
    correct = [float(result.get("correct_score") or 0.0) for result in results]
    reward = [float(result.get("reward") or 0.0) for result in results]
    selected = [result for result in results if result.get("selected")]
    return {
        "trajectories": n,
        "mean_correct_score": statistics.mean(correct) if correct else 0.0,
        "fully_correct": sum(value == 1.0 for value in correct),
        "mean_reward": statistics.mean(reward) if reward else 0.0,
        "mean_tokens": statistics.mean(tokens) if tokens else 0.0,
        "median_tokens": statistics.median(tokens) if tokens else 0.0,
        "parsed_answer": sum(bool(result.get("final_answer")) for result in results),
        "selected": len(selected),
        "name_penalized": sum(float(result.get("name_penalty") or 0.0) < 0 for result in results),
        "duplicate_trajectory": sum(duplicated(result) for result in results),
        "termination": dict(Counter(result.get("termination_reason") for result in results)),
    }


def percent(value, total):
    return 100.0 * value / total if total else 0.0


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--post-run", required=True)
    args = parser.parse_args()
    cfg = load_run_config(args.config)
    paths = {
        "isft_iteration_0": cfg.cleaned_path(0),
        "isft_iteration_1": cfg.cleaned_path(1),
        "validation_baseline": os.path.join(
            cfg.runs_root, args.baseline_run, "iteration_0", "cleaned", "iteration_0.jsonl"
        ),
        "validation_post_isft0": os.path.join(
            cfg.runs_root, args.post_run, "iteration_0", "cleaned", "iteration_0.jsonl"
        ),
    }
    report = {name: summarize(path) for name, path in paths.items()}
    out_dir = os.path.join(cfg.run_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "isft_cycle_summary.json")
    csv_path = os.path.join(out_dir, "isft_cycle_summary.csv")
    md_path = os.path.join(out_dir, "isft_cycle_summary.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    csv_stages = (
        ("Generation iteration 0", report["isft_iteration_0"]),
        ("Generation iteration 1", report["isft_iteration_1"]),
        ("Validation baseline", report["validation_baseline"]),
        ("Validation post-iSFT0", report["validation_post_isft0"]),
    )
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "stage", "trajectories", "mean_correct", "fully_correct",
            "mean_tokens", "parsed_answer", "agreement",
        ])
        for label, values in csv_stages:
            writer.writerow([
                label,
                values["trajectories"],
                values["mean_correct_score"],
                values["fully_correct"],
                values["mean_tokens"],
                values["parsed_answer"],
                values["termination"].get("agreement", 0),
            ])

    base = report["validation_baseline"]
    post = report["validation_post_isft0"]
    lines = [
        f"# {cfg.run_name}: iSFT cycle summary",
        "",
        "## Fixed validation: base vs iSFT iteration 0",
        "",
        "| Metric | Base | Post-iSFT0 | Delta |",
        "|---|---:|---:|---:|",
        f"| Mean correct | {base['mean_correct_score']:.4f} | {post['mean_correct_score']:.4f} | {post['mean_correct_score']-base['mean_correct_score']:+.4f} |",
        f"| Fully correct | {base['fully_correct']}/{base['trajectories']} | {post['fully_correct']}/{post['trajectories']} | {post['fully_correct']-base['fully_correct']:+d} |",
        f"| Mean tokens | {base['mean_tokens']:.2f} | {post['mean_tokens']:.2f} | {post['mean_tokens']-base['mean_tokens']:+.2f} |",
        f"| Parsed answer | {percent(base['parsed_answer'], base['trajectories']):.1f}% | {percent(post['parsed_answer'], post['trajectories']):.1f}% | {percent(post['parsed_answer'], post['trajectories'])-percent(base['parsed_answer'], base['trajectories']):+.1f} pp |",
        f"| Agreement | {percent(base['termination'].get('agreement',0), base['trajectories']):.1f}% | {percent(post['termination'].get('agreement',0), post['trajectories']):.1f}% | {percent(post['termination'].get('agreement',0), post['trajectories'])-percent(base['termination'].get('agreement',0), base['trajectories']):+.1f} pp |",
        "",
        "## Training rollouts (different train slices; descriptive only)",
        "",
        "| Metric | Iteration 0 | Iteration 1 |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Trajectories", "trajectories"),
        ("Mean correct", "mean_correct_score"),
        ("Fully correct", "fully_correct"),
        ("Mean reward", "mean_reward"),
        ("Mean tokens", "mean_tokens"),
        ("Selected", "selected"),
        ("Name penalized", "name_penalized"),
        ("Duplicate trajectory", "duplicate_trajectory"),
    ):
        left = report["isft_iteration_0"][key]
        right = report["isft_iteration_1"][key]
        lines.append(f"| {label} | {left} | {right} |")
    lines.extend(
        [
            "",
            "> Iteration 0 and iteration 1 use different training questions. The fixed validation table is the causal before/after comparison. This run uses the 50-task one-GPU comparison scale.",
            "",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
