"""Compare fixed validation runs before iSFT, after iSFT, and after iDPO."""
from argparse import ArgumentParser
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.summarize_isft_cycle import percent, summarize
from utils.run_config import load_run_config


def load_pairs(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validation_path(runs_root, run_name):
    return os.path.join(
        runs_root, run_name, "iteration_0", "cleaned", "iteration_0.jsonl"
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--idpo-config", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--post-isft-run", required=True)
    parser.add_argument("--post-idpo-run", required=True)
    args = parser.parse_args()
    cfg = load_run_config(args.idpo_config)

    report = {
        "validation_baseline": summarize(validation_path(cfg.runs_root, args.baseline_run)),
        "validation_post_isft0": summarize(validation_path(cfg.runs_root, args.post_isft_run)),
        "validation_post_idpo0": summarize(validation_path(cfg.runs_root, args.post_idpo_run)),
    }
    pairs = load_pairs(cfg.dpo_pairs_path(0))
    gaps = [float(row["distance"]) for row in pairs]
    chosen_values = [float(row["chosen_value"]) for row in pairs]
    report["idpo_preferences"] = {
        "pairs": len(pairs),
        "alice_pairs": sum(row.get("speaker") == "Alice" for row in pairs),
        "bob_pairs": sum(row.get("speaker") == "Bob" for row in pairs),
        "mean_chosen_value": statistics.mean(chosen_values) if chosen_values else 0.0,
        "mean_reward_gap": statistics.mean(gaps) if gaps else 0.0,
    }

    out_dir = os.path.join(cfg.run_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "idpo_cycle_summary.json")
    md_path = os.path.join(out_dir, "idpo_cycle_summary.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    stages = [
        ("Base", report["validation_baseline"]),
        ("Post-iSFT0", report["validation_post_isft0"]),
        ("Post-iDPO0", report["validation_post_idpo0"]),
    ]
    lines = [
        f"# {cfg.run_name}: fixed-validation comparison",
        "",
        "| Stage | Mean correct | Fully correct | Mean tokens | Parsed | Agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, values in stages:
        total = values["trajectories"]
        lines.append(
            f"| {label} | {values['mean_correct_score']:.4f} | "
            f"{values['fully_correct']}/{total} | {values['mean_tokens']:.2f} | "
            f"{percent(values['parsed_answer'], total):.1f}% | "
            f"{percent(values['termination'].get('agreement', 0), total):.1f}% |"
        )
    pref = report["idpo_preferences"]
    lines.extend(
        [
            "",
            "## iDPO preference data",
            "",
            f"- Pairs: {pref['pairs']} (Alice {pref['alice_pairs']}, Bob {pref['bob_pairs']})",
            f"- Mean chosen value: {pref['mean_chosen_value']:.4f}",
            f"- Mean reward gap: {pref['mean_reward_gap']:.4f}",
            "",
            "> All three evaluation rows use the same validation split, sample count, seed, decoding temperature, and conversation limits.",
            "",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines))
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()

