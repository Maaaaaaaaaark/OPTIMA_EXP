"""Write English JSON/CSV/Markdown reports for one author-style hybrid cycle."""
from argparse import ArgumentParser
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.summarize_idpo_cycle import load_pairs, validation_path
from scripts.summarize_isft_cycle import percent, summarize
from utils.run_config import load_run_config


def main():
    parser = ArgumentParser()
    parser.add_argument("--sft-config", required=True)
    parser.add_argument("--dpo-config", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--post-run", required=True)
    args = parser.parse_args()
    sft = load_run_config(args.sft_config)
    dpo = load_run_config(args.dpo_config)
    report = {
        "hybrid_iteration_0_generation": summarize(sft.cleaned_path(0)),
        "hybrid_iteration_1_generation": summarize(sft.cleaned_path(1)),
        "validation_baseline": summarize(validation_path(sft.runs_root, args.baseline_run)),
        "validation_post_hybrid0": summarize(validation_path(sft.runs_root, args.post_run)),
    }
    pairs = load_pairs(dpo.dpo_pairs_path(0))
    report["hybrid_dpo_preferences"] = {
        "pairs": len(pairs),
        "alice_pairs": sum(pair.get("speaker") == "Alice" for pair in pairs),
        "bob_pairs": sum(pair.get("speaker") == "Bob" for pair in pairs),
    }
    out_dir = os.path.join(dpo.run_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "hybrid_cycle_summary.json")
    csv_path = os.path.join(out_dir, "hybrid_cycle_summary.csv")
    md_path = os.path.join(out_dir, "hybrid_cycle_summary.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    stages = (
        ("Generation iteration 0", report["hybrid_iteration_0_generation"]),
        ("Generation iteration 1", report["hybrid_iteration_1_generation"]),
        ("Validation baseline", report["validation_baseline"]),
        ("Validation post-hybrid0", report["validation_post_hybrid0"]),
    )
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "trajectories", "mean_correct", "fully_correct", "mean_tokens", "parsed_answer", "agreement"])
        for label, values in stages:
            writer.writerow([
                label, values["trajectories"], values["mean_correct_score"],
                values["fully_correct"], values["mean_tokens"], values["parsed_answer"],
                values["termination"].get("agreement", 0),
            ])
    lines = [
        f"# {dpo.run_name}: author-style iSFT-DPO cycle",
        "",
        "| Stage | Mean correct | Fully correct | Mean tokens | Parsed | Agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, values in stages:
        total = values["trajectories"]
        lines.append(
            f"| {label} | {values['mean_correct_score']:.4f} | {values['fully_correct']}/{total} | "
            f"{values['mean_tokens']:.2f} | {percent(values['parsed_answer'], total):.1f}% | "
            f"{percent(values['termination'].get('agreement', 0), total):.1f}% |"
        )
    lines.extend([
        "", "## Preference data", "",
        f"- Total pairs: {report['hybrid_dpo_preferences']['pairs']}",
        f"- Alice pairs: {report['hybrid_dpo_preferences']['alice_pairs']}",
        f"- Bob pairs: {report['hybrid_dpo_preferences']['bob_pairs']}",
        "", "> Hybrid iteration 0 performs independent Alice/Bob SFT first, then independent Alice/Bob standard DPO. Standalone iDPO uses RPO instead. This run uses the 50-task one-GPU comparison scale.", "",
    ])
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines))
    print(f"JSON: {json_path}\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
