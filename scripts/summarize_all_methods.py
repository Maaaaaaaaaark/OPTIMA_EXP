"""Build one English local comparison for iSFT, iDPO and iSFT-DPO."""
from argparse import ArgumentParser
import csv
import json
import os


def read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = ArgumentParser()
    parser.add_argument("--task", choices=("information", "debate"), required=True)
    args = parser.parse_args()
    if args.task == "information":
        label, stem = "Information Exchange / HotpotQA", "hotpotqa"
    else:
        label, stem = "Debate / ARC", "arc"
    isft = read(f"runs/gemma2-2b-{stem}-3090-isft/comparison/isft_cycle_summary.json")
    idpo = read(f"runs/gemma2-2b-{stem}-3090-idpo/comparison/idpo_cycle_summary.json")
    hybrid = read(f"runs/gemma2-2b-{stem}-3090-hybrid/comparison/hybrid_cycle_summary.json")
    methods = {
        "iSFT": {
            "iteration_0": isft["isft_iteration_0"],
            "iteration_1": isft["isft_iteration_1"],
            "fixed_validation": isft["validation_post_isft0"],
        },
        "iDPO": {
            "iteration_0": idpo["idpo_iteration_0"],
            "iteration_1": idpo["idpo_iteration_1"],
            "fixed_validation": idpo["validation_post_idpo0"],
        },
        "iSFT-DPO": {
            "iteration_0": hybrid["hybrid_iteration_0_generation"],
            "iteration_1": hybrid["hybrid_iteration_1_generation"],
            "fixed_validation": hybrid["validation_post_hybrid0"],
        },
    }
    report = {
        "experiment": label,
        "validation_baseline": isft["validation_baseline"],
        "methods": methods,
        "note": (
            "Iteration-0 and iteration-1 generation use different training questions; "
            "fixed validation is the causal model comparison. This is a one-GPU, "
            "50-task compute-controlled reproduction of the author's algorithm, not "
            "the original 2,000-13,000-task training scale."
        ),
    }
    out_dir = os.path.join("reports", "information_exchange" if args.task == "information" else "debate")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "all_methods_summary.json")
    csv_path = os.path.join(out_dir, "all_methods_summary.csv")
    md_path = os.path.join(out_dir, "all_methods_summary.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "split", "trajectories", "mean_correct", "fully_correct", "mean_tokens", "parsed_answer", "agreement"])
        base = report["validation_baseline"]
        writer.writerow(["Base", "fixed_validation", base["trajectories"], base["mean_correct_score"], base["fully_correct"], base["mean_tokens"], base["parsed_answer"], base["termination"].get("agreement", 0)])
        for method, splits in methods.items():
            for split, values in splits.items():
                writer.writerow([method, split, values["trajectories"], values["mean_correct_score"], values["fully_correct"], values["mean_tokens"], values["parsed_answer"], values["termination"].get("agreement", 0)])
    lines = [f"# {label}: one-round OPTIMA comparison", "", "## Fixed validation", "", "| Method | Mean correct | Fully correct | Mean tokens |", "|---|---:|---:|---:|"]
    base = report["validation_baseline"]
    lines.append(f"| Base | {base['mean_correct_score']:.4f} | {base['fully_correct']}/{base['trajectories']} | {base['mean_tokens']:.2f} |")
    for method, splits in methods.items():
        value = splits["fixed_validation"]
        lines.append(f"| {method} | {value['mean_correct_score']:.4f} | {value['fully_correct']}/{value['trajectories']} | {value['mean_tokens']:.2f} |")
    lines.extend(["", "## Generation iteration 0 vs iteration 1", "", "| Method | Correct 0 | Correct 1 | Delta | Tokens 0 | Tokens 1 | Delta |", "|---|---:|---:|---:|---:|---:|---:|"])
    for method, splits in methods.items():
        before, after = splits["iteration_0"], splits["iteration_1"]
        lines.append(f"| {method} | {before['mean_correct_score']:.4f} | {after['mean_correct_score']:.4f} | {after['mean_correct_score']-before['mean_correct_score']:+.4f} | {before['mean_tokens']:.2f} | {after['mean_tokens']:.2f} | {after['mean_tokens']-before['mean_tokens']:+.2f} |")
    lines.extend(["", f"> {report['note']}", ""])
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines))
    print(f"JSON: {json_path}\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
