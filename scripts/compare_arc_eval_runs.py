"""Compare fixed ARC validation runs before and after iSFT."""
from argparse import ArgumentParser
from collections import Counter
import json
import statistics


def normalize(value):
    return str(value or "").strip().strip("\\").lower()


def summarize(path):
    results = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                results.extend(json.loads(line).get("results", []))
    tokens = [int(result.get("token_count", 0)) for result in results]
    raw_correct = [
        result
        for result in results
        if normalize(result.get("answer")) == normalize(result.get("final_answer"))
    ]
    qualified = [
        result for result in results if float(result.get("correct_score") or 0) == 1.0
    ]
    return {
        "trajectories": len(results),
        "raw_exact_correct": len(raw_correct),
        "raw_exact_accuracy": len(raw_correct) / len(results) if results else 0.0,
        "reward_qualified_correct": len(qualified),
        "reward_qualified_accuracy": len(qualified) / len(results) if results else 0.0,
        "correct_but_format_penalized": len(raw_correct) - len(qualified),
        "mean_tokens": statistics.mean(tokens) if tokens else 0.0,
        "median_tokens": statistics.median(tokens) if tokens else 0.0,
        "parsed_answer": sum(bool(result.get("final_answer")) for result in results),
        "termination": dict(Counter(result.get("termination_reason") for result in results)),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    before = summarize(args.baseline)
    after = summarize(args.post)
    comparison = {"baseline": before, "post_isft": after}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("ARC Debate: baseline vs post-iSFT (fixed agreement, same 50 tasks)")
    print("=" * 72)
    for key in (
        "raw_exact_accuracy",
        "reward_qualified_accuracy",
        "mean_tokens",
        "median_tokens",
        "parsed_answer",
        "correct_but_format_penalized",
    ):
        left, right = before[key], after[key]
        print(f"{key:32s} {left!s:>12s} -> {right!s:<12s}")
    print(f"termination                      {before['termination']} -> {after['termination']}")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
