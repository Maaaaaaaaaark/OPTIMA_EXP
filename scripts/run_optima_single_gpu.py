"""Run one-round iSFT, iDPO and iSFT-DPO experiments on one selected GPU."""
from argparse import ArgumentParser
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--task", choices=("information", "debate", "both"), required=True
    )
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--min-free-mib", type=int, default=21000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--stage", choices=("all", "isft", "idpo", "hybrid"), default="all"
    )
    args = parser.parse_args()
    common = [
        "--gpu-id", str(args.gpu_id),
        "--min-free-mib", str(args.min_free_mib),
        "--poll-seconds", str(args.poll_seconds),
        "--eval-count", str(args.eval_count),
    ]
    if args.reset:
        common.append("--reset")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.chdir(PROJECT_ROOT)
    tasks = ("information", "debate") if args.task == "both" else (args.task,)
    for task in tasks:
        stem = "hotpot_qa" if task == "information" else "arc"
        isft_config = f"configs/gemma2-2b/server3090/{stem}_isft.yaml"
        idpo_config = f"configs/gemma2-2b/server3090/{stem}_idpo.yaml"
        hybrid_sft_config = f"configs/gemma2-2b/server3090/{stem}_hybrid_sft.yaml"
        hybrid_dpo_config = f"configs/gemma2-2b/server3090/{stem}_hybrid_dpo.yaml"
        print(f"[optima] starting {task}: stage={args.stage}", flush=True)
        if args.stage in ("all", "isft"):
            subprocess.run(
                [sys.executable, "scripts/run_isft_cycle_single_gpu.py",
                 "--config", isft_config, *common],
                check=True,
                env=env,
            )
        if args.stage in ("all", "idpo"):
            subprocess.run(
                [sys.executable, "scripts/run_idpo_cycle_single_gpu.py",
                 "--isft-config", isft_config, "--idpo-config", idpo_config, *common],
                check=True,
                env=env,
            )
        if args.stage in ("all", "hybrid"):
            baseline_run = (
                "gemma2-2b-hotpotqa-3090-isft-baseline-eval"
                if task == "information"
                else "gemma2-2b-arc-3090-isft-baseline-eval"
            ) + str(args.eval_count)
            subprocess.run(
                [
                    sys.executable, "scripts/run_hybrid_cycle_single_gpu.py",
                    "--sft-config", hybrid_sft_config,
                    "--dpo-config", hybrid_dpo_config,
                    "--baseline-run", baseline_run,
                    *common,
                ],
                check=True,
                env=env,
            )
        if args.stage == "all":
            subprocess.run(
                [sys.executable, "scripts/summarize_all_methods.py", "--task", task],
                check=True,
                env=env,
            )
        print(f"[optima] completed {task}", flush=True)


if __name__ == "__main__":
    main()
