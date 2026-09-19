"""Run one author-style dual-agent iSFT-DPO cycle on one selected GPU."""
from argparse import ArgumentParser
import os
import sys
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from scripts.run_idpo_cycle_single_gpu import make_eval_config, require_checkpoint
from scripts.run_isft_cycle_single_gpu import deploy, run, stop_servers, wait_for_gpu
from utils.run_config import load_run_config


def main():
    parser = ArgumentParser()
    parser.add_argument("--sft-config", required=True)
    parser.add_argument("--dpo-config", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--min-free-mib", type=int, default=21000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    sft = load_run_config(args.sft_config)
    dpo = load_run_config(args.dpo_config)
    python_bin = sys.executable
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    post_run = f"{dpo.run_name}-post-hybrid0-eval{args.eval_count}"
    generated_dir = os.path.join("configs", "generated")
    os.makedirs(generated_dir, exist_ok=True)
    post_cfg = os.path.join(generated_dir, f"{post_run}.yaml")
    with open(args.sft_config, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    make_eval_config(
        raw, post_cfg, post_run, args.eval_count,
        dpo.alice_checkpoint_path(0), dpo.bob_checkpoint_path(0),
    )

    wait_for_gpu(args.gpu_id, args.min_free_mib, args.poll_seconds)
    try:
        print("[hybrid] phase 1/6: iteration-0 iSFT generation and training", flush=True)
        deploy(PROJECT_ROOT, python_bin, args.gpu_id, sft.base_model_path, sft.base_model_path, env)
        command = [python_bin, "sft_script.py", "--config", args.sft_config, "--iteration", "0"]
        if args.reset:
            command.append("--overwrite")
        run(command, env)
        require_checkpoint(sft.alice_checkpoint_path(0), "hybrid Alice SFT")
        require_checkpoint(sft.bob_checkpoint_path(0), "hybrid Bob SFT")

        print("[hybrid] phase 2/6: author MCTS preference generation", flush=True)
        wait_for_gpu(args.gpu_id, args.min_free_mib, args.poll_seconds)
        deploy(PROJECT_ROOT, python_bin, args.gpu_id, sft.alice_checkpoint_path(0), sft.bob_checkpoint_path(0), env)
        command = [python_bin, "idpo_script.py", "--config", args.dpo_config, "--iteration", "0", "--stage", "generate"]
        if args.reset:
            command.append("--overwrite")
        run(command, env)
        run([python_bin, "idpo_script.py", "--config", args.dpo_config, "--iteration", "0", "--stage", "score"], env)

        print("[hybrid] phase 3/6: independent standard-DPO training", flush=True)
        run([python_bin, "idpo_script.py", "--config", args.dpo_config, "--iteration", "0", "--stage", "train"], env)
        require_checkpoint(dpo.alice_checkpoint_path(0), "hybrid Alice DPO")
        require_checkpoint(dpo.bob_checkpoint_path(0), "hybrid Bob DPO")

        print("[hybrid] phase 4/6: iteration-1 generation (no training)", flush=True)
        wait_for_gpu(args.gpu_id, args.min_free_mib, args.poll_seconds)
        deploy(PROJECT_ROOT, python_bin, args.gpu_id, dpo.alice_checkpoint_path(0), dpo.bob_checkpoint_path(0), env)
        run([python_bin, "sft_script.py", "--config", args.sft_config, "--iteration", "1", "--no_train"], env)

        print("[hybrid] phase 5/6: fixed post-hybrid evaluation", flush=True)
        wait_for_gpu(args.gpu_id, args.min_free_mib, args.poll_seconds)
        deploy(PROJECT_ROOT, python_bin, args.gpu_id, dpo.alice_checkpoint_path(0), dpo.bob_checkpoint_path(0), env)
        command = [python_bin, "sft_script.py", "--config", post_cfg, "--iteration", "0", "--no_train"]
        if args.reset:
            command.append("--overwrite")
        run(command, env)

        print("[hybrid] phase 6/6: English local reports", flush=True)
        run([
            python_bin, "scripts/summarize_hybrid_cycle.py",
            "--sft-config", args.sft_config, "--dpo-config", args.dpo_config,
            "--baseline-run", args.baseline_run, "--post-run", post_run,
        ], env)
        print("[hybrid] ALL PHASES COMPLETE", flush=True)
    finally:
        stop_servers(PROJECT_ROOT, python_bin, env)


if __name__ == "__main__":
    main()

