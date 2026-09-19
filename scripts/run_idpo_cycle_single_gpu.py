"""One-command single-GPU iDPO generation, training, evaluation and report."""
from argparse import ArgumentParser
from copy import deepcopy
import os
import shutil
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from scripts.run_isft_cycle_single_gpu import deploy, run, stop_servers, wait_for_gpu
from utils.run_config import load_run_config


def make_eval_config(raw, path, run_name, sample_count, alice_model, bob_model):
    data = deepcopy(raw)
    data["run_name"] = run_name
    data["dataset_split"] = "validation"
    data["sample_count"] = sample_count
    data["explore_count"] = 1
    data["iteration_times"] = 1
    data["temperature_iter0"] = 0.0
    data["temperature"] = 0.0
    data["train_enabled"] = False
    data["thread_count"] = min(2, int(data.get("thread_count", 2)))
    data["alice"]["initial_model_path"] = alice_model
    data["bob"]["initial_model_path"] = bob_model
    data["alice"]["checkpoint_root"] = f"checkpoints/{run_name}/alice"
    data["bob"]["checkpoint_root"] = f"checkpoints/{run_name}/bob"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def require_checkpoint(path, label):
    if not os.path.exists(os.path.join(path, "config.json")):
        raise RuntimeError(f"missing {label} checkpoint: {path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--idpo-config", required=True)
    parser.add_argument("--isft-config", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=21000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg = load_run_config(args.idpo_config)
    isft_cfg = load_run_config(args.isft_config)
    for path, label in (
        (cfg.alice.initial_model_path, "Alice iSFT"),
        (cfg.bob.initial_model_path, "Bob iSFT"),
    ):
        require_checkpoint(path, label)
    free_disk = shutil.disk_usage(PROJECT_ROOT).free // (1024**3)
    if free_disk < 20:
        raise RuntimeError(f"only {free_disk} GiB disk free; require at least 20 GiB")

    python_bin = sys.executable
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    baseline_run = f"{isft_cfg.run_name}-baseline-eval{args.eval_count}"
    post_isft_run = f"{isft_cfg.run_name}-post-isft0-eval{args.eval_count}"
    post_idpo_run = f"{cfg.run_name}-post-idpo0-eval{args.eval_count}"
    generated_dir = os.path.join("configs", "generated")
    os.makedirs(generated_dir, exist_ok=True)
    post_idpo_cfg = os.path.join(generated_dir, f"{post_idpo_run}.yaml")
    with open(args.idpo_config, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    make_eval_config(
        raw,
        post_idpo_cfg,
        post_idpo_run,
        args.eval_count,
        cfg.alice_checkpoint_path(0),
        cfg.bob_checkpoint_path(0),
    )

    wait_for_gpu(args.gpu_id, args.min_free_mib, args.poll_seconds)
    try:
        print("[idpo-orchestrator] phase 1/5: branch generation", flush=True)
        deploy(
            PROJECT_ROOT,
            python_bin,
            args.gpu_id,
            cfg.alice.initial_model_path,
            cfg.bob.initial_model_path,
            env,
        )
        command = [
            python_bin, "idpo_script.py", "--config", args.idpo_config,
            "--iteration", "0", "--stage", "generate",
        ]
        if args.reset:
            command.append("--overwrite")
        run(command, env)

        print("[idpo-orchestrator] phase 2/5: reward and preference pairs", flush=True)
        run(
            [
                python_bin, "idpo_script.py", "--config", args.idpo_config,
                "--iteration", "0", "--stage", "score",
            ],
            env,
        )
        run(
            [python_bin, "scripts/check_idpo.py", "--config", args.idpo_config,
             "--iteration", "0", "--show", "5"],
            env,
        )

        print("[idpo-orchestrator] phase 3/5: independent Alice/Bob DPO training", flush=True)
        run(
            [
                python_bin, "idpo_script.py", "--config", args.idpo_config,
                "--iteration", "0", "--stage", "train",
            ],
            env,
        )
        require_checkpoint(cfg.alice_checkpoint_path(0), "Alice iDPO")
        require_checkpoint(cfg.bob_checkpoint_path(0), "Bob iDPO")

        print("[idpo-orchestrator] phase 4/5: fixed post-iDPO evaluation", flush=True)
        wait_for_gpu(args.gpu_id, args.min_free_mib, args.poll_seconds)
        deploy(
            PROJECT_ROOT,
            python_bin,
            args.gpu_id,
            cfg.alice_checkpoint_path(0),
            cfg.bob_checkpoint_path(0),
            env,
        )
        command = [
            python_bin, "sft_script.py", "--config", post_idpo_cfg,
            "--iteration", "0", "--no_train",
        ]
        if args.reset:
            command.append("--overwrite")
        run(command, env)

        print("[idpo-orchestrator] phase 5/5: three-way report", flush=True)
        run(
            [
                python_bin, "scripts/summarize_idpo_cycle.py",
                "--idpo-config", args.idpo_config,
                "--baseline-run", baseline_run,
                "--post-isft-run", post_isft_run,
                "--post-idpo-run", post_idpo_run,
            ],
            env,
        )
        print("[idpo-orchestrator] ALL IDPO PHASES COMPLETE", flush=True)
    finally:
        stop_servers(PROJECT_ROOT, python_bin, env)


if __name__ == "__main__":
    main()

