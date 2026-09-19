"""One-command single-GPU iSFT cycle for Gemma 2 2B.

Sequence: fixed baseline eval -> iteration-0 generation/score/train -> deploy
the two independent checkpoints -> iteration-1 generation/score (no train) ->
fixed post-iSFT eval -> automatic comparison report.
"""
from argparse import ArgumentParser
from copy import deepcopy
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from utils.run_config import load_run_config


def gpu_free_mib(gpu_id):
    output = subprocess.check_output(
        [
            "nvidia-smi", "-i", str(gpu_id),
            "--query-gpu=memory.free", "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def wait_for_gpu(gpu_id, minimum, poll_seconds):
    while True:
        free = gpu_free_mib(gpu_id)
        print(f"[orchestrator] GPU {gpu_id}: {free} MiB free (need {minimum})", flush=True)
        if free >= minimum:
            return
        time.sleep(poll_seconds)


def wait_endpoint(url, timeout=600):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last = exc
        time.sleep(5)
    raise RuntimeError(f"endpoint not ready: {url}: {last}")


def run(command, env):
    print("[orchestrator] $ " + " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def deploy(project_root, python_bin, gpu_id, alice_model, bob_model, env):
    deploy_script = os.path.join(project_root, "scripts", "deploy_transformers.sh")
    deploy_env = dict(env)
    deploy_env.update(
        {
            "START_BOTH": "1",
            "PYTHON_BIN": python_bin,
            "GPU_ID": str(gpu_id),
            "DTYPE": "half",
        }
    )
    run([deploy_script, alice_model, bob_model], deploy_env)
    wait_endpoint("http://127.0.0.1:8100/v1/models")
    wait_endpoint("http://127.0.0.1:8101/v1/models")
    print(f"[orchestrator] deployed Alice={alice_model} Bob={bob_model}", flush=True)


def stop_servers(project_root, python_bin, env):
    deploy_env = dict(env)
    deploy_env["PYTHON_BIN"] = python_bin
    subprocess.run(
        [os.path.join(project_root, "scripts", "deploy_transformers.sh"), "stop"],
        env=deploy_env,
        check=False,
    )


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


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=21000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    os.chdir(project_root)
    cfg = load_run_config(args.config)
    if cfg.iteration_times < 2:
        raise ValueError("config iteration_times must be at least 2")
    python_bin = sys.executable
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    free_disk = shutil.disk_usage(project_root).free // (1024**3)
    if free_disk < 30:
        raise RuntimeError(f"only {free_disk} GiB disk free; require at least 30 GiB")

    with open(args.config, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    generated_dir = os.path.join("configs", "generated")
    os.makedirs(generated_dir, exist_ok=True)
    baseline_run = f"{cfg.run_name}-baseline-eval{args.eval_count}"
    post_run = f"{cfg.run_name}-post-isft0-eval{args.eval_count}"
    baseline_cfg = os.path.join(generated_dir, f"{baseline_run}.yaml")
    post_cfg = os.path.join(generated_dir, f"{post_run}.yaml")
    make_eval_config(raw, baseline_cfg, baseline_run, args.eval_count, cfg.base_model_path, cfg.base_model_path)
    make_eval_config(
        raw, post_cfg, post_run, args.eval_count,
        cfg.alice_checkpoint_path(0), cfg.bob_checkpoint_path(0),
    )

    wait_for_gpu(args.gpu_id, args.min_free_mib, args.poll_seconds)
    try:
        print("[orchestrator] phase 1/5: fixed baseline evaluation", flush=True)
        deploy(project_root, python_bin, args.gpu_id, cfg.base_model_path, cfg.base_model_path, env)
        command = [python_bin, "sft_script.py", "--config", baseline_cfg, "--iteration", "0", "--no_train"]
        if args.reset:
            command.append("--overwrite")
        run(command, env)

        print("[orchestrator] phase 2/5: iSFT iteration 0 generation + training", flush=True)
        wait_for_gpu(args.gpu_id, args.min_free_mib, args.poll_seconds)
        deploy(project_root, python_bin, args.gpu_id, cfg.base_model_path, cfg.base_model_path, env)
        command = [python_bin, "sft_script.py", "--config", args.config, "--iteration", "0"]
        if args.reset:
            command.append("--overwrite")
        run(command, env)

        for path in (cfg.alice_checkpoint_path(0), cfg.bob_checkpoint_path(0)):
            if not os.path.exists(os.path.join(path, "config.json")):
                raise RuntimeError(f"missing trained checkpoint: {path}")

        print("[orchestrator] phase 3/5: iSFT iteration 1 generation + scoring", flush=True)
        wait_for_gpu(args.gpu_id, args.min_free_mib, args.poll_seconds)
        deploy(
            project_root, python_bin, args.gpu_id,
            cfg.alice_checkpoint_path(0), cfg.bob_checkpoint_path(0), env,
        )
        run(
            [python_bin, "sft_script.py", "--config", args.config, "--iteration", "1", "--no_train"],
            env,
        )

        print("[orchestrator] phase 4/5: fixed post-iSFT evaluation", flush=True)
        wait_for_gpu(args.gpu_id, args.min_free_mib, args.poll_seconds)
        deploy(
            project_root, python_bin, args.gpu_id,
            cfg.alice_checkpoint_path(0), cfg.bob_checkpoint_path(0), env,
        )
        command = [python_bin, "sft_script.py", "--config", post_cfg, "--iteration", "0", "--no_train"]
        if args.reset:
            command.append("--overwrite")
        run(command, env)

        print("[orchestrator] phase 5/5: reports", flush=True)
        run(
            [
                python_bin, "scripts/summarize_isft_cycle.py",
                "--config", args.config,
                "--baseline-run", baseline_run,
                "--post-run", post_run,
            ],
            env,
        )
        print("[orchestrator] ALL ISFT PHASES COMPLETE", flush=True)
    finally:
        stop_servers(project_root, python_bin, env)


if __name__ == "__main__":
    main()
