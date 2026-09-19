"""iSFT orchestrator for the Qwen OPTIMA pipeline (model-agnostic).

One iteration = generate (both agents, via external vLLM endpoints) ->
score in-process -> select -> build speaker datasets -> train Alice then
Bob as two fully independent checkpoints.

vLLM is NEVER started or killed from Python: deploy with scripts/deploy_vllm.sh
before running, and re-deploy between iterations when checkpoints change.
"""
import os
import random
import subprocess
import sys
import time
import gc

import numpy as np
import requests

from dataloader.dataloader import (
    DataloaderForHotpotQA,
    DataloaderForMWHQA,
    DataloaderForCBT,
    DataloaderForGSM8K,
    DataloaderForMATH,
    DataloaderForTrivalQA,
    DataloaderForARC,
    DataloaderForMix,
    DataloaderForMMLU,
)
from train.generate import generate_all, write_transcripts
from reward.scorer import score_all
from train.dataset_build import select_trajectories, build_speaker_datasets
from utils.run_config import RunConfig, freeze_config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_dataloader(cfg: RunConfig):
    dataset = cfg.dataset_path
    split = cfg.dataset_split
    if cfg.dataset_type == "hotpot_qa":
        return DataloaderForHotpotQA(dataset=dataset, split=split)
    if cfg.dataset_type == "mwh_qa":
        return DataloaderForMWHQA(dataset_path=dataset, split=split)
    if cfg.dataset_type == "trival_qa":
        return DataloaderForTrivalQA(dataset=dataset, split=split)
    if cfg.dataset_type == "cbt":
        return DataloaderForCBT(dataset=dataset, split=split)
    if cfg.dataset_type == "gsm8k":
        return DataloaderForGSM8K(dataset=dataset, split=split)
    if cfg.dataset_type == "math":
        return DataloaderForMATH(dataset=dataset, split=split)
    if cfg.dataset_type == "arc":
        return DataloaderForARC(dataset=dataset, split=split)
    if cfg.dataset_type == "mmlu":
        return DataloaderForMMLU(dataset=dataset, split=split)
    raise ValueError(f"unknown dataset_type: {cfg.dataset_type}")


def release_vllm_processes() -> None:
    """Stop supported inference servers and descendants, then wait for GPU RAM.

    This is opt-in through ``release_vllm_before_scoring`` and is intended for
    a single-GPU sequential pipeline.  It never targets unrelated Python
    processes.
    """
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError(
            "release_vllm_before_scoring requires psutil"
        ) from exc

    parents = []
    # The Transformers launcher records exactly the two workers owned by this
    # run.  Prefer those PIDs on a shared multi-GPU server so another user's
    # inference process is never selected merely because its command matches.
    pid_dir = os.environ.get(
        "OPTIMA_INFERENCE_PID_DIR", os.path.join("logs", "transformers_pids")
    )
    pid_files = [os.path.join(pid_dir, name) for name in ("alice.pid", "bob.pid")]
    owned_pids = []
    for pid_file in pid_files:
        try:
            with open(pid_file, encoding="utf-8") as handle:
                owned_pids.append(int(handle.read().strip()))
        except (FileNotFoundError, ValueError):
            pass

    if owned_pids:
        for pid in owned_pids:
            try:
                process = psutil.Process(pid)
                command = " ".join(process.cmdline())
                if "scripts/transformers_openai_server.py" in command:
                    parents.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    else:
        # Compatibility fallback for the legacy vLLM launcher, which does not
        # create PID files.  Only processes owned by the current Unix user are
        # eligible on shared machines.
        current_uid = os.getuid() if hasattr(os, "getuid") else None
        for process in psutil.process_iter(["pid", "cmdline", "uids"]):
            try:
                uids = process.info.get("uids")
                if current_uid is not None and uids is not None and uids.real != current_uid:
                    continue
                command = " ".join(process.info.get("cmdline") or [])
                if ("vllm serve" in command or
                        "scripts/transformers_openai_server.py" in command):
                    parents.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    targets = []
    for parent in parents:
        try:
            targets.extend(parent.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        targets.append(parent)

    # Children first avoids leaving CUDA worker processes behind.
    unique = {process.pid: process for process in targets}
    ordered = list(unique.values())
    for process in ordered:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(ordered, timeout=15)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        psutil.wait_procs(alive, timeout=10)

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print(f"[memory] released {len(unique)} inference server/worker processes")


def health_check_endpoint(url: str, model_name: str, timeout: float = 120.0) -> None:
    """One quick request per endpoint; raise on failure instead of the old
    infinite poll loop."""
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4,
                },
                timeout=30,
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                print(f"[health] {url} ready: {content!r}")
                return
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except Exception as e:
            last_error = repr(e)
        time.sleep(5)
    raise RuntimeError(f"endpoint {url} (model {model_name}) not ready: {last_error}")


def run_i_sft(cfg: RunConfig, iterations=None, train: bool = True) -> None:
    """Run the given iterations (default: all)."""
    set_seed(cfg.seed)
    os.makedirs(cfg.run_dir, exist_ok=True)
    freeze_config(cfg, cfg.run_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path or cfg.base_model_path)

    if iterations is None:
        iterations = list(range(cfg.iteration_times))

    for i in iterations:
        print(f"\n{'=' * 70}\n[iteration {i}] start\n{'=' * 70}")
        cfg.make_iteration_dirs(i)

        health_check_endpoint(cfg.alice.url, cfg.alice.served_model_name, cfg.health_check_timeout)
        health_check_endpoint(cfg.bob.url, cfg.bob.served_model_name, cfg.health_check_timeout)

        # fresh dataloader per iteration, advanced past previous slices
        loader = build_dataloader(cfg)
        for _ in range(i * cfg.sample_count):
            loader.sample_once()

        generate_all(cfg, loader, i)
        if cfg.release_vllm_before_scoring:
            release_vllm_processes()
        score_all(cfg, i)
        selected = select_trajectories(cfg, i)
        write_transcripts(cfg, i)
        build_speaker_datasets(cfg, tokenizer, i, selected)

        if cfg.train_enabled and train:
            if cfg.from_initial or i == 0:
                alice_base = cfg.base_model_path
                bob_base = cfg.base_model_path
            else:
                alice_base = cfg.alice_checkpoint_path(i - 1)
                bob_base = cfg.bob_checkpoint_path(i - 1)
            _train_one(cfg, alice_base, cfg.alice_dataset_path(i), cfg.alice_checkpoint_path(i))
            _train_one(cfg, bob_base, cfg.bob_dataset_path(i), cfg.bob_checkpoint_path(i))
            print(
                f"[iteration {i}] trained checkpoints:\n"
                f"  alice: {cfg.alice_checkpoint_path(i)}\n"
                f"  bob:   {cfg.bob_checkpoint_path(i)}\n"
                "NOTE: if another iteration follows, re-deploy vLLM with these "
                "checkpoints first (scripts/deploy_vllm.sh)."
            )
        print(f"[iteration {i}] done")


def _train_one(cfg: RunConfig, model_path: str, dataset_path: str, output_dir: str) -> None:
    import glob

    if glob.glob(os.path.join(output_dir, "*.safetensors")):
        print(f"[train] {output_dir} already exists; skipping training")
        return
    sft = cfg.sft
    cmd = [
        sys.executable, "train/sft_trainer.py",
        "--model_name_or_path", model_path,
        "--dataset_path", dataset_path,
        "--output_dir", output_dir,
        "--seed", str(cfg.seed),
        "--learning_rate", str(sft.learning_rate),
        "--num_train_epochs", str(sft.num_train_epochs),
        "--per_device_train_batch_size", str(sft.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(sft.gradient_accumulation_steps),
        "--max_seq_length", str(sft.max_seq_length),
        "--lr_scheduler_type", sft.lr_scheduler_type,
        "--warmup_ratio", str(sft.warmup_ratio),
        "--logging_steps", str(sft.logging_steps),
        "--report_to", sft.report_to,
        "--response_template", sft.response_template,
        "--instruction_template", sft.instruction_template,
    ]
    if sft.bf16:
        cmd.append("--bf16")
    if sft.fp16:
        cmd.append("--fp16")
    if sft.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if sft.use_lora:
        cmd.extend([
            "--use_lora",
            "--lora_r", str(sft.lora_r),
            "--lora_alpha", str(sft.lora_alpha),
            "--lora_dropout", str(sft.lora_dropout),
            "--lora_target_modules", ",".join(sft.lora_target_modules),
        ])
    print(f"[train] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---- legacy entry points (old sft_script.py / scrips/sft_main.py) ----
def sft_train(*args, **kwargs):
    raise RuntimeError(
        "sft_train is retired: the Llama pipeline was replaced by the Qwen "
        "run-config pipeline. Use `python sft_script.py --config configs/qwen0.5b/*.yaml`."
    )


def sft_train_v2(*args, **kwargs):
    raise RuntimeError(
        "sft_train_v2 is retired: the Llama pipeline was replaced by the Qwen "
        "run-config pipeline. Use `python sft_script.py --config configs/qwen0.5b/*.yaml`."
    )
