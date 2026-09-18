"""Orchestration for independent Alice/Bob iterative DPO."""
import glob
import os
import subprocess
import sys

from train.dpo_generate import (
    build_dpo_datasets,
    generate_dpo_branches,
    score_dpo_branches,
)
from train.sft import (
    build_dataloader,
    health_check_endpoint,
    release_vllm_processes,
    set_seed,
)
from utils.run_config import RunConfig, freeze_config


def _train_one(cfg: RunConfig, model_path: str, dataset_path: str, output_dir: str) -> None:
    if glob.glob(os.path.join(output_dir, "*.safetensors")):
        print(f"[dpo-train] {output_dir} already exists; skipping")
        return
    dpo = cfg.dpo
    cmd = [
        sys.executable,
        "train/dpo_trainer.py",
        "--model_name_or_path",
        model_path,
        "--dataset_path",
        dataset_path,
        "--output_dir",
        output_dir,
        "--seed",
        str(cfg.seed),
        "--learning_rate",
        str(dpo.learning_rate),
        "--num_train_epochs",
        str(dpo.num_train_epochs),
        "--per_device_train_batch_size",
        str(dpo.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(dpo.gradient_accumulation_steps),
        "--max_length",
        str(dpo.max_length),
        "--max_prompt_length",
        str(dpo.max_prompt_length),
        "--beta",
        str(dpo.beta),
        "--lr_scheduler_type",
        dpo.lr_scheduler_type,
        "--warmup_ratio",
        str(dpo.warmup_ratio),
        "--logging_steps",
        str(dpo.logging_steps),
        "--report_to",
        dpo.report_to,
    ]
    if dpo.bf16:
        cmd.append("--bf16")
    if dpo.fp16:
        cmd.append("--fp16")
    if dpo.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if dpo.use_lora:
        cmd.extend(
            [
                "--use_lora",
                "--lora_r",
                str(dpo.lora_r),
                "--lora_alpha",
                str(dpo.lora_alpha),
                "--lora_dropout",
                str(dpo.lora_dropout),
                "--lora_target_modules",
                ",".join(dpo.lora_target_modules),
            ]
        )
    print(f"[dpo-train] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_idpo(cfg: RunConfig, iteration: int, stage: str = "all") -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.run_dir, exist_ok=True)
    os.makedirs(cfg.dpo_iteration_dir(iteration), exist_ok=True)
    freeze_config(cfg, cfg.run_dir)

    if stage in ("all", "generate"):
        health_check_endpoint(
            cfg.alice.url, cfg.alice.served_model_name, cfg.health_check_timeout
        )
        health_check_endpoint(
            cfg.bob.url, cfg.bob.served_model_name, cfg.health_check_timeout
        )
        loader = build_dataloader(cfg)
        for _ in range(iteration * cfg.sample_count):
            loader.sample_once()
        generate_dpo_branches(cfg, loader, iteration)
        if stage == "generate":
            print("[idpo] generation complete; inference servers remain running")
            return

    if stage in ("all", "score"):
        if cfg.release_vllm_before_scoring:
            release_vllm_processes()
        print(f"[idpo-reward] loading frozen model from {cfg.reward_model_path}")
        score_dpo_branches(cfg, iteration)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
        alice_n, bob_n = build_dpo_datasets(cfg, tokenizer, iteration)
        if stage == "score":
            print(f"[idpo] scoring complete: Alice {alice_n}, Bob {bob_n} pairs")
            return

    if stage in ("all", "train"):
        from datasets import load_from_disk

        alice_n = len(load_from_disk(cfg.alice_dpo_dataset_path(iteration))["train"])
        bob_n = len(load_from_disk(cfg.bob_dpo_dataset_path(iteration))["train"])
        if alice_n == 0 or bob_n == 0:
            raise RuntimeError(
                f"cannot train: empty DPO split (Alice={alice_n}, Bob={bob_n}). "
                "Generate more branches/tasks or inspect the reward thresholds."
            )
        if iteration == 0 or cfg.from_initial:
            alice_base = cfg.alice.initial_model_path
            bob_base = cfg.bob.initial_model_path
        else:
            alice_base = cfg.alice_checkpoint_path(iteration - 1)
            bob_base = cfg.bob_checkpoint_path(iteration - 1)
        _train_one(
            cfg,
            alice_base,
            cfg.alice_dpo_dataset_path(iteration),
            cfg.alice_checkpoint_path(iteration),
        )
        _train_one(
            cfg,
            bob_base,
            cfg.bob_dpo_dataset_path(iteration),
            cfg.bob_checkpoint_path(iteration),
        )
        print(
            f"[idpo iteration {iteration}] done\n"
            f"  Alice: {cfg.alice_checkpoint_path(iteration)}\n"
            f"  Bob:   {cfg.bob_checkpoint_path(iteration)}"
        )
