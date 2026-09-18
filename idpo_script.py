"""Entry point for the dual-agent Gemma/Qwen iDPO pipeline."""
from argparse import ArgumentParser
import os
import shutil

from sft_script import exclusive_run_lock
from train.idpo import run_idpo
from utils.run_config import load_run_config


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument(
        "--stage", choices=("all", "generate", "score", "train"), default="all"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_run_config(args.config)
    if not cfg.run_name:
        raise ValueError("run_name must be set")
    lock_name = f"{cfg.run_name}-idpo"
    with exclusive_run_lock(cfg.runs_root, lock_name):
        if args.overwrite:
            if args.stage not in ("all", "generate"):
                parser.error("--overwrite is only valid with --stage all/generate")
            for target in (cfg.run_dir, cfg.alice.checkpoint_root, cfg.bob.checkpoint_root):
                if os.path.exists(target):
                    print(f"[overwrite] removing {target}")
                    shutil.rmtree(target)
        run_idpo(cfg, args.iteration, args.stage)


if __name__ == "__main__":
    main()
