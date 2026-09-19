"""iSFT entry point for the Qwen OPTIMA pipeline.

Usage (on the Linux box, vLLM endpoints already deployed via
scripts/deploy_vllm.sh):

    python sft_script.py --config configs/qwen0.5b/hotpot_qa.yaml
    python sft_script.py --config configs/qwen0.5b/arc.yaml --iterations 1 --no_train
"""
from argparse import ArgumentParser
import contextlib
import os
import shutil

from train.sft import run_i_sft, set_seed
from utils.run_config import load_run_config, freeze_config


@contextlib.contextmanager
def exclusive_run_lock(runs_root: str, run_name: str):
    """Prevent two orchestrators from writing the same run concurrently.

    The lock lives outside the run directory so ``--overwrite`` cannot remove
    it while it is held. Linux/Colab releases the advisory lock automatically
    when the process exits, even after an exception.
    """
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - pipeline targets Linux
        raise RuntimeError("exclusive run locking requires Linux fcntl") from exc

    lock_dir = os.path.join(runs_root, ".locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{run_name}.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"run {run_name!r} is already active in another sft_script process"
            ) from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def main():
    argumentParser = ArgumentParser()
    argumentParser.add_argument(
        "--config", type=str, default="configs/qwen0.5b/arc.yaml",
        help="path to the run-config YAML",
    )
    argumentParser.add_argument(
        "--iterations", type=int, default=None,
        help="run only the first N iterations (default: all)",
    )
    argumentParser.add_argument(
        "--iteration", type=int, default=None,
        help="run exactly one zero-based iteration (for sequential GPU orchestration)",
    )
    argumentParser.add_argument(
        "--no_train", action="store_true",
        help="skip SFT training (generation + scoring + datasets only)",
    )
    argumentParser.add_argument(
        "--overwrite", action="store_true",
        help="delete runs/{run_name} and checkpoints/{run_name} before starting",
    )
    args = argumentParser.parse_args()
    if args.iterations is not None and args.iteration is not None:
        argumentParser.error("--iterations and --iteration are mutually exclusive")

    cfg = load_run_config(args.config)
    if not cfg.run_name:
        raise ValueError("run_name must be set in the config")

    with exclusive_run_lock(cfg.runs_root, cfg.run_name):
        if args.overwrite:
            targets = {
                cfg.run_dir,
                cfg.alice.checkpoint_root,
                cfg.bob.checkpoint_root,
            }
            for root in sorted(targets):
                if os.path.exists(root):
                    print(f"[overwrite] removing {root}")
                    shutil.rmtree(root)

        set_seed(cfg.seed)
        freeze_config(cfg, cfg.run_dir)

        iterations = list(range(cfg.iteration_times))
        if args.iteration is not None:
            if args.iteration < 0 or args.iteration >= cfg.iteration_times:
                argumentParser.error("--iteration is outside the configured range")
            iterations = [args.iteration]
        elif args.iterations is not None:
            iterations = iterations[: args.iterations]

        run_i_sft(cfg, iterations=iterations, train=not args.no_train)


if __name__ == "__main__":
    main()
