"""Central run configuration for the Qwen-based OPTIMA pipeline.

Replaces the old flat recipe YAMLs (train/sft_recipes/*.yaml) with a single
dataclass-driven config per run. All paths are relative to the project root
(or absolute), no hardcoded author-machine paths.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import os
import yaml


@dataclass
class AgentConfig:
    name: str = "Alice"              # role name used in prompts / transcripts
    served_model_name: str = "alice"  # --served-model-name of its vLLM endpoint
    url: str = ""                     # full ".../v1/chat/completions" endpoint
    initial_model_path: str = ""      # model served at iteration 0
    checkpoint_root: str = ""         # checkpoints/{run}/{name}


@dataclass
class SFTTrainingConfig:
    learning_rate: float = 2.0e-5
    num_train_epochs: float = 4.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 1
    report_to: str = "none"
    # Qwen2.5 chat-template headers; used by DataCollatorForCompletionOnlyLM
    response_template: str = "<|im_start|>assistant\n"
    instruction_template: str = "<|im_start|>user\n"
    train_ratio: float = 0.98
    # SFTTrainer truncates the TAIL, which would cut the last assistant turn;
    # filter templated samples that exceed max_seq_length instead.
    filter_long_samples: bool = True


@dataclass
class RunConfig:
    seed: int = 42
    run_name: str = ""
    dataset_type: str = ""       # "hotpot_qa" | "mwh_qa" | "trival_qa" | "cbt" | "gsm8k" | "math" | "arc" | "mmlu"
    dataset_path: str = ""       # HF hub id or local path ("" -> legacy utils.config default)
    dataset_split: str = "train"
    base_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_path: str = ""     # defaults to base_model_path
    reward_model_path: str = ""  # frozen base model for R_loss; defaults to base_model_path
    alice: AgentConfig = field(default_factory=AgentConfig)
    bob: AgentConfig = field(default_factory=AgentConfig)
    sft: SFTTrainingConfig = field(default_factory=SFTTrainingConfig)
    sample_count: int = 100        # tasks per iteration
    explore_count: int = 8         # trajectories per task
    max_round: int = 10            # max turns per conversation
    max_tokens_per_turn: int = 2000
    iteration_times: int = 3
    episilon: float = 0.6          # reward threshold for SFT data selection
    lambda1: float = -0.5          # token penalty weight (negative)
    lambda2: float = 0.6           # R_loss weight
    cal_ppl: bool = True
    from_initial: bool = True      # debate: restart from base every iteration
    prompt_pool_path: str = ""     # format-diversity pool used at iteration 0
    temperature_iter0: float = 0.3
    temperature: float = 0.7
    selection_trim_low: float = 0.0   # paper: top-70% -> (0.0, 0.7)
    selection_trim_high: float = 0.7
    train_enabled: bool = True
    runs_root: str = "runs"
    thread_count: int = 16
    scorer_batch_size: int = 16
    scorer_device: str = "cuda:0"
    health_check_timeout: float = 120.0

    # ---- derived paths ----
    @property
    def run_dir(self) -> str:
        return os.path.join(self.runs_root, self.run_name)

    def iteration_dir(self, i: int) -> str:
        return os.path.join(self.run_dir, f"iteration_{i}")

    def raw_path(self, i: int) -> str:
        return os.path.join(self.iteration_dir(i), "raw", f"iteration_{i}.jsonl")

    def rewarded_path(self, i: int) -> str:
        return os.path.join(self.iteration_dir(i), "rewarded", f"iteration_{i}.jsonl")

    def cleaned_path(self, i: int) -> str:
        return os.path.join(self.iteration_dir(i), "cleaned", f"iteration_{i}.jsonl")

    def transcripts_dir(self, i: int) -> str:
        return os.path.join(self.iteration_dir(i), "transcripts")

    def alice_dataset_path(self, i: int) -> str:
        return os.path.join(self.iteration_dir(i), "alice_dataset")

    def bob_dataset_path(self, i: int) -> str:
        return os.path.join(self.iteration_dir(i), "bob_dataset")

    def alice_checkpoint_path(self, i: int) -> str:
        return os.path.join(self.alice.checkpoint_root, f"iteration_{i}")

    def bob_checkpoint_path(self, i: int) -> str:
        return os.path.join(self.bob.checkpoint_root, f"iteration_{i}")

    def make_iteration_dirs(self, i: int) -> None:
        for d in (
            self.iteration_dir(i),
            os.path.dirname(self.raw_path(i)),
            os.path.dirname(self.rewarded_path(i)),
            os.path.dirname(self.cleaned_path(i)),
            self.transcripts_dir(i),
            self.alice_dataset_path(i),
            self.bob_dataset_path(i),
        ):
            os.makedirs(d, exist_ok=True)


def load_run_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    alice = AgentConfig(**raw.pop("alice", {}))
    bob = AgentConfig(**raw.pop("bob", {}))
    sft = SFTTrainingConfig(**raw.pop("sft", {}))
    cfg = RunConfig(**raw, alice=alice, bob=bob, sft=sft)
    # defaults
    if not cfg.tokenizer_path:
        cfg.tokenizer_path = cfg.base_model_path
    if not cfg.reward_model_path:
        cfg.reward_model_path = cfg.base_model_path
    for agent in (cfg.alice, cfg.bob):
        if not agent.initial_model_path:
            agent.initial_model_path = cfg.base_model_path
        if not agent.checkpoint_root:
            agent.checkpoint_root = os.path.join("checkpoints", cfg.run_name, agent.name.lower())
    return cfg


def freeze_config(cfg: RunConfig, run_dir: str) -> None:
    """Dump the effective config into the run dir for auditability."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)
