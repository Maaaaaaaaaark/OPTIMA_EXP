"""Config-loading tests: the Gemma configs must load through the same
RunConfig machinery as the Qwen configs, with the original author's reward
formula and selection thresholds left untouched."""
import pytest

from utils.run_config import load_run_config
from tests.conftest import GEMMA_CONFIG, GEMMA_SMOKE_CONFIG, QWEN15_CONFIG, QWEN05_CONFIG

yaml = pytest.importorskip("yaml")


def test_gemma_hotpot_config_loads():
    cfg = load_run_config(GEMMA_CONFIG)
    assert cfg.run_name == "gemma2-2b-hotpotqa-isft"
    assert cfg.dataset_type == "hotpot_qa"
    assert cfg.base_model_path == "google/gemma-2-2b-it"
    # Alice, Bob and the frozen reward scorer share the same initial model
    assert cfg.tokenizer_path == "google/gemma-2-2b-it"
    assert cfg.reward_model_path == "google/gemma-2-2b-it"
    assert cfg.alice.initial_model_path == "google/gemma-2-2b-it"
    assert cfg.bob.initial_model_path == "google/gemma-2-2b-it"
    assert cfg.alice.served_model_name == "alice"
    assert cfg.bob.served_model_name == "bob"
    assert cfg.alice.url.endswith(":8100/v1/chat/completions")
    assert cfg.bob.url.endswith(":8101/v1/chat/completions")
    # Gemma chat-template adaptation
    assert cfg.merge_system_into_user is True
    assert cfg.require_name_prefix is True
    assert cfg.sft.response_template == "<start_of_turn>model\n"
    assert cfg.sft.instruction_template == "<start_of_turn>user\n"


def test_gemma_smoke_config_loads_with_conservative_values():
    cfg = load_run_config(GEMMA_SMOKE_CONFIG)
    assert cfg.base_model_path == "google/gemma-2-2b-it"
    assert cfg.merge_system_into_user is True
    assert cfg.sample_count == 10
    assert cfg.explore_count == 2
    assert cfg.max_round == 8
    assert cfg.max_tokens_per_turn == 256
    assert cfg.iteration_times == 1
    assert cfg.thread_count == 2
    assert cfg.scorer_batch_size == 1
    assert cfg.require_name_prefix is True
    assert cfg.train_enabled is False
    assert cfg.sft.bf16 is False


def test_reward_formula_and_selection_unchanged_vs_qwen15():
    gemma = load_run_config(GEMMA_CONFIG)
    qwen = load_run_config(QWEN15_CONFIG)
    # requirement: keep the original reward formula, selection threshold,
    # correctness/token/PPL terms and thresholds identical
    assert gemma.episilon == qwen.episilon
    assert gemma.lambda1 == qwen.lambda1
    assert gemma.lambda2 == qwen.lambda2
    assert gemma.cal_ppl == qwen.cal_ppl
    assert gemma.selection_trim_low == qwen.selection_trim_low
    assert gemma.selection_trim_high == qwen.selection_trim_high
    assert gemma.require_name_prefix == qwen.require_name_prefix
    # same generation budget / conversation protocol
    assert gemma.sample_count == qwen.sample_count
    assert gemma.explore_count == qwen.explore_count
    assert gemma.max_round == qwen.max_round
    assert gemma.max_tokens_per_turn == qwen.max_tokens_per_turn
    assert gemma.iteration_times == qwen.iteration_times
    assert gemma.from_initial == qwen.from_initial
    # no extra reward scorer was introduced: R_loss scorer is the base model
    assert gemma.reward_model_path == gemma.base_model_path


def test_qwen_configs_still_load_unchanged():
    for path in (QWEN15_CONFIG, QWEN05_CONFIG):
        cfg = load_run_config(path)
        assert cfg.base_model_path.startswith("Qwen/")
        # Qwen pipeline must not pick up the Gemma adaptation
        assert cfg.merge_system_into_user is False
        assert cfg.require_name_prefix is True
        assert cfg.sft.response_template == "<|im_start|>assistant\n"
        assert cfg.sft.instruction_template == "<|im_start|>user\n"


def test_frozen_config_roundtrip(tmp_path):
    from utils.run_config import freeze_config

    cfg = load_run_config(GEMMA_SMOKE_CONFIG)
    freeze_config(cfg, str(tmp_path))
    out_path = tmp_path / "config.yaml"
    assert out_path.exists()
    dumped = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert dumped["base_model_path"] == "google/gemma-2-2b-it"
    assert dumped["merge_system_into_user"] is True
    assert dumped["sample_count"] == 10
