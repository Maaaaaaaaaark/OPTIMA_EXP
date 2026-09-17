"""Module-import test: the project modules the pipeline depends on (including
the new Gemma touchpoints) must import cleanly in the train environment.
Mirrors step 7 of README_RUN_QWEN's verification checklist.

Skipped when the heavyweight optional deps are missing (e.g. a bare checkout
without torch), but a plain ImportError in a real environment is a failure.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("sympy")


def test_pipeline_modules_import():
    import agent.agent
    import message.message
    import model.llm
    import answerParser.parser
    import utils.config
    import utils.run_config
    import utils.prompt_template
    import utils.utils_token
    import dataloader.dataloader
    import reward.reward
    import reward.deploy_reward
    import reward.scorer
    import train.datagenerate
    import train.generate
    import train.sft
    import train.dataset_build
    import train.sft_trainer  # noqa: F401 (module import, not main())


def test_gemma_adaptation_symbols_exist():
    # the three Gemma adaptation points share one helper + one config flag
    from message.message import adapt_messages_for_chat_template
    from agent.agent import VllmAgent
    from reward.scorer import frame_utterance_for_loss
    from utils.run_config import RunConfig

    assert hasattr(RunConfig, "merge_system_into_user")
    assert callable(adapt_messages_for_chat_template)
    assert callable(frame_utterance_for_loss)
    vllm_init = VllmAgent.__init__.__code__.co_varnames
    assert "merge_system_into_user" in vllm_init
