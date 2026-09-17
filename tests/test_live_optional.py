"""Optional live smoke test against a real vLLM endpoint (env-gated).

Skipped by default so the offline suite stays dependency-free. Run from the
train environment once a Gemma endpoint is deployed:

    OPTIMA_LIVE_ENDPOINT=http://127.0.0.1:8100/v1/chat/completions \
    OPTIMA_LIVE_SERVED_MODEL=alice \
    python -m pytest tests/test_live_optional.py -v
"""
import os

import pytest

LIVE_ENDPOINT = os.environ.get("OPTIMA_LIVE_ENDPOINT", "")
LIVE_SERVED_MODEL = os.environ.get("OPTIMA_LIVE_SERVED_MODEL", "alice")

pytestmark = pytest.mark.skipif(
    not LIVE_ENDPOINT,
    reason="set OPTIMA_LIVE_ENDPOINT to a deployed vLLM chat/completions URL",
)


def test_one_step_request_against_live_endpoint():
    """One real agent step: the merged Gemma payload must be accepted by the
    served model and the reply must parse back into a turn."""
    from agent.agent import VllmAgent

    agent = VllmAgent(
        url=LIVE_ENDPOINT,
        my_model_name=LIVE_SERVED_MODEL,
        name="Alice",
        temperature=0.3,
        merge_system_into_user=True,
    )
    agent.init_system_prompt(
        'You are ${name} solving the question "${question}". '
        'Information: ${information}. '
        'You must begin your response with "${name}:".',
        {"name": "Alice", "question": "What is 1+1?", "information": "none"},
    )
    turn = agent.step()
    assert turn.speaker == "Alice"
    assert turn.finish_reason == "stop"
    assert "Alice:" in turn.content
