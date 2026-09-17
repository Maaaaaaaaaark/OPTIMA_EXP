"""VllmAgent request-shaping tests for Gemma (no network, requests mocked).

Verifies that with merge_system_into_user=True the outgoing OpenAI-compatible
payload has no "system" role, alternates user/assistant starting with user
(Gemma's template constraint), and that the Qwen path (flag off) is untouched.
"""
import pytest

from agent.agent import VllmAgent
from message.message import llmMessage

pytest.importorskip("requests")


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def make_ok_response(content):
    return FakeResponse(
        200,
        {
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"completion_tokens": len(content.split())},
        },
    )


def alternates_from_user(messages):
    return all(
        (m["role"] == "user") == (i % 2 == 0) for i, m in enumerate(messages)
    )


def test_alice_first_request_merges_system(monkeypatch):
    payloads = []
    monkeypatch.setattr(
        "agent.agent.requests.post",
        lambda url, headers=None, json=None, timeout=None: payloads.append(json)
        or make_ok_response("Alice: hi <A>X</A>"),
    )
    agent = VllmAgent(
        url="http://127.0.0.1:8100/v1/chat/completions",
        my_model_name="alice",
        name="Alice",
        temperature=0.3,
        merge_system_into_user=True,
    )
    # "You must begin your response with" marks iteration 0 -> no prefill
    template = (
        "You are ${name}... information: ${information}. "
        'You must begin your response with "${name}:".'
    )
    agent.init_system_prompt(template, {"name": "Alice", "information": "ctx_a"})
    sys = agent.system_prompt.content
    turn = agent.step()
    assert turn.speaker == "Alice"
    assert turn.content == "Alice: hi <A>X</A>"
    sent = payloads[0]
    assert "messages" in sent
    msgs = sent["messages"]
    assert all(m["role"] != "system" for m in msgs)
    assert alternates_from_user(msgs)
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == sys
    assert agent.memory[-1].content == turn.content  # memory keeps the raw turn


def test_bob_first_request_merges_system_with_partner_turn(monkeypatch):
    payloads = []
    monkeypatch.setattr(
        "agent.agent.requests.post",
        lambda url, headers=None, json=None, timeout=None: payloads.append(json)
        or make_ok_response("Bob: agree <A>X</A>"),
    )
    agent = VllmAgent(
        url="http://127.0.0.1:8101/v1/chat/completions",
        my_model_name="bob",
        name="Bob",
        temperature=0.3,
        merge_system_into_user=True,
    )
    template = (
        "You are ${name}... information: ${information}. "
        'You must begin your response with "${name}:".'
    )
    agent.init_system_prompt(template, {"name": "Bob", "information": "ctx_b"})
    sys = agent.system_prompt.content
    agent.add_memory(llmMessage(role="user", content="Alice: clue <A>X</A>"))
    agent.step()
    msgs = payloads[0]["messages"]
    assert alternates_from_user(msgs)
    assert msgs[0]["content"] == sys + "\n\n" + "Alice: clue <A>X</A>"
    assert "ctx_b" in msgs[0]["content"] and "ctx_a" not in msgs[0]["content"]


def test_qwen_path_unchanged_without_flag(monkeypatch):
    payloads = []
    monkeypatch.setattr(
        "agent.agent.requests.post",
        lambda url, headers=None, json=None, timeout=None: payloads.append(json)
        or make_ok_response("hi"),
    )
    agent = VllmAgent(
        url="http://127.0.0.1:8100/v1/chat/completions",
        my_model_name="alice",
        name="Alice",
        temperature=0.3,
    )
    agent.init_system_prompt("You are ${name}.", {"name": "Alice"})
    agent.step()
    msgs = payloads[0]["messages"]
    assert msgs[0]["role"] == "system"  # untouched Qwen behavior


def test_prefill_path_keeps_alternation_after_merge(monkeypatch):
    payloads = []
    monkeypatch.setattr(
        "agent.agent.requests.post",
        lambda url, headers=None, json=None, timeout=None: payloads.append(json)
        or make_ok_response(" sure, <A>Y</A>"),
    )
    agent = VllmAgent(
        url="http://127.0.0.1:8100/v1/chat/completions",
        my_model_name="alice",
        name="Alice",
        temperature=0.7,
        merge_system_into_user=True,
        use_name_prefix=True,
    )
    # no iteration-0 marker -> the name prefix is prefilled as assistant text
    agent.init_system_prompt("You are ${name} (plain, no prefix rule).", {"name": "Alice"})
    agent.add_memory(llmMessage(role="user", content="Bob: anything"))
    turn = agent.step()
    sent = payloads[0]
    assert sent["continue_final_message"] is True
    assert sent["add_generation_prompt"] is False
    msgs = sent["messages"]
    assert alternates_from_user(msgs)
    assert msgs[-1] == {"role": "assistant", "content": "Alice:"}
    assert turn.content == "Alice: sure, <A>Y</A>"  # prefix restored


def test_400_still_yields_error_turn(monkeypatch):
    monkeypatch.setattr(
        "agent.agent.requests.post",
        lambda url, headers=None, json=None, timeout=None: FakeResponse(400, {"error": "bad"}),
    )
    agent = VllmAgent(
        url="http://127.0.0.1:8100/v1/chat/completions",
        my_model_name="alice",
        name="Alice",
        temperature=0.3,
        merge_system_into_user=True,
    )
    agent.init_system_prompt("You are ${name}.", {"name": "Alice"})
    turn = agent.step()
    assert turn.content == "error"
    assert turn.finish_reason == "error"
    assert len(agent.memory) == 1  # error turns are not added to memory
