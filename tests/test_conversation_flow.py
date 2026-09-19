"""End-to-end trajectory generation with a mocked vLLM endpoint.

Covers requirement 9's behavioral checks without a GPU or network:
- <A> answer parsing and the agreement stopping rule still work
- speakers alternate, Alice speaks first, max_round still bounds the loop
- Alice/Bob system prompts and private contexts differ and stay isolated
- every Gemma request is well-formed (no system role, strict alternation)
"""
import pytest

from train.generate import generate_trajectory, TaskSample
from utils.run_config import load_run_config
from tests.conftest import GEMMA_SMOKE_CONFIG

pytest.importorskip("requests")
pytest.importorskip("numpy")


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def make_fake_post(script_alice, script_bob, captured):
    counters = {"8100": 0, "8101": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        key = "8100" if ":8100" in url else "8101"
        script = script_alice if key == "8100" else script_bob
        idx = counters[key]
        counters[key] += 1
        content = script[idx] if idx < len(script) else script[-1]
        captured.append(json)
        return FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": len(content.split())},
            },
        )

    return fake_post


def make_task():
    return TaskSample(
        task_id=0,
        question="Who won the 2020 championship?",
        answer=["Team A", "team a"],
        context_first=["Alice knows: Team A won in 2020."],
        context_second=["Bob knows: the winner was Team A."],
        data_type="qa",
        dataset_name="hotpot_qa",
    )


def test_agreement_stop_and_parser(monkeypatch):
    cfg = load_run_config(GEMMA_SMOKE_CONFIG)
    captured = []
    monkeypatch.setattr(
        "agent.agent.requests.post",
        make_fake_post(
            script_alice=["Alice: clue <A>Team A</A>", "Alice: ok <A>Team A</A>"],
            script_bob=["Bob: disagree <A>Team B</A>", "Bob: hmm <A>Team A</A>"],
            captured=captured,
        ),
    )
    traj = generate_trajectory(cfg, make_task(), traj_id=0, iteration=0)

    assert traj.termination_reason == "agreement"
    assert traj.final_answer == "Team A"
    speakers = [t.speaker for t in traj.turns]
    assert speakers == ["Alice", "Bob", "Alice", "Bob"]  # agreement on Bob's 2nd turn
    assert [t.parsed_answer for t in traj.turns] == ["Team A", "Team B", "Team A", "Team A"]
    assert traj.turns[0].content.startswith("Alice:")


def test_max_round_still_bounds_conversation(monkeypatch):
    cfg = load_run_config(GEMMA_SMOKE_CONFIG)
    cfg.max_round = 4
    monkeypatch.setattr(
        "agent.agent.requests.post",
        make_fake_post(
            script_alice=["Alice: <A>p</A>"] * 8,
            script_bob=["Bob: <A>q</A>"] * 8,
            captured=[],
        ),
    )
    traj = generate_trajectory(cfg, make_task(), traj_id=0, iteration=0)
    assert traj.termination_reason == "max_round"
    assert len(traj.turns) == cfg.max_round
    speakers = [t.speaker for t in traj.turns]
    assert speakers == ["Alice", "Bob"] * (cfg.max_round // 2)


def test_same_speaker_repetition_is_not_agreement(monkeypatch):
    cfg = load_run_config(GEMMA_SMOKE_CONFIG)
    cfg.max_round = 4
    monkeypatch.setattr(
        "agent.agent.requests.post",
        make_fake_post(
            script_alice=["Alice: <A>Team A</A>"] * 4,
            script_bob=["Bob: Correct.", "Bob: Agreed."],
            captured=[],
        ),
    )
    traj = generate_trajectory(cfg, make_task(), traj_id=0, iteration=0)
    assert traj.termination_reason == "max_round"
    assert [turn.speaker for turn in traj.turns] == ["Alice", "Bob", "Alice", "Bob"]


def test_prompts_and_private_contexts_differ(monkeypatch):
    cfg = load_run_config(GEMMA_SMOKE_CONFIG)
    monkeypatch.setattr(
        "agent.agent.requests.post",
        make_fake_post(
            script_alice=["Alice: <A>Team A</A>"] * 8,
            script_bob=["Bob: <A>Team A</A>"] * 8,
            captured=[],
        ),
    )
    traj = generate_trajectory(cfg, make_task(), traj_id=0, iteration=0)
    # system prompts differ (name/partner/context), private contexts isolated
    assert traj.system_first != traj.system_second
    assert "Alice" in traj.system_first and "Bob" in traj.system_first
    assert "Alice knows" in traj.system_first and "Bob knows" not in traj.system_first
    assert "Bob knows" in traj.system_second and "Alice knows" not in traj.system_second
    assert traj.context_first != traj.context_second
    assert traj.question  # the shared question is recorded for both agents


def test_gemma_requests_are_well_formed(monkeypatch):
    cfg = load_run_config(GEMMA_SMOKE_CONFIG)
    captured = []
    monkeypatch.setattr(
        "agent.agent.requests.post",
        make_fake_post(
            script_alice=["Alice: <A>Team A</A>"] * 8,
            script_bob=["Bob: <A>Team A</A>"] * 8,
            captured=captured,
        ),
    )
    generate_trajectory(cfg, make_task(), traj_id=0, iteration=0)
    assert captured  # at least Alice + Bob each spoke once
    for payload in captured:
        msgs = payload["messages"]
        assert all(m["role"] != "system" for m in msgs)
        assert all((m["role"] == "user") == (i % 2 == 0) for i, m in enumerate(msgs))
        # each agent only ever sees its own private context
        first = msgs[0]["content"]
        assert ("Alice knows" in first) != ("Bob knows" in first)
