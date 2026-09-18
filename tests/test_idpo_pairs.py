import pytest

from train.dpo_generate import select_state_pair


def branch(candidate, reward, task_id=0):
    return {
        "candidate": candidate,
        "effective_reward": reward,
        "reward": reward,
        "task_id": task_id,
        "system_first": "alice system",
        "system_second": "bob system",
    }


def test_select_state_pair_uses_same_state_unique_candidates_and_mean_value():
    state = {
        "state_id": "alice_root",
        "speaker": "Alice",
        "prefix_turns": [],
        "branches": [
            branch("Alice: good", 0.8),
            branch("Alice: good", 0.6),
            branch("Alice: bad", 0.1),
            branch("Alice: error", -1.0),
        ],
    }
    pair = select_state_pair(state, min_value=0.4, min_gap=0.2)
    assert pair is not None
    assert pair["chosen"] == "Alice: good"
    assert pair["rejected"] == "Alice: error"
    assert pair["chosen_value"] == pytest.approx(0.7)
    assert pair["distance"] == pytest.approx(1.7)


def test_select_state_pair_rejects_small_gap_or_single_candidate():
    state = {
        "state_id": "bob_after_alice",
        "speaker": "Bob",
        "branches": [branch("Bob: x", 0.7), branch("Bob: y", 0.6)],
    }
    assert select_state_pair(state, min_value=0.4, min_gap=0.2) is None
    state["branches"] = [branch("Bob: x", 0.9), branch("Bob: x", -0.5)]
    assert select_state_pair(state, min_value=0.4, min_gap=0.2) is None
