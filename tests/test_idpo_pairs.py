import pytest
from types import SimpleNamespace

from train.dpo_generate import (
    MCTSNode,
    _author_pairs,
    _backpropagate,
    _normalized_edit_distance,
    _response_edit_distance,
    select_state_pair,
)


def branch(candidate, reward, task_id=0):
    return {
        "candidate": candidate,
        "effective_reward": reward,
        "reward": reward,
        "task_id": task_id,
        "system_first": "alice system",
        "system_second": "bob system",
    }


def test_author_edit_distance_threshold_helpers():
    assert _normalized_edit_distance("Alice: abc", "Alice: abc") == 0.0
    assert _response_edit_distance("Bob: answer", "Bob: answer") == 0.0
    assert _normalized_edit_distance("Alice: abc", "Bob: xyz") > 0.25


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


def test_author_mcts_backpropagates_and_routes_pair_to_acting_agent():
    root = MCTSNode("root")
    alice_good = MCTSNode(
        "good", {"speaker": "Alice", "content": "Alice: <A>x</A>"}, root
    )
    alice_bad = MCTSNode(
        "bad", {"speaker": "Alice", "content": "Alice: unsure"}, root
    )
    root.children = [alice_good, alice_bad]
    _backpropagate(alice_good, 0.9)
    _backpropagate(alice_bad, 0.1)
    cfg = SimpleNamespace(
        dpo=SimpleNamespace(min_value=0.4, min_reward_gap=0.2, pair_keep_ratio=0.5)
    )
    task = SimpleNamespace(task_id=7)
    pairs = _author_pairs(
        cfg,
        task,
        root,
        [root, alice_good, alice_bad],
        {"system_first": "Alice system", "system_second": "Bob system"},
    )
    assert len(pairs) == 1
    assert pairs[0]["speaker"] == "Alice"
    assert pairs[0]["chosen"] == "Alice: <A>x</A>"
    assert pairs[0]["rejected"] == "Alice: unsure"
    assert pairs[0]["distance"] == pytest.approx(0.8)


def test_author_mcts_keeps_reward_ranked_top_half():
    root = MCTSNode("root")
    parent_a = MCTSNode("pa", {"speaker": "Alice", "content": "Alice: a"}, root)
    parent_b = MCTSNode("pb", {"speaker": "Alice", "content": "Alice: b"}, root)
    root.children = [parent_a, parent_b]
    parent_a.value, parent_b.value = 0.9, 0.0
    for index, parent in enumerate((parent_a, parent_b)):
        good = MCTSNode(
            f"g{index}", {"speaker": "Bob", "content": f"Bob: good{index}"}, parent
        )
        bad = MCTSNode(
            f"b{index}", {"speaker": "Bob", "content": f"Bob: bad{index}"}, parent
        )
        good.value, bad.value = 0.8 - index * 0.1, 0.0
        parent.children = [good, bad]
    cfg = SimpleNamespace(
        dpo=SimpleNamespace(min_value=0.4, min_reward_gap=0.2, pair_keep_ratio=0.5)
    )
    pairs = _author_pairs(
        cfg,
        SimpleNamespace(task_id=1),
        root,
        [root, parent_a, parent_b, *parent_a.children, *parent_b.children],
        {"system_first": "a", "system_second": "b"},
    )
    # Three eligible parents -> floor(3 * 0.5) = one highest-value pair.
    assert len(pairs) == 1
    assert pairs[0]["chosen_value"] == pytest.approx(0.9)
