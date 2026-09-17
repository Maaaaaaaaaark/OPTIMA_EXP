"""Reward-formula regression tests.

The scoring rules are shared by the Qwen and Gemma pipelines
(reward/scorer.py::score_task); these tests pin the original OPTIMA formula
and its special cases so any change breaks loudly:

    R = R_task - lambda_token * (tokens / max_tokens_in_task)
              + lambda_loss / max_per_turn_loss

Special rules: token_count <= 40 -> token_score = -1; "large" in the
conversation -> token_score = -1; token_count > 2000 or empty conversation ->
ppl_score = -1; duplicated utterance -> ppl_score = 0; name mixing ->
correct_score = 0.
"""
import json
import os

import pytest

from reward.scorer import score_task

pytest.importorskip("numpy")


class StubTokenizer:
    """Enough of a tokenizer for cal_f1_score: whitespace tokenization."""

    pad_token_id = 32000
    eos_token_id = 32001

    def tokenize(self, text):
        return text.lower().strip().split()


class FakeLossScorer:
    """Replaces the frozen model; per-turn losses are scripted."""

    def __init__(self, losses):
        self.tokenizer = StubTokenizer()
        self._losses = list(losses)

    def per_turn_losses(self, texts):
        return self._losses[: len(texts)]


def base_result(**overrides):
    result = {
        "answer": "hello",
        "final_answer": "hello",
        "token_count": 100,
        "conversation": ["Alice: hello", "Bob: hi"],
    }
    result.update(overrides)
    return result


def test_golden_reward_value():
    # correct=1.0, token=-0.6*100/200=-0.3, ppl=1.0/0.5=2.0 -> 2.7
    result = score_task(
        base_result(),
        FakeLossScorer([0.5, 0.5]),
        max_token_count=200,
        lambda1=-0.6,
        lambda2=1.0,
        score_type="f1-score",
        cal_ppl=True,
        check_name_format=True,
    )
    assert result["correct_score"] == 1.0
    assert result["token_score"] == pytest.approx(-0.3)
    assert result["ppl_score"] == pytest.approx(2.0)
    assert result["reward"] == pytest.approx(2.7)
    assert result["reward"] == pytest.approx(
        result["correct_score"] + result["token_score"] + result["ppl_score"]
    )


def test_token_penalty_formula_uses_max_token_in_task():
    result = score_task(
        base_result(token_count=50),
        FakeLossScorer([1.0, 1.0]),
        max_token_count=250,
        lambda1=-0.6,
        lambda2=1.0,
    )
    assert result["token_score"] == pytest.approx(-0.6 * 50 / 250)


def test_short_trajectory_gets_minus_one_token_score():
    result = score_task(
        base_result(token_count=40),
        FakeLossScorer([1.0, 1.0]),
        max_token_count=200,
        lambda1=-0.6,
        lambda2=1.0,
    )
    assert result["token_score"] == -1.0


def test_large_sentinel_gets_minus_one_token_score():
    result = score_task(
        base_result(token_count=500, conversation=["Alice: a lot", "large"]),
        FakeLossScorer([1.0]),
        max_token_count=200,
        lambda1=-0.6,
        lambda2=1.0,
    )
    assert result["token_score"] == -1.0
    assert result["reward"] == pytest.approx(
        result["correct_score"] + result["token_score"] + result["ppl_score"]
    )


def test_overlong_trajectory_gets_minus_one_ppl_score():
    result = score_task(
        base_result(token_count=2001),
        FakeLossScorer([1.0, 1.0]),
        max_token_count=2000,
        lambda1=-0.6,
        lambda2=1.0,
    )
    assert result["ppl_score"] == -1.0


def test_duplicate_utterance_zeroes_ppl_score():
    result = score_task(
        base_result(conversation=["Alice: same thing", "Alice: same thing"]),
        FakeLossScorer([1.0, 1.0]),
        max_token_count=200,
        lambda1=-0.6,
        lambda2=1.0,
    )
    assert result["ppl_score"] == 0.0
    assert result["correct_score"] == 1.0  # only the ppl term is penalized


def test_name_mixing_zeroes_correct_score():
    result = score_task(
        base_result(conversation=["Alice: x Bob: y"]),
        FakeLossScorer([1.0]),
        max_token_count=200,
        lambda1=-0.6,
        lambda2=1.0,
        check_name_format=True,
    )
    assert result["correct_score"] == 0.0


def test_exact_match_list_answer():
    result = score_task(
        base_result(answer=["yes", "no"], final_answer="yes"),
        FakeLossScorer([1.0, 1.0]),
        max_token_count=200,
        lambda1=-0.6,
        lambda2=1.0,
        score_type="exact-match",
    )
    assert result["correct_score"] == 1.0


def test_f1_over_answer_list_takes_max():
    result = score_task(
        base_result(answer=["hello world", "goodbye"], final_answer="hello"),
        FakeLossScorer([1.0, 1.0]),
        max_token_count=200,
        lambda1=-0.6,
        lambda2=1.0,
        score_type="f1-score",
    )
    # f1("hello world", "hello") = 2*(0.5*1)/(1.5) = 2/3; f1("goodbye") = 0
    assert result["correct_score"] == pytest.approx(2.0 / 3.0)


def test_empty_conversation_gets_minus_one_ppl_score():
    result = score_task(
        base_result(conversation=[]),
        FakeLossScorer([]),
        max_token_count=200,
        lambda1=-0.6,
        lambda2=1.0,
    )
    assert result["ppl_score"] == -1.0


# -------------------------------------------------- selection (dataset_build)
def selection_cfg(tmp_path, **overrides):
    from utils.run_config import RunConfig

    cfg = RunConfig(run_name="t", runs_root=str(tmp_path), **overrides)
    return cfg


def write_rewarded(cfg, tasks):
    os.makedirs(os.path.dirname(cfg.rewarded_path(0)), exist_ok=True)
    with open(cfg.rewarded_path(0), "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")


def test_apply_name_penalty_is_minus_ten():
    pytest.importorskip("datasets")
    from train.dataset_build import apply_name_penalty

    clean = {"conversation": ["Alice: ok", "Bob: ok"]}
    apply_name_penalty(clean)
    assert clean["name_penalty"] == 0.0

    mixed = {"conversation": ["Alice: x Bob: y"]}
    apply_name_penalty(mixed)
    assert mixed["name_penalty"] == -10.0

    repeated = {"conversation": ["Alice: a Alice: b"]}
    apply_name_penalty(repeated)
    assert repeated["name_penalty"] == -10.0

    disabled = {"conversation": ["Alice: x Bob: y"]}
    apply_name_penalty(disabled, enabled=False)
    assert disabled["name_penalty"] == 0.0


def test_select_trajectories_argmax_trim_and_episilon(tmp_path):
    pytest.importorskip("datasets")
    from train.dataset_build import select_trajectories

    cfg = selection_cfg(
        tmp_path,
        selection_trim_low=0.0,
        selection_trim_high=0.7,
        episilon=0.5,
        require_name_prefix=True,
    )
    # rewards 9..0 across 10 tasks; per-task argmax is trivial (one result each)
    tasks = [
        {"task_id": i, "results": [{"reward": float(r), "conversation": ["Alice: ok"]}]}
        for i, r in enumerate(range(10, 0, -1))
    ]
    write_rewarded(cfg, tasks)
    selected = select_trajectories(cfg, 0)

    # trim window keeps the top 70% (7 of 10), episilon 0.5 drops nothing here
    assert len(selected) == 7
    rewards = [r["reward"] for r in selected]
    assert rewards == sorted(rewards, reverse=True)  # global sort by reward
    # rank is assigned in descending effective-reward order
    assert [r["rank"] for r in selected] == list(range(7))
    # all rows were written back with selected/rank markers
    for task in tasks:
        assert task["results"][0]["selected"] is (task["results"][0]["reward"] >= 4.0)


def test_select_trajectories_episilon_cutoff(tmp_path):
    pytest.importorskip("datasets")
    from train.dataset_build import select_trajectories

    cfg = selection_cfg(
        tmp_path, selection_trim_low=0.0, selection_trim_high=1.0, episilon=0.5
    )
    tasks = [
        {"task_id": i, "results": [{"reward": r, "conversation": ["Alice: ok"]}]}
        for i, r in enumerate([0.1, 0.2, 0.4, 0.5, 0.6])
    ]
    write_rewarded(cfg, tasks)
    selected = select_trajectories(cfg, 0)
    assert [r["reward"] for r in selected] == [0.6, 0.5]
    assert [r["rank"] for r in selected] == [0, 1]


def test_select_trajectories_name_penalty_flips_argmax(tmp_path):
    pytest.importorskip("datasets")
    from train.dataset_build import select_trajectories

    cfg = selection_cfg(
        tmp_path, selection_trim_low=0.0, selection_trim_high=1.0, episilon=-100.0
    )
    task = {
        "task_id": 0,
        "results": [
            {"reward": 5.0, "conversation": ["Alice: x Bob: y"]},  # mixed -> -10
            {"reward": 1.0, "conversation": ["Alice: ok"]},
        ],
    }
    write_rewarded(cfg, [task])
    selected = select_trajectories(cfg, 0)
    # effective rewards: -5.0 vs 1.0 -> the clean trajectory wins
    assert len(selected) == 1
    assert selected[0]["reward"] == 1.0
    assert task["results"][0]["name_penalty"] == -10.0
