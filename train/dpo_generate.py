"""Dual-agent preference generation for iterative DPO.

The author-faithful path implements OPTIMA's conversation-tree MCTS: eight
searches per task, three terminal rollouts per expansion, online frozen-reward
backpropagation, top-10 softmax node selection, and same-parent max/min
preference pairs.  Alice and Bob intentionally remain separate policies.

The earlier bounded two-state generator remains available only for backward
compatibility with old configs and tests; server3090 configs use MCTS.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from string import Template
from typing import Any, Dict, List, Optional, Tuple
import gc
import json
import math
import os
import random
import threading
import traceback

from datasets import Dataset, DatasetDict
import numpy as np
import torch

from agent.agent import VllmAgent
from answerParser.parser import hotpot_qa_parser, math_parser
from message.message import Turn, llmMessage, adapt_messages_for_chat_template
from reward.scorer import LossScorer, score_task
from train.dataset_build import apply_name_penalty
from train.generate import (
    TaskSample,
    Trajectory,
    _prepare_prompts,
    rewrite_sorted_jsonl,
    task_rng,
)
from utils.run_config import RunConfig

_write_lock = threading.Lock()
_score_lock = threading.Lock()


def _parse_answer(turn: Turn, data_type: str) -> Optional[str]:
    return math_parser(turn) if data_type == "math" else hotpot_qa_parser(turn)


def _make_agents(
    cfg: RunConfig,
    task: TaskSample,
    first_template: str,
    second_template: str,
    rng: random.Random,
) -> Tuple[VllmAgent, VllmAgent]:
    seed_provider = lambda: rng.randint(0, 2**31 - 1)
    agents = (
        VllmAgent(
            cfg.alice.url,
            cfg.alice.served_model_name,
            "Alice",
            cfg.temperature,
            cfg.max_tokens_per_turn,
            seed_provider,
            cfg.require_name_prefix,
            cfg.merge_system_into_user,
        ),
        VllmAgent(
            cfg.bob.url,
            cfg.bob.served_model_name,
            "Bob",
            cfg.temperature,
            cfg.max_tokens_per_turn,
            seed_provider,
            cfg.require_name_prefix,
            cfg.merge_system_into_user,
        ),
    )
    agents[0].init_system_prompt(
        first_template,
        {
            "name": "Alice",
            "partner": "Bob",
            "question": task.question,
            "information": "\n".join(task.context_first),
        },
    )
    agents[1].init_system_prompt(
        second_template,
        {
            "name": "Bob",
            "partner": "Alice",
            "question": task.question,
            "information": "\n".join(task.context_second),
        },
    )
    return agents


def _replay_prefix(agents: Tuple[VllmAgent, VllmAgent], prefix: List[Dict[str, Any]]) -> None:
    """Restore an exact shared state in each speaker's own role routing."""
    for item in prefix:
        speaker = item["speaker"]
        content = item["content"]
        own = 0 if speaker == "Alice" else 1
        agents[own].add_memory(llmMessage(role="assistant", content=content))
        agents[1 - own].add_memory(llmMessage(role="user", content=content))


def _rollout(
    cfg: RunConfig,
    task: TaskSample,
    state_id: str,
    branch_id: int,
    current: int,
    prefix: List[Dict[str, Any]],
    first_template: str,
    second_template: str,
    seed: str,
    author_mcts: bool = False,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    agents = _make_agents(cfg, task, first_template, second_template, rng)
    _replay_prefix(agents, prefix)

    turns = [Turn(**turn) for turn in prefix]
    conversation = [turn.content for turn in turns]
    final_answer = ""
    final_answer_speaker: Optional[str] = None
    for turn in turns:
        parsed = turn.parsed_answer or _parse_answer(turn, task.data_type)
        if parsed not in (None, ""):
            final_answer = parsed
            final_answer_speaker = turn.speaker

    termination = "max_round"
    candidate = ""
    while len(turns) < cfg.max_round:
        turn = agents[current].step()
        if not candidate:
            candidate = turn.content
        turns.append(turn)
        conversation.append(turn.content)
        if turn.content == "error":
            termination = "error"
            if author_mcts:
                final_answer = "error"
            break
        if turn.token_count >= cfg.max_tokens_per_turn and not author_mcts:
            conversation.append("large")
            termination = "large"
            break
        if task.data_type != "qa" and not turn.content.strip().startswith(turn.speaker):
            turn.content = f"{turn.speaker}:{turn.content}"
            conversation[-1] = turn.content
            if len(turns) == len(prefix) + 1:
                candidate = turn.content
        turn.parsed_answer = _parse_answer(turn, task.data_type)
        if turn.parsed_answer not in (None, ""):
            if (
                turn.parsed_answer == final_answer
                and final_answer_speaker is not None
                and turn.speaker != final_answer_speaker
            ):
                termination = "agreement"
                break
            final_answer = turn.parsed_answer
            final_answer_speaker = turn.speaker
        current = 1 - current
        agents[current].add_memory(llmMessage(role="user", content=turn.content))

    # The author's MCTS explicitly backpropagates an ``error`` answer when a
    # rollout reaches max_depth without cross-speaker answer agreement.
    if author_mcts and termination == "max_round":
        final_answer = "error"

    traj = Trajectory(
        task_id=task.task_id,
        trajectory_id=branch_id,
        question=task.question,
        answer=task.answer,
        context_first=task.context_first,
        context_second=task.context_second,
        data_type=task.data_type,
        dataset_name=task.dataset_name,
        score_type="exact-match" if task.data_type in ("debate", "math") else "f1-score",
        system_first=agents[0].system_prompt.content,
        system_second=agents[1].system_prompt.content,
        turns=turns,
        conversation=conversation,
        final_answer=final_answer,
        token_count=sum(turn.token_count for turn in turns),
        termination_reason=termination,
    ).to_result()
    traj.update(
        {
            "state_id": state_id,
            "branch_id": branch_id,
            "branch_speaker": "Alice" if prefix == [] else "Bob",
            "candidate": candidate,
            "prefix_turns": prefix,
        }
    )
    return traj


@dataclass(eq=False)
class MCTSNode:
    """One action node in the author's conversation-tree search."""

    node_id: str
    turn: Optional[Dict[str, Any]] = None
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    value: float = 0.0
    visits: int = 0

    @property
    def depth(self) -> int:
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth

    def prefix_turns(self) -> List[Dict[str, Any]]:
        turns = []
        node = self
        while node.parent is not None:
            turns.append(node.turn)
            node = node.parent
        return list(reversed(turns))


def _edit_distance(left: str, right: str) -> int:
    """Small dependency-free Levenshtein implementation used by author MCTS."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _normalized_edit_distance(left: str, right: str) -> float:
    return _edit_distance(left, right) / (max(len(left), len(right)) + 1)


def _response_edit_distance(existing: str, response: str) -> float:
    """Author expansion rule normalizes against the new response length."""
    return _edit_distance(existing, response) / (len(response) + 1)


def _insert_rollout(root_node, selected_node, result, node_counter):
    """Merge a terminal rollout into the search tree and return its leaf."""
    node = selected_node
    prefix_len = len(selected_node.prefix_turns())
    for turn in result.get("turns", [])[prefix_len:]:
        content = str(turn.get("content", ""))
        child = next(
            (
                existing
                for existing in node.children
                if _response_edit_distance(
                    str(existing.turn.get("content", "")), content
                ) < 0.10
            ),
            None,
        )
        if child is None:
            child = MCTSNode(
                node_id=f"node_{node_counter[0]}",
                turn=dict(turn),
                parent=node,
            )
            node_counter[0] += 1
            node.children.append(child)
        node = child
    return node


def _backpropagate(node: MCTSNode, value: float) -> None:
    while node is not None:
        node.visits += 1
        node.value += (value - node.value) / node.visits
        node = node.parent


def _select_mcts_node(root, all_nodes, selected_nodes, rng, top_k):
    # This mirrors the author's selectable-node rule: root, or a node whose
    # descendants show that it belongs to an expandable conversation path.
    candidates = [
        node
        for node in all_nodes
        if node is root or (node.children and node.children[0].children)
    ]
    candidates = [
        node
        for node in candidates
        if not any(
            _normalized_edit_distance(
                "" if node.turn is None else str(node.turn.get("content", "")),
                "" if selected.turn is None else str(selected.turn.get("content", "")),
            ) < 0.25
            for selected in selected_nodes
        )
    ]
    if not candidates:
        return root
    candidates.sort(key=lambda node: node.value, reverse=True)
    candidates = candidates[: min(top_k, len(candidates))]
    maximum = max(node.value for node in candidates)
    weights = [math.exp(node.value - maximum) for node in candidates]
    pick = rng.random() * sum(weights)
    cumulative = 0.0
    for node, weight in zip(candidates, weights):
        cumulative += weight
        if cumulative >= pick:
            return node
    return candidates[-1]


def _score_mcts_result(cfg, scorer, result, token_budget):
    with _score_lock:
        score_task(
            result,
            scorer,
            token_budget,
            cfg.lambda1,
            cfg.lambda2,
            result.get("score_type", "f1-score"),
            cfg.cal_ppl,
            cfg.require_name_prefix,
        )
    # Author MCTS backpropagates the joint trajectory reward directly.  The
    # -10 cleaning penalty belongs to the iSFT data-cleaning path, not MCTS.
    result["name_penalty"] = 0.0
    result["effective_reward"] = result["reward"]
    return result["effective_reward"]


def _author_pairs(cfg, task, root, all_nodes, representative):
    pairs = []
    for node in all_nodes:
        if len(node.children) < 2:
            continue
        children = sorted(node.children, key=lambda child: child.value, reverse=True)
        best, worst = children[0], children[-1]
        gap = best.value - worst.value
        if best.value <= cfg.dpo.min_value or gap <= cfg.dpo.min_reward_gap:
            continue
        speaker = "Alice" if node.depth % 2 == 0 else "Bob"
        pairs.append(
            {
                "task_id": task.task_id,
                "state_id": node.node_id,
                "speaker": speaker,
                "system_first": representative.get("system_first", ""),
                "system_second": representative.get("system_second", ""),
                "prefix_turns": node.prefix_turns(),
                "chosen": best.turn["content"],
                "rejected": worst.turn["content"],
                "chosen_value": best.value,
                "rejected_value": worst.value,
                "distance": gap,
            }
        )
    pairs.sort(key=lambda pair: pair["chosen_value"], reverse=True)
    # The author's code retains the reward-ranked top half when at least two
    # usable pairs exist for the task.
    if len(pairs) >= 2:
        keep = max(1, int(cfg.dpo.pair_keep_ratio * len(pairs)))
        pairs = pairs[:keep]
    return pairs


def generate_author_mcts_task(cfg, task, iteration, token_budget, scorer):
    prompt_rng = task_rng(cfg.seed, iteration, task.task_id)
    first_template, second_template = _prepare_prompts(
        cfg, task.data_type, iteration, prompt_rng
    )
    root = MCTSNode("root")
    all_nodes = [root]
    selected_nodes = set()
    node_counter = [0]
    states = []
    all_results = []
    rng = random.Random(f"{cfg.seed}:{iteration}:{task.task_id}:mcts")

    for search_id in range(cfg.dpo.search_iterations):
        selected = _select_mcts_node(
            root, all_nodes, selected_nodes, rng, cfg.dpo.candidate_top_k
        )
        selected_nodes.add(selected)
        prefix = selected.prefix_turns()
        current = len(prefix) % 2
        state_results = []
        for rollout_id in range(cfg.dpo.rollouts_per_expansion):
            result = _rollout(
                cfg,
                task,
                selected.node_id,
                search_id * cfg.dpo.rollouts_per_expansion + rollout_id,
                current,
                prefix,
                first_template,
                second_template,
                f"{cfg.seed}:{iteration}:{task.task_id}:{search_id}:{rollout_id}",
                author_mcts=True,
            )
            value = _score_mcts_result(cfg, scorer, result, token_budget)
            before = set(all_nodes)
            leaf = _insert_rollout(root, selected, result, node_counter)
            cursor = leaf
            while cursor is not None:
                if cursor not in before and cursor not in all_nodes:
                    all_nodes.append(cursor)
                cursor = cursor.parent
            # Include any newly-created intermediate nodes.
            stack = [root]
            all_nodes = []
            while stack:
                cursor = stack.pop()
                all_nodes.append(cursor)
                stack.extend(cursor.children)
            _backpropagate(leaf, value)
            state_results.append(result)
            all_results.append(result)
        states.append(
            {
                "state_id": f"search_{search_id}:{selected.node_id}",
                "speaker": "Alice" if current == 0 else "Bob",
                "prefix_turns": prefix,
                "branches": state_results,
            }
        )

    representative = all_results[0] if all_results else {}
    pairs = _author_pairs(cfg, task, root, all_nodes, representative)
    return {
        "task_id": task.task_id,
        "question": task.question,
        "answer": task.answer,
        "token_budget": token_budget,
        "states": states,
        "results": all_results,
        "tree": {
            "nodes": len(all_nodes),
            "search_iterations": cfg.dpo.search_iterations,
            "rollouts_per_expansion": cfg.dpo.rollouts_per_expansion,
        },
        "pairs": pairs,
    }, pairs


def _probe_token_budget(cfg, dataloader, iteration):
    probe_path = os.path.join(cfg.dpo_iteration_dir(iteration), "token_budget_probe.json")
    if os.path.exists(probe_path):
        with open(probe_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        for probe_id in range(cfg.dpo.token_budget_probe_count):
            dataloader.sample_once(rng=task_rng(cfg.seed, iteration, probe_id))
        return int(stored["token_budget"])

    counts = []
    for probe_id in range(cfg.dpo.token_budget_probe_count):
        rng = task_rng(cfg.seed, iteration, probe_id)
        question, answer, context1, context2 = dataloader.sample_once(rng=rng)
        task = TaskSample(
            probe_id, question, answer, context1, context2,
            dataloader.data_type, dataloader.dataset_name,
        )
        first, second = _prepare_prompts(cfg, task.data_type, iteration, rng)
        result = _rollout(
            cfg, task, "token_probe", 0, 0, [], first, second,
            f"{cfg.seed}:{iteration}:probe:{probe_id}",
        )
        if result.get("termination_reason") not in ("error", "large"):
            counts.append(int(result.get("token_count", 0)))
    if not counts:
        raise RuntimeError("token-budget probe produced no valid trajectories")
    token_budget = max(1, int(np.percentile(counts, cfg.dpo.token_budget_percentile * 100)) + 1)
    with open(probe_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "probe_count": cfg.dpo.token_budget_probe_count,
                "percentile": cfg.dpo.token_budget_percentile,
                "token_budget": token_budget,
                "token_counts": counts,
            },
            handle,
            indent=2,
        )
    print(f"[idpo-mcts] token budget={token_budget} from {len(counts)} probes")
    return token_budget


def generate_author_mcts(cfg, dataloader, iteration):
    """Author-style 8x3 MCTS with online joint-trajectory reward."""
    os.makedirs(cfg.dpo_iteration_dir(iteration), exist_ok=True)
    token_budget = _probe_token_budget(cfg, dataloader, iteration)
    done = set()
    if os.path.exists(cfg.dpo_rewarded_path(iteration)):
        with open(cfg.dpo_rewarded_path(iteration), encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["task_id"])
    if not os.path.exists(cfg.dpo_raw_path(iteration)):
        for path in (cfg.dpo_pairs_path(iteration), cfg.dpo_rewarded_path(iteration)):
            if os.path.exists(path):
                os.remove(path)

    tasks = []
    for task_id in range(cfg.sample_count):
        rng = task_rng(cfg.seed, iteration, task_id)
        question, answer, context1, context2 = dataloader.sample_once(rng=rng)
        if task_id not in done:
            tasks.append(
                TaskSample(
                    task_id, question, answer, context1, context2,
                    dataloader.data_type, dataloader.dataset_name,
                )
            )
    print(f"[idpo-mcts] {len(done)} tasks already done; {len(tasks)} to run")
    scorer = LossScorer(
        cfg.reward_model_path,
        cfg.scorer_device,
        cfg.scorer_batch_size,
        cfg.merge_system_into_user,
    )

    def work(task):
        row, pairs = generate_author_mcts_task(
            cfg, task, iteration, token_budget, scorer
        )
        with _write_lock:
            for path in (cfg.dpo_raw_path(iteration), cfg.dpo_rewarded_path(iteration)):
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[idpo-mcts] task {task.task_id} done: "
            f"{cfg.dpo.search_iterations * cfg.dpo.rollouts_per_expansion} rollouts, "
            f"{len(pairs)} pairs"
        )

    with ThreadPoolExecutor(max_workers=cfg.thread_count) as executor:
        futures = [executor.submit(work, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
    rewrite_sorted_jsonl(cfg.dpo_raw_path(iteration))
    rewrite_sorted_jsonl(cfg.dpo_rewarded_path(iteration))
    # Rebuild the pair file from completed task rows. This makes resume atomic
    # at task granularity and prevents duplicate/missing pairs after a crash.
    with open(cfg.dpo_pairs_path(iteration), "w", encoding="utf-8") as output:
        with open(cfg.dpo_rewarded_path(iteration), encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                for pair in json.loads(line).get("pairs", []):
                    output.write(json.dumps(pair, ensure_ascii=False) + "\n")
    rewrite_sorted_jsonl(cfg.dpo_pairs_path(iteration))
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cfg.sample_count


def generate_dpo_task(cfg: RunConfig, task: TaskSample, iteration: int, output_path: str) -> None:
    prompt_rng = task_rng(cfg.seed, iteration, task.task_id)
    first_template, second_template = _prepare_prompts(
        cfg, task.data_type, iteration, prompt_rng
    )

    alice_branches = [
        _rollout(
            cfg,
            task,
            "alice_root",
            branch,
            0,
            [],
            first_template,
            second_template,
            f"{cfg.seed}:{iteration}:{task.task_id}:alice:{branch}",
        )
        for branch in range(cfg.explore_count)
    ]

    anchor = next(
        (
            result["turns"][0]
            for result in alice_branches
            if result.get("turns") and result["turns"][0].get("content") != "error"
        ),
        None,
    )
    bob_branches: List[Dict[str, Any]] = []
    if anchor is not None:
        # Only the first Alice turn is fixed.  Every Bob candidate therefore
        # has exactly the same policy prompt, which is required by DPO.
        prefix = [anchor]
        bob_branches = [
            _rollout(
                cfg,
                task,
                "bob_after_alice",
                branch,
                1,
                prefix,
                first_template,
                second_template,
                f"{cfg.seed}:{iteration}:{task.task_id}:bob:{branch}",
            )
            for branch in range(cfg.explore_count)
        ]

    row = {
        "task_id": task.task_id,
        "states": [
            {
                "state_id": "alice_root",
                "speaker": "Alice",
                "prefix_turns": [],
                "branches": alice_branches,
            },
            {
                "state_id": "bob_after_alice",
                "speaker": "Bob",
                "prefix_turns": [anchor] if anchor is not None else [],
                "branches": bob_branches,
            },
        ],
    }
    with _write_lock:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[idpo-generate] task {task.task_id} done "
        f"({len(alice_branches)} Alice + {len(bob_branches)} Bob branches)"
    )


def generate_dpo_branches(cfg: RunConfig, dataloader, iteration: int) -> int:
    output_path = cfg.dpo_raw_path(iteration)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    tasks: List[TaskSample] = []
    for task_id in range(cfg.sample_count):
        rng = task_rng(cfg.seed, iteration, task_id)
        question, answer, context1, context2 = dataloader.sample_once(rng=rng)
        if task_id in done:
            continue
        tasks.append(
            TaskSample(
                task_id,
                question,
                answer,
                context1,
                context2,
                dataloader.data_type,
                dataloader.dataset_name,
            )
        )
    print(f"[idpo-generate] {len(done)} tasks already done; {len(tasks)} to run")
    with ThreadPoolExecutor(max_workers=cfg.thread_count) as executor:
        futures = [
            executor.submit(generate_dpo_task, cfg, task, iteration, output_path)
            for task in tasks
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                traceback.print_exc()
                raise
    rewrite_sorted_jsonl(output_path)
    return cfg.sample_count


def score_dpo_branches(cfg: RunConfig, iteration: int) -> int:
    scorer = LossScorer(
        cfg.reward_model_path,
        cfg.scorer_device,
        cfg.scorer_batch_size,
        cfg.merge_system_into_user,
    )
    rows = []
    with open(cfg.dpo_raw_path(iteration), "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    for row in rows:
        all_branches = [
            branch
            for state in row["states"]
            for branch in state.get("branches", [])
        ]
        # Match score_all(): token reward is normalized across all sampled
        # trajectories for the task, not separately for Alice and Bob states.
        max_tokens = max(
            [
                branch.get("token_count", 0)
                for branch in all_branches
                if "large" not in branch.get("conversation", [])
            ]
            or [1]
        )
        for state in row["states"]:
            branches = state["branches"]
            for branch in branches:
                score_task(
                    branch,
                    scorer,
                    max_tokens,
                    cfg.lambda1,
                    cfg.lambda2,
                    branch.get("score_type", "f1-score"),
                    cfg.cal_ppl,
                    cfg.require_name_prefix,
                )
                apply_name_penalty(branch, cfg.require_name_prefix)
                branch["effective_reward"] = branch["reward"] + branch["name_penalty"]
    os.makedirs(os.path.dirname(cfg.dpo_rewarded_path(iteration)), exist_ok=True)
    with open(cfg.dpo_rewarded_path(iteration), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return len(rows)


def select_state_pair(state: Dict[str, Any], min_value: float, min_gap: float) -> Optional[Dict[str, Any]]:
    """Select max/min unique candidate groups using mean rollout reward."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for branch in state.get("branches", []):
        candidate = (branch.get("candidate") or "").strip()
        if not candidate or candidate == "error":
            continue
        groups.setdefault(candidate, []).append(branch)
    if len(groups) < 2:
        return None
    ranked = []
    for candidate, branches in groups.items():
        values = [float(branch.get("effective_reward", branch.get("reward", 0.0))) for branch in branches]
        ranked.append((sum(values) / len(values), candidate, branches[0]))
    ranked.sort(key=lambda item: item[0], reverse=True)
    best, worst = ranked[0], ranked[-1]
    if best[0] <= min_value or best[0] - worst[0] <= min_gap:
        return None
    representative = best[2]
    return {
        "task_id": representative.get("task_id"),
        "state_id": state.get("state_id"),
        "speaker": state.get("speaker"),
        "system_first": representative.get("system_first", ""),
        "system_second": representative.get("system_second", ""),
        "prefix_turns": state.get("prefix_turns", []),
        "chosen": best[1],
        "rejected": worst[1],
        "chosen_value": best[0],
        "rejected_value": worst[0],
        "distance": best[0] - worst[0],
    }


def _format_pair(tokenizer, cfg: RunConfig, pair: Dict[str, Any]) -> Dict[str, Any]:
    speaker = pair["speaker"]
    system = pair["system_first"] if speaker == "Alice" else pair["system_second"]
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    for turn in pair.get("prefix_turns", []):
        role = "assistant" if turn.get("speaker") == speaker else "user"
        messages.append({"role": role, "content": turn.get("content", "")})
    messages = adapt_messages_for_chat_template(messages, cfg.merge_system_into_user)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    def completion(text: str) -> str:
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": text}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return full[len(prompt):] if full.startswith(prompt) else text

    return {
        "prompt": prompt,
        "chosen": completion(pair["chosen"]),
        "rejected": completion(pair["rejected"]),
        "task_id": pair["task_id"],
        "state_id": pair["state_id"],
        "chosen_value": pair["chosen_value"],
        "rejected_value": pair["rejected_value"],
        "distance": pair["distance"],
    }


def build_dpo_datasets(cfg: RunConfig, tokenizer, iteration: int) -> Tuple[int, int]:
    raw_pairs: List[Dict[str, Any]] = []
    speaker_pairs: Dict[str, List[Dict[str, Any]]] = {"Alice": [], "Bob": []}
    if cfg.dpo.author_mcts:
        # MCTS already selected same-parent max/min children, applied the
        # author thresholds, and retained the reward-ranked top half.
        with open(cfg.dpo_pairs_path(iteration), "r", encoding="utf-8") as handle:
            raw_pairs = [json.loads(line) for line in handle if line.strip()]
        for pair in raw_pairs:
            speaker_pairs[pair["speaker"]].append(_format_pair(tokenizer, cfg, pair))
    else:
        with open(cfg.dpo_rewarded_path(iteration), "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for state in row.get("states", []):
                    pair = select_state_pair(
                        state, cfg.dpo.min_value, cfg.dpo.min_reward_gap
                    )
                    if pair is not None:
                        raw_pairs.append(pair)
                        speaker_pairs[pair["speaker"]].append(_format_pair(tokenizer, cfg, pair))

        with open(cfg.dpo_pairs_path(iteration), "w", encoding="utf-8") as handle:
            for pair in raw_pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    def save(speaker: str, path: str) -> int:
        rows = speaker_pairs[speaker]
        dataset = Dataset.from_list(rows) if rows else Dataset.from_dict(
            {"prompt": [], "chosen": [], "rejected": []}
        )
        n = len(dataset)
        split = max(1, int(cfg.dpo.train_ratio * n)) if n else 0
        split = min(split, n)
        DatasetDict(
            {
                "train": dataset.select(range(split)),
                "test": dataset.select(range(split, n)),
            }
        ).save_to_disk(path)
        return n

    alice_n = save("Alice", cfg.alice_dpo_dataset_path(iteration))
    bob_n = save("Bob", cfg.bob_dpo_dataset_path(iteration))
    print(
        f"[idpo-pairs] Alice {alice_n}, Bob {bob_n}; "
        f"min_value>{cfg.dpo.min_value}, gap>{cfg.dpo.min_reward_gap}"
    )
    return alice_n, bob_n
