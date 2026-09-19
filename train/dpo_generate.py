"""Dual-agent preference generation for iterative DPO.

The retired implementation launched seven vLLM replicas and mixed both
speakers into one policy.  This module keeps the useful OPTIMA rule -- compare
alternative replies from the *same conversation state* by their downstream
trajectory reward -- while preserving independent Alice and Bob policies.

For every task we create two shallow search states:

* Alice: the initial state (system prompt only).
* Bob: the state after one fixed Alice reply.

Each state is branched ``explore_count`` times and every branch is rolled out
to a terminal conversation.  Scoring happens later, after inference servers
have been released, so two Gemma 2 agents and the frozen reward model never
need to coexist on a 15 GB T4.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from string import Template
from typing import Any, Dict, List, Optional, Tuple
import gc
import json
import os
import random
import threading
import traceback

from datasets import Dataset, DatasetDict
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
            break
        if turn.token_count >= cfg.max_tokens_per_turn:
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
