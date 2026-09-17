"""Iteration quality checker for the Qwen OPTIMA iSFT pipeline.

Run this after a ``--no_train`` iteration to verify the six acceptance
criteria automatically:

  1. counts          -- 100 tasks x 8 trajectories = 800 transcripts + index
  2. independence    -- alternating speakers, distinct system prompts,
                        distinct private context, no shared memory
  3. format          -- name prefixes, <A>...</A> answers, no name mixing,
                        no duplication / blanks / mojibake / truncation
  4. termination     -- agreement / max_round / large / error breakdown
  5. reward+select   -- score fields present, ranks monotone, selected
                        trajectories actually better than unselected
  6. datasets        -- alice_dataset / bob_dataset non-empty and correctly
                        role-routed (own turns = assistant, partner = user)

Usage (from the project root, in the ``train_env`` environment):

    python scripts/check_iteration.py --config configs/qwen0.5b/hotpot_qa.yaml
    python scripts/check_iteration.py --config configs/qwen0.5b/arc.yaml --show 5
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.run_config import load_run_config  # noqa: E402


# ------------------------------------------------------------------ reporting
class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, title, detail=""):
        self.rows.append((status, title, detail))

    def ok(self, title, detail=""):
        self.add("PASS", title, detail)

    def fail(self, title, detail=""):
        self.add("FAIL", title, detail)

    def warn(self, title, detail=""):
        self.add("WARN", title, detail)

    def info(self, title, detail=""):
        self.add("INFO", title, detail)

    def counts(self):
        c = Counter(s for s, _, _ in self.rows)
        return c.get("FAIL", 0), c.get("WARN", 0)

    def dump(self):
        icons = {
            "PASS": "[PASS]", "FAIL": "[FAIL]",
            "WARN": "[WARN]", "INFO": "[INFO]",
        }
        for status, title, detail in self.rows:
            print(f"{icons[status]} {title}")
            if detail:
                for line in str(detail).splitlines():
                    print(f"        {line}")


# -------------------------------------------------------------------- helpers
def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def flatten(results_rows):
    """[(task_id, result_dict)] over every trajectory in the file."""
    out = []
    for row in results_rows:
        for res in row.get("results", []):
            out.append((row.get("task_id", -1), res))
    return out


def split_chat(text):
    """Parse a templated Qwen chat string into [(role, content), ...]."""
    parts = re.split(r"<\|im_start\|>(\w+)\n", text)
    out = []
    for i in range(1, len(parts) - 1, 2):
        role = parts[i]
        content = parts[i + 1].replace("<|im_end|>", "").rstrip("\n")
        out.append((role, content))
    return out


def has_name_prefix(content, speaker):
    return re.match(rf"^\s*\**\s*{speaker}\s*:", content or "") is not None


def strip_name_prefix(content):
    return re.sub(r"^\s*\**\s*(Alice|Bob)\s*:\s*", "", content or "").strip()


def partner_of(speaker):
    return "Bob" if speaker == "Alice" else "Alice"


# ------------------------------------------------------------------- checkers
def check_counts(cfg, iteration, raw, rep):
    tasks = raw
    n_tasks = len(tasks)
    per_task = [len(t.get("results", [])) for t in tasks]
    n_traj = sum(per_task)

    if n_tasks == cfg.sample_count:
        rep.ok(f"任务数 = {n_tasks}")
    else:
        rep.fail(f"任务数 = {n_tasks}，期望 {cfg.sample_count}")

    bad = [(t.get("task_id"), len(t.get("results", [])))
           for t in tasks if len(t.get("results", [])) != cfg.explore_count]
    if not bad:
        rep.ok(f"每题轨迹数 = {cfg.explore_count}（共 {n_traj} 条对话）")
    else:
        rep.fail(
            f"{len(bad)} 个任务的轨迹数 != {cfg.explore_count}",
            "前 10 个: " + ", ".join(f"task {t}: {n} 条" for t, n in bad[:10]),
        )

    # ids must be contiguous 0..N-1 with no duplicates
    ids = [t.get("task_id") for t in tasks]
    if sorted(ids) == list(range(cfg.sample_count)):
        rep.ok("task_id 连续且无重复 (0..%d)" % (cfg.sample_count - 1))
    else:
        rep.fail("task_id 不连续或有重复", f"实得 {len(set(ids))} 个唯一 id")

    traj_dir = cfg.transcripts_dir(iteration)
    files = sorted(os.listdir(traj_dir)) if os.path.isdir(traj_dir) else []
    traj_files = [f for f in files if f.startswith("task_") and f.endswith(".txt")]
    if len(traj_files) == n_traj and n_traj > 0:
        rep.ok(f"transcript 文件数 = {len(traj_files)}")
    else:
        rep.fail(f"transcript 文件数 = {len(traj_files)}，期望 {n_traj}")

    index = os.path.join(traj_dir, "all_transcripts.txt")
    if os.path.exists(index) and os.path.getsize(index) > 0:
        rep.ok(f"all_transcripts.txt 存在（{os.path.getsize(index) / 1e6:.1f} MB）")
    else:
        rep.fail("all_transcripts.txt 缺失或为空")


def check_independence(cfg, flat, rep):
    same_system = 0
    same_context = 0
    non_alternating = 0
    wrong_first = 0
    empty_context = 0
    for task_id, res in flat:
        if res.get("system_first") and res["system_first"] == res.get("system_second"):
            same_system += 1
        cf, cs = res.get("context_first") or [], res.get("context_second") or []
        if cf and cf == cs:
            same_context += 1
        if not cf and not cs:
            empty_context += 1
        turns = res.get("turns", [])
        speakers = [t.get("speaker") for t in turns]
        expected = ["Alice" if i % 2 == 0 else "Bob" for i in range(len(speakers))]
        if speakers != expected:
            non_alternating += 1
        if speakers and speakers[0] != "Alice":
            wrong_first += 1

    n = len(flat)
    if same_system == 0:
        rep.ok("两个 agent 的 system prompt 全部不同")
    else:
        rep.fail(f"{same_system}/{n} 条轨迹的 system prompt 完全相同（未独立）")

    if same_context == 0:
        rep.ok("两边的私有 context 全部不同")
    else:
        rep.fail(f"{same_context}/{n} 条轨迹两边拿到相同 context")

    if empty_context == n and n > 0:
        rep.info("两边 context 均为空（debate 任务的预期情况，双方共享题目）")
    elif empty_context:
        rep.info(f"{empty_context}/{n} 条轨迹两边 context 均为空")

    if non_alternating == 0:
        rep.ok("发言严格 Alice/Bob 交替，无共享 memory 迹象")
    else:
        rep.fail(f"{non_alternating}/{n} 条轨迹发言未严格交替（角色路由可能有问题）")

    if wrong_first == 0:
        rep.ok("每条轨迹都由 Alice 先发言")
    else:
        rep.warn(f"{wrong_first}/{n} 条轨迹不是 Alice 先发言")


def check_format(cfg, flat, rep):
    n = len(flat)
    if n == 0:
        rep.fail("没有轨迹可检查")
        return
    pref_ok = pref_bad = 0
    no_answer = 0
    name_mixed = 0
    dup_traj = 0
    blank_turns = 0
    mojibake = 0
    truncated = 0
    length_finish = 0
    empty_reply = 0
    total_turns = 0

    for task_id, res in flat:
        turns = res.get("turns", [])
        total_turns += len(turns)
        for t in turns:
            content = t.get("content") or ""
            speaker = t.get("speaker") or ""
            if has_name_prefix(content, speaker):
                pref_ok += 1
            else:
                pref_bad += 1
            if not content.strip():
                blank_turns += 1
            if "�" in content or any(
                ord(ch) < 9 for ch in content
            ):
                mojibake += 1
            if t.get("finish_reason") == "length":
                length_finish += 1

        if not res.get("final_answer"):
            no_answer += 1

        conv = res.get("conversation", [])
        for sentence in conv:
            text = str(sentence)
            if re.search(r"Alice:", text) and re.search(r"Bob:", text):
                name_mixed += 1
                break
            if len(re.findall(r"Alice:", text)) >= 2 or len(re.findall(r"Bob:", text)) >= 2:
                name_mixed += 1
                break

        seen = set()
        for sentence in conv:
            key = strip_name_prefix(str(sentence)).lower()
            if key and key in seen:
                dup_traj += 1
                break
            seen.add(key)

        if res.get("termination_reason") == "large":
            truncated += 1
        if res.get("termination_reason") == "error":
            empty_reply += 1

    pref_rate = pref_ok / total_turns if total_turns else 0.0
    if total_turns == 0:
        rep.fail("轨迹中没有任何发言")
    elif not cfg.require_name_prefix:
        rep.ok(
            "已关闭文本名字前缀要求；speaker 元数据负责角色归属",
            f"自然产生名字前缀: {pref_ok}/{total_turns} ({pref_rate:.1%})，不参与验收",
        )
    elif pref_rate >= 0.99:
        rep.ok(f"发言带名字前缀: {pref_ok}/{total_turns} ({pref_rate:.1%})")
    elif pref_rate >= 0.9:
        rep.warn(
            f"发言带名字前缀: {pref_ok}/{total_turns} ({pref_rate:.1%})，有 {pref_bad} 条缺前缀",
            "缺前缀会让重复检测与 name_penalty 失效，建议检查 prompt 或加大 prefill 覆盖",
        )
    else:
        rep.fail(
            f"仅 {pref_rate:.1%} 的发言带名字前缀（{pref_bad} 条缺失）",
            "模型没有遵守名字指令；hotpot_qa 走无 prefill 路径，最可能是 prompt 未被遵守",
        )

    ans_rate = 1 - no_answer / n
    if ans_rate >= 0.9:
        rep.ok(f"解析出 <A> 答案: {n - no_answer}/{n} ({ans_rate:.1%})")
    elif ans_rate >= 0.5:
        rep.warn(f"解析出 <A> 答案: {n - no_answer}/{n} ({ans_rate:.1%})")
    else:
        rep.fail(f"仅 {ans_rate:.1%} 的轨迹解析出 <A> 答案（{no_answer}/{n} 无答案）")

    if not cfg.require_name_prefix:
        rep.info("no-name-prefix 模式不应用 Alice/Bob 名字混用惩罚")
    elif name_mixed == 0:
        rep.ok("没有单条发言同时混用 Alice/Bob 名字")
    else:
        rep.warn(f"{name_mixed}/{n} 条轨迹存在名字混用（会被 -10 惩罚）")

    if dup_traj == 0:
        rep.ok("没有重复发言")
    elif dup_traj / n <= 0.05:
        rep.warn(f"{dup_traj}/{n} 条轨迹有重复发言（ppl_score 会被置 0）")
    else:
        rep.fail(f"{dup_traj}/{n} 条轨迹有重复发言（比例过高）")

    if blank_turns == 0:
        rep.ok("没有空白发言")
    else:
        rep.fail(f"{blank_turns} 条空白发言")

    if mojibake == 0:
        rep.ok("未发现乱码 / 控制字符")
    else:
        rep.warn(f"{mojibake} 条发言含替换字符或控制字符，需人工抽查")

    if length_finish == 0:
        rep.ok("没有因长度上限被截断的发言")
    else:
        rep.warn(
            f"{length_finish} 条发言 finish_reason=length（被 max_tokens 截断）",
            f"占全部发言的 {length_finish / total_turns:.1%}",
        )


def check_termination(cfg, flat, rep):
    n = len(flat)
    if n == 0:
        return
    reasons = Counter(r.get("termination_reason", "?") for _, r in flat)
    lines = [f"{k}: {v} ({v / n:.1%})" for k, v in reasons.most_common()]
    rep.info(f"终止原因分布（共 {n} 条）", "\n".join(lines))

    err = reasons.get("error", 0)
    large = reasons.get("large", 0)
    agree = reasons.get("agreement", 0)

    if err == 0:
        rep.ok("无 error 终止")
    elif err / n <= 0.05:
        rep.warn(f"error 终止 {err} 条（{err / n:.1%}），略高但可接受")
    else:
        rep.fail(
            f"error 终止 {err}/{n} = {err / n:.1%}，超过 5%",
            "先不要训练：查看 logs/vllm_*.log 与上面 [agent] 失败日志",
        )

    if large / n > 0.2:
        rep.fail(f"large 终止 {large}/{n} = {large / n:.1%}，超过 20%：模型不遵守长度约束")
    elif large:
        rep.warn(f"large 终止 {large}/{n} = {large / n:.1%}")

    if agree / n >= 0.5:
        rep.ok(f"agreement 终止 {agree}/{n} = {agree / n:.1%}（对话正常收敛）")
    elif agree:
        rep.warn(f"agreement 终止 {agree}/{n} = {agree / n:.1%} 偏低，多数靠 max_round 结束")
    else:
        rep.warn(
            "没有任何 agreement 终止：所有对话都跑满 max_round",
            "debate 的 Bob 被指示「不要给答案」，若从不输出 <A> 则无法触发一致终止",
        )


def check_reward_selection(cfg, flat_cleaned, rep):
    n = len(flat_cleaned)
    missing = Counter()
    # rank is legitimately None for unselected trajectories, so test for the
    # field's presence rather than its value
    for _, r in flat_cleaned:
        for field in ("correct_score", "token_score", "ppl_score", "reward", "selected", "rank"):
            if field not in r:
                missing[field] += 1
    if not missing:
        rep.ok("全部轨迹都含 correct/token/ppl/reward/selected/rank 字段")
    else:
        rep.fail(
            "以下字段缺失: " + ", ".join(f"{k} ({v} 条)" for k, v in missing.items()),
            "reward 打分或筛选可能没有跑完",
        )

    sel = [r for _, r in flat_cleaned if r.get("selected")]
    uns = [r for _, r in flat_cleaned if not r.get("selected")]
    if not sel:
        rep.fail("没有任何轨迹被选中，SFT 数据集会是空的")
        return

    def mean(rows, field):
        vals = [r.get(field) for r in rows if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    rep.info(
        f"选中 {len(sel)} 条 / 共 {n} 条",
        f"选中   reward={mean(sel, 'reward'):.4f}  correct={mean(sel, 'correct_score'):.4f}  "
        f"token={mean(sel, 'token_score'):.4f}  ppl={mean(sel, 'ppl_score'):.4f}\n"
        f"未选中 reward={mean(uns, 'reward'):.4f}  correct={mean(uns, 'correct_score'):.4f}  "
        f"token={mean(uns, 'token_score'):.4f}  ppl={mean(uns, 'ppl_score'):.4f}",
    )

    r_sel, r_uns = mean(sel, "reward"), mean(uns, "reward")
    if r_sel > r_uns:
        rep.ok(f"选中组 reward 更高（{r_sel:.4f} > {r_uns:.4f}）")
    else:
        rep.fail(f"选中组 reward 未高于未选中组（{r_sel:.4f} vs {r_uns:.4f}）")

    c_sel, c_uns = mean(sel, "correct_score"), mean(uns, "correct_score")
    if c_sel > c_uns:
        rep.ok(f"选中组任务正确率更高（{c_sel:.4f} > {c_uns:.4f}）：reward 与任务质量一致")
    elif c_sel == c_uns:
        rep.warn(
            f"选中组与未选中组正确率相同（{c_sel:.4f}）",
            "高 reward 完全由 token/ppl 项驱动，correct_score 未提供区分度",
        )
    else:
        rep.warn(
            f"选中组正确率反而更低（{c_sel:.4f} < {c_uns:.4f}）",
            "reward 被 token 惩罚与 ppl 项主导：检查 lambda1/lambda2 与 episilon",
        )

    # rank must be strictly increasing with decreasing reward
    ranked = sorted([r for r in sel if r.get("rank") is not None], key=lambda r: r["rank"])
    rewards = [r["reward"] for r in ranked]
    if rewards == sorted(rewards, reverse=True):
        rep.ok(f"rank 与 reward 单调一致（rank 0 reward={rewards[0]:.4f} 最高）")
    else:
        bad = next(i for i in range(1, len(rewards)) if rewards[i] > rewards[i - 1])
        rep.fail(f"rank 与 reward 不单调，首个逆序在 rank {bad}")

    if cfg.episilon is not None and sel:
        below = [r for r in sel if r["reward"] < cfg.episilon]
        if not below:
            rep.ok(f"选中轨迹 reward 全部 >= episilon({cfg.episilon})")
        else:
            rep.fail(f"{len(below)} 条选中轨迹 reward < episilon({cfg.episilon})")


def check_datasets(cfg, iteration, cleaned_flat, rep):
    by_id = {(t, r.get("trajectory_id", 0)): r for t, r in cleaned_flat}
    try:
        from datasets import load_from_disk
    except ImportError:
        load_from_disk = None

    for speaker, role_key, path_fn in (
        ("Alice", "system_first", cfg.alice_dataset_path),
        ("Bob", "system_second", cfg.bob_dataset_path),
    ):
        path = path_fn(iteration)
        label = f"{speaker} 数据集"
        if not os.path.isdir(path):
            rep.fail(f"{label} 目录不存在: {path}")
            continue
        if load_from_disk is None:
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(path) for f in fs
            )
            if size > 0:
                rep.info(f"{label} 存在（{size / 1e6:.2f} MB），未安装 datasets 无法深入检查")
            else:
                rep.fail(f"{label} 为空目录")
            continue

        ds = load_from_disk(path)
        if "train" not in ds or len(ds["train"]) == 0:
            rep.fail(f"{label} 的 train 划分为空")
            continue
        n_train, n_test = len(ds["train"]), len(ds.get("test", []))
        rep.ok(f"{label}: train {n_train} 行 / test {n_test} 行")

        # role routing: own turns -> assistant, partner -> user
        bad_role = 0
        checked = 0
        for row in ds["train"]:
            key = (row.get("task_id"), row.get("trajectory_id", 0))
            src = by_id.get(key)
            if src is None:
                continue
            checked += 1
            msgs = split_chat(row["text"])
            if not msgs or msgs[0][0] != "system":
                bad_role += 1
                continue
            # compare stripped: chat templates may trim surrounding whitespace
            if msgs[0][1].strip() != (src.get(role_key) or "").strip():
                bad_role += 1
                continue
            own = [t["content"] for t in src.get("turns", []) if t.get("speaker") == speaker]
            other = [
                t["content"] for t in src.get("turns", [])
                if t.get("speaker") == partner_of(speaker)
            ]
            got_assistant = [c.strip() for r, c in msgs if r == "assistant"]
            got_user = [c.strip() for r, c in msgs if r == "user"]
            if got_assistant != [c.strip() for c in own]:
                bad_role += 1
                continue
            if got_user != [c.strip() for c in other]:
                bad_role += 1

        if checked == 0:
            rep.warn(f"{label}: 无法与 cleaned 轨迹对应（task_id/trajectory_id 列缺失？）")
        elif bad_role == 0:
            rep.ok(
                f"{label}: 角色路由正确（{checked} 行核对通过）",
                f"{speaker} 的发言在 assistant 位置，伙伴的发言在 user 位置",
            )
        else:
            rep.fail(
                f"{label}: {bad_role}/{checked} 行角色路由或 system prompt 不匹配",
                "训练数据构造有误，务必先修复再训练",
            )


# ----------------------------------------------------------------------- main
def show_samples(cleaned_flat, n_show):
    sel = [x for x in cleaned_flat if x[1].get("selected")]
    uns = [x for x in cleaned_flat if not x[1].get("selected")]
    print("\n" + "=" * 74)
    print(f"采样展示（选中 {min(n_show, len(sel))} 条 / 未选中 {min(n_show, len(uns))} 条）")
    print("=" * 74)
    for title, pool in (("选中（高 reward）", sel), ("未选中", uns)):
        for task_id, res in pool[:n_show]:
            print(
                f"\n--- [{title}] task {task_id} traj {res.get('trajectory_id')} "
                f"reward={(res.get('reward') if res.get('reward') is not None else 0.0):.3f} "
                f"(correct {res.get('correct_score')} token {res.get('token_score')} "
                f"ppl {res.get('ppl_score')}) rank={res.get('rank')} ---"
            )
            print(f"  termination: {res.get('termination_reason')} | "
                  f"tokens: {res.get('token_count')} | "
                  f"golden: {res.get('answer')!r} | final: {res.get('final_answer')!r}")
            for i, t in enumerate(res.get("turns", [])):
                content = (t.get("content") or "").replace("\n", " ")
                if len(content) > 160:
                    content = content[:160] + " …"
                print(f"  [{i + 1}] {t.get('speaker')}({t.get('token_count')}t): {content}")


def main():
    parser = argparse.ArgumentParser(description="iSFT iteration quality checker")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--show", type=int, default=3,
                        help="how many selected/unselected dialogues to print")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg = load_run_config(args.config)
    i = args.iteration

    raw_path = cfg.raw_path(i)
    if not os.path.exists(raw_path):
        print(f"[check] {raw_path} 不存在。先在部署好的 vLLM 上运行:\n"
              f"  python sft_script.py --config {args.config} --no_train --iterations {i + 1}")
        return 1

    raw = read_jsonl(raw_path)
    rewarded = read_jsonl(cfg.rewarded_path(i))
    cleaned = read_jsonl(cfg.cleaned_path(i))
    flat_raw = flatten(raw)
    flat_rewarded = flatten(rewarded)
    flat_cleaned = flatten(cleaned) or flat_rewarded

    print("=" * 74)
    print(f"iSFT iteration {i} 质量核查 — {cfg.run_name} ({cfg.dataset_type})")
    print(f"raw: {raw_path}")
    print("=" * 74)

    rep = Report()
    check_counts(cfg, i, raw, rep)
    check_independence(cfg, flat_raw, rep)
    check_format(cfg, flat_raw, rep)
    check_termination(cfg, flat_raw, rep)
    check_reward_selection(cfg, flat_cleaned, rep)
    check_datasets(cfg, i, flat_cleaned, rep)

    print()
    rep.dump()

    if args.show:
        show_samples(flat_cleaned, args.show)

    n_fail, n_warn = rep.counts()
    print("\n" + "=" * 74)
    if n_fail == 0 and n_warn == 0:
        print("结论：全部通过 — 可以进入含训练的 smoke test")
    elif n_fail == 0:
        print(f"结论：通过，但有 {n_warn} 项警告 — 建议人工确认警告项后再训练")
    else:
        print(f"结论：不通过（{n_fail} 项失败，{n_warn} 项警告）— 先不要训练")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
