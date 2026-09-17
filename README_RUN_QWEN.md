# OPTIMA (iSFT) on Qwen2.5-0.5B-Instruct — 运行指南

基于 [OPTIMA](https://arxiv.org/abs/2410.08115) 的复现改造：**Alice / Bob 两个参数完全独立**的
Qwen2.5-0.5B-Instruct 双智能体，任务为：

- **Information Exchange（IE）**：HotpotQA（`configs/qwen0.5b/hotpot_qa.yaml`）
- **Debate**：ARC-Challenge（`configs/qwen0.5b/arc.yaml`）

每轮 iteration：生成（8 轨迹/题 × 100 题）→ 打分（R = R_task − λ_token·token + λ_loss·R_loss）
→ 筛选（每题最优 + 全局 top-70% + 阈值）→ 按 speaker 拆成 Alice/Bob 两份数据 → 各自**全参数 SFT**。
每一条完整对话都会写入 transcript 文件供人工检查。

## 1. 环境（Linux，单卡 ≥24GB）

两个 conda 环境：

```bash
# 训练环境（sft_script.py / sft_trainer.py / reward scorer）
conda create -n train_env python=3.10 -y
conda activate train_env
pip install "torch>=2.4,<3" "transformers>=4.46,<4.49" trl==0.10.1 \
    "accelerate>=0.34,<2" "datasets>=2.14,<4" peft \
    openai rouge editdistance sympy pyyaml tqdm pydantic numpy requests sentencepiece

# vLLM 服务环境（deploy_vllm.sh）
conda create -n vllm_env python=3.10 -y
conda activate vllm_env
pip install vllm==0.6.3.post1 torch==2.4.0 "transformers>=4.45.2,<4.47"
```

> **transformers 版本区间是硬性要求**：
> - 下限：名字前缀 prefill 依赖 `continue_final_message` + `add_generation_prompt: false`，
>   低于 4.45 时该 flag 被惰性求值的 jinja 模板忽略，请求会 400 或生成错误格式。
> - 上限：**transformers 5.0（2026-07 发布）重写了 tokenizer 体系**（`apply_chat_template`
>   不再返回 str、大量旧 API 移除），trl 0.10.1 与 vllm 0.6.3.post1 都是 2024-10 的版本，
>   与 5.x 不兼容。只写 `transformers>=4.45` 会让 pip 装到 5.x，两个环境都会崩。

依赖清单见 `requirements.txt`（**不需要** ray / deepspeed / alignment-handbook）。

## 2. 准备模型与数据

```bash
# Qwen2.5-0.5B-Instruct（base + reward 打分共用）
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# 数据集（联网自动下载；也可自行 cache 后设置 HF_DATASETS_CACHE）
#   hotpotqa   (name=distractor, split=train)   -> config 的 dataset_path: hotpotqa
#   allenai/ai2_arc (name=ARC-Challenge, split=train) -> config 的 dataset_path: allenai/ai2_arc
```

修改对应 YAML 中 `base_model_path` 为本机路径（或保留 HF hub id 联网加载）。

## 3. 运行（含训练）

```bash
# 方式 A：手动两步（推荐先这样跑）
scripts/deploy_vllm.sh                              # base 模型上两个端点（8100 alice / 8101 bob）
python sft_script.py --config configs/qwen0.5b/hotpot_qa.yaml
scripts/deploy_vllm.sh stop

# 方式 B：全自动（逐 iteration 部署-运行-停服；from_initial=false 时
#         自动在 iteration 间换成新训练出的 checkpoint）
scripts/run_pipeline.sh configs/qwen0.5b/hotpot_qa.yaml
scripts/run_pipeline.sh configs/qwen0.5b/arc.yaml
```

常用参数：`--iterations N`（只跑前 N 轮）、`--no_train`（只生成+打分+建数据，不训练）、
`--overwrite`（清掉 runs/{run_name} 与 checkpoints/{run_name} 后重来）。

> ⚠️ 多轮 iteration 时若手动跑：iteration 0 之后，`from_initial=false` 的配置
> （hotpot）需要**手动用新 checkpoint 重新部署** vLLM，再跑下一轮：
>
> ```bash
> scripts/deploy_vllm.sh checkpoints/qwen0.5b-hotpotqa-isft/alice/iteration_0 \
>                       checkpoints/qwen0.5b-hotpotqa-isft/bob/iteration_0
> python sft_script.py --config configs/qwen0.5b/hotpot_qa.yaml --iterations 2
> ```
>
> arc 配置 `from_initial=true`，每轮都从 base 重启，无需换 checkpoint。

## 4. 输出结构（人工检查点）

```
runs/{run_name}/
├── config.yaml                     # 冻结的有效配置
└── iteration_i/
    ├── raw/iteration_i.jsonl       # 原始轨迹（含 turns/token_count/termination_reason）
    ├── rewarded/iteration_i.jsonl  # + correct/token/ppl/reward 分解
    ├── cleaned/iteration_i.jsonl   # + name_penalty/selected/rank
    ├── transcripts/                # ★ 每轨迹一个 txt：task_XXXXX_traj_XX.txt
    │   └── all_transcripts.txt     #    合并版（完整对话 + 奖励分解 + 是否入选）
    ├── alice_dataset/              # Alice 视角 SFT 数据（save_to_disk，含 train/test）
    └── bob_dataset/                # Bob 视角 SFT 数据
checkpoints/{run_name}/{alice,bob}/iteration_i/   # safetensors + tokenizer（可直接 vllm serve）
```

transcript 里每条发言都带 speaker / token 数 / `<A>...</A>` 解析结果 / finish_reason，
末尾标注 `selected: True rank: N` 的轨迹就是进入 SFT 训练集的对话。

## 5. 行为说明（与原 Llama 实现的差异）

- **完全去 Llama 硬编码**：chat template 用 Qwen2.5 tokenizer 原生模板；SFT mask 用
  trl `DataCollatorForCompletionOnlyLM`（`response_template="<|im_start|>assistant\n"`）。
- **双独立参数**：Alice 数据 = [Alice 系统提示] + Alice→assistant、Bob→user；Bob 镜像；
  两次独立全参 SFT，输出两个独立 checkpoint。
- **确定性**：全局 seed → 每题/每轨迹派生 `random.Random`，vLLM 请求带 per-request seed；
  同 seed 重跑 `diff -r` 结果为空。
- **vLLM 由 shell 脚本管理**：Python 内绝不启动/杀死 vLLM。
- **reward 打分进程内计算**：冻结的 Qwen-0.5B base 模型，无 Ray。
- **断点续跑**：raw jsonl 按 task_id 续跑；已存在的 checkpoint 不重复训练。
- 旧 Llama 流水线入口（`sft_train`/`sft_train_v2`）保留为报错 stub；dpo 相关旧代码
  （第 2/3 步范围）仍可正常 import。

## 6. 验证清单（首次在 Linux 上跑之前/之后）

0. 环境就绪：`vllm_env` 里 `python -c "import vllm; print(vllm.__version__)"` 输出
   `0.6.3.post1`；`train_env` 里 `python -c "import transformers; print(transformers.__version__)"`
   输出 `4.46.x`–`4.48.x`（**不是 5.x**）；`pip show trl` 输出 `0.10.1`。
1. **tokenizer/collator 单测**（train_env）：
   ```python
   from transformers import AutoTokenizer
   from trl import DataCollatorForCompletionOnlyLM
   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
   assert tok.pad_token_id == 151643 and tok.eos_token_id == 151645
   assert tok("<|im_start|>assistant\n").input_ids == [151644, 77091, 198]
   msgs = [{"role":"system","content":"sys"},{"role":"user","content":"u1"},
           {"role":"assistant","content":"a1"},{"role":"user","content":"u2"},
           {"role":"assistant","content":"a2"}]
   text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
   coll = DataCollatorForCompletionOnlyLM(
       response_template="<|im_start|>assistant\n",
       instruction_template="<|im_start|>user\n", tokenizer=tok)
   batch = coll([{"input_ids": tok(text).input_ids}])
   # labels: system/user 全 -100，两段 assistant 内容有真实 label
   ```
2. **prefill 单测**：deploy 后 curl 发送带 `continue_final_message: true,
   add_generation_prompt: false, seed: 7` 的请求；同 seed 两次输出逐字节一致；
   返回 content 是 `Alice:` 的自然延续、不含 `<|im_end|>` 残留。400 报错即
   serving 环境 transformers<4.45。
3. **无训练 smoke**：`deploy_vllm.sh` + `python sft_script.py --config
   configs/qwen0.5b/hotpot_qa.yaml --no_train --iterations 1` → raw/rewarded/cleaned
   各 100 行、按 (task_id, trajectory_id) 排序；`transcripts/` 800 个 txt 可读；
   arc.yaml 同样跑一遍。
4. **确定性**：删 run 目录，同 seed 重跑（--no_train），`diff -r` 为空。
5. **含训练 smoke**：`python sft_script.py --config configs/qwen0.5b/hotpot_qa.yaml
   --iterations 1` → `checkpoints/{run}/{alice,bob}/iteration_0` 各含
   safetensors + tokenizer；训练 loss 下降；显存峰值约 8-9GB。
6. **闭环**：用训练后的两个 checkpoint 重新 deploy，跑 iteration 1（--iterations 2）
   → transcript 与 iteration 0 有可观察差异；arc 从 base 重启（from_initial），
   hotpot 从 iteration_0 继续。
7. **兼容性**（train_env，无 ray）：
   ```bash
   python -c "import train.datagenerate, train.sft, reward.reward, reward.deploy_reward, dataloader.dataloader, utils.config"
   ```

## 7. 常见问题

- **prefill 请求 400**：serving 环境 transformers<4.45，升级后重启 vLLM。
- **SFT 后模型说话不带名字**：检查数据集构建时是否被 `filter_long_samples` 过滤了
  超长样本（SFTTrainer 从尾部截断会砍掉最后一轮 assistant 目标）。
- **显存不足**：`sft.max_seq_length` 调小（如 1536）、`per_device_train_batch_size` 调小；
  vLLM 端 `MEM_UTIL=0.4 scripts/deploy_vllm.sh ...`。
- **reward 打分与 vLLM 共存**（手动模式下生成完立即打分，vLLM 还占着显存）：
  `deploy_vllm.sh` 默认 `MEM_UTIL=0.35`，两个端点合计留出约 7GB 给进程内的冻结 0.5B
  打分模型，24GB 卡上无需停止 vLLM；若仍 OOM，把 `scorer_batch_size` 调小。
- **OOM 在 reward 打分**：`scorer_batch_size: 8`。
