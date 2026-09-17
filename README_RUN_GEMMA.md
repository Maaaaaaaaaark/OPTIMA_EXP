# OPTIMA iSFT on google/gemma-2-2b-it (HotpotQA 双 Agent)

用 `google/gemma-2-2b-it` 替代 Qwen2.5-1.5B 跑 HotpotQA 双 Agent 实验。
**完全增量式**：现有 Qwen 配置和代码一律不动，只新增两个配置 + 一个开关
`merge_system_into_user`（默认 `false`，Qwen 路径行为不变）。

- 主实验配置：`configs/gemma2-2b/hotpot_qa.yaml`（与 `configs/qwen1.5b/hotpot_qa.yaml` 参数一致）
- Colab T4 冒烟配置：`configs/gemma2-2b/hotpot_qa_colab_smoke.yaml`（`--no_train`）

## 与 Qwen 实验保持不变的项

- OPTIMA 原作者协议：Generate → Rank → Select → Train，`<A>{answer}</A>` 解析、
  双方答案一致即停止（agreement stop）、per-task argmax + top-70% trim 窗口 +
  episilon 阈值、name prefix 要求
- Reward 公式：`R = R_task - lambda1 * (tokens / max_tokens_in_task) + lambda2 / max_per_turn_loss`
  （lambda1 = -0.6, lambda2 = 1.0，正确率 / token reward / PPL reward 均不变）
- 冻结 Reward Scorer：就是同一个 Gemma 2 2B 基模型（`reward_model_path` 留空 →
  默认取 `base_model_path`），**不新增任何 Reward Scorer**
- Prompt 实际内容与两个 Agent 的私有信息隔离（Alice/Bob 各只见自己的上下文）

## 唯一的适配：Gemma 的 chat template 不支持 system 角色

Gemma 官方 jinja 模板对以下输入直接报错：

1. `messages[0]['role'] == 'system'` → `System role not supported`
2. 角色必须以 user 开头并严格 user/assistant 交替

项目通过 OpenAI 兼容接口发 system/user/assistant 消息，因此新增一个配置开关
`merge_system_into_user: true`，在**不改动任何 prompt 文本**的前提下做最小适配：

| 场景 | 适配前 | 适配后 |
|---|---|---|
| Alice 首请求 | `[system(S), assistant(A1), user(B1)]` | `[user(S), assistant(A1), user(B1)]` |
| Bob 首请求 | `[system(S), user(A1), assistant(B1)]` | `[user(S + "\n\n" + A1), assistant(B1)]` |

规则：开头 system → user；连续同角色消息用 `"\n\n"` 合并。三处统一走同一规则
（推理和 SFT 训练看到完全相同的消息序列）：

- 推理请求：`agent/agent.py::VllmAgent._request`（在 prefill 追加之前适配）
- SFT 数据集模板：`train/dataset_build.py::_templated_row`
- 冻结 Loss Scorer 的 PPL 分帧：`reward/scorer.py::frame_utterance_for_loss`
  （Gemma 下 PPL 帧从 `[assistant]` 变为 `[user(""), assistant]`，渲染出的
  `<start_of_turn>model` 头部等价，loss 公式不变；Gemma 无 pad token，已回退用 eos）

Qwen 配置 `merge_system_into_user: false`，上述三处全部走原路径，输出逐字节不变。

## 依赖兼容性结论（无需升级）

| 依赖 | 现有版本 | Gemma2 支持情况 |
|---|---|---|
| transformers | 4.46.3 | Gemma2/Gemma2ForCausalLM 自 4.42.0 加入（4.42.4 修复完整），且 ≥ 4.45 满足 prefill 机制要求 ✓ |
| vLLM | 0.6.3.post1 | Gemma2 自 v0.5.2 注册 ✓，prefill（continue_final_message）需 ≥ 0.6.3 ✓ |
| torch | 2.4.0 | ✓ |
| Python | 3.12 | vLLM 0.6.3.post1 setup.py 声明 3.8–3.12 ✓ |

**不需要任何升级。** T4 没有原生 bf16，因此部署命令必须显式设置
`DTYPE=half`；冻结 Reward Scorer 也会在不支持 bf16 的 CUDA 设备上以
fp16 加载。数据类型适配不改变 Reward 公式。

## Linux 运行（主实验）

```bash
# 1. 部署两个 vLLM 端点（都服务同一个 Gemma 2 2B）
DTYPE=half MEM_UTIL=0.40 MAX_MODEL_LEN=4096 scripts/deploy_vllm.sh \
    google/gemma-2-2b-it google/gemma-2-2b-it

# 2. 由脚本逐轮执行：部署 → 生成 → 释放 vLLM → 打分 → LoRA SFT
scripts/run_pipeline.sh configs/gemma2-2b/hotpot_qa.yaml
```

内存说明：生成阶段两个 vLLM 端点各使用最多 40% GPU。生成结束后配置项
`release_vllm_before_scoring: true` 会先终止两个服务及其 CUDA worker，再独占
GPU 加载冻结 2B Scorer；评分结束后也会主动释放 scorer 显存。

## Colab Tesla T4 冒烟（15 GB，先 --no_train）

```bash
VLLM_BIN=/content/optima-vllm/bin/vllm \
DTYPE=half MEM_UTIL=0.40 MAX_MODEL_LEN=4096 scripts/deploy_vllm.sh \
    google/gemma-2-2b-it google/gemma-2-2b-it
python sft_script.py --config configs/gemma2-2b/hotpot_qa_colab_smoke.yaml --no_train
python scripts/check_iteration.py \
    --config configs/gemma2-2b/hotpot_qa_colab_smoke.yaml --show 3
```

冒烟规模：10 任务 × 2 轨迹、max_round 8、每轮 256 token、1 个 iteration、
thread_count 2、scorer_batch_size 1、`train_enabled: false`（`--no_train` 双保险）。

## T4 训练方式

Gemma 主配置使用 fp16 LoRA（batch size 1、gradient accumulation 16），不在
T4 上做全参数 Adam 训练。训练完成后 LoRA 会自动合并回基础模型并保存为标准
Hugging Face 完整 checkpoint，因此下一轮 vLLM 仍可直接加载
`checkpoints/<run>/alice|bob/iteration_<i>`，无需改变部署接口。

## 测试

在 train_env / vllm_env 里跑（本机 Windows 无 Python，测试无法在本地执行）：

```bash
python -m pytest tests/ -v                       # 全部离线测试（mock + 合成 tokenizer）
OPTIMA_LIVE_ENDPOINT=http://127.0.0.1:8100/v1/chat/completions \
OPTIMA_LIVE_SERVED_MODEL=alice \
python -m pytest tests/test_live_optional.py -v  # 可选：打真实端点
```

覆盖点：两个 Gemma 配置可加载且参数正确、各模块可导入、Alice/Bob prompt 与
私有上下文不同且隔离、Gemma 请求无 system 角色且严格交替、`<A>` 解析与
agreement stop、Reward 公式回归（含 -1/-10 特殊规则与 argmax/trim/episilon 选择）。

## 输出结构

与 Qwen 相同：`runs/<run_name>/iteration_<i>/{raw,rewarded,cleaned}/iteration_<i>.jsonl`、
`transcripts/`、`alice_dataset/`、`bob_dataset/`，检查点
`checkpoints/<run_name>/{alice,bob}/iteration_<i>`。
