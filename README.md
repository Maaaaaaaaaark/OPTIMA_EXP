# OPTIMA 论文与代码审计：Qwen-0.5B、双独立 Agent 与复现实验方案

审计日期：2026-09-17

## 1. 结论先行

这项计划在研究问题上是合理的，但应称为“OPTIMA 的小模型、异构参数化双 Agent 扩展”，不能称为对论文数值的严格复现，原因有三点：

1. 论文使用 Llama 3 8B / Llama 3.2 3B；Qwen-0.5B 的容量、tokenizer、chat template 与基础能力差异很大。
2. 论文明确说明两个 Agent 使用同一个训练后的模型；你的设定要求 Alice 与 Bob 拥有独立参数，改变了优化对象。
3. 当前仓库只是作者的研究代码快照，存在硬编码、未固定生成 seed、论文与配置不完全一致等问题，不能不经改造直接跑。

最稳妥的实验顺序是：

1. 先做 20-100 个样本的端到端 smoke test，验证日志、终止条件、评分和角色隔离。
2. 用 Qwen-0.5B 做“共享参数双 Agent”单 seed，验证 OPTIMA pipeline 本身。
3. 再做“两个独立模型 Alice/Bob”单 seed，回答你的核心问题。
4. 最终结论至少在一个 Information Exchange 任务和一个 Debate 任务上跑 3 个 seed。单 seed 只能作调试或 pilot，不能用于判断方法是否稳定。

建议明确模型 ID。本文以下默认 `Qwen/Qwen2.5-0.5B-Instruct`；如果实际想用 Qwen2、Qwen3 或 base 而非 instruct，配置与 prompt 行为需要重新确认。

## 2. 论文方法的准确还原

OPTIMA 每轮执行 Generate -> Rank -> Select -> Train。

轨迹奖励为：

`R = R_task - lambda_token * normalized_tokens + lambda_loss * R_loss`

代码中以负的 `lambda1` 表示 token 惩罚，例如论文 `lambda_token=0.6` 对应代码 `lambda1=-0.6`。`R_loss` 在论文中用于约束可读性：取轨迹各轮在冻结 base model 下的最大语言建模 loss，再以其倒数形式进入 reward。这里的 reward model 应保持冻结，不应随 Alice/Bob 一起更新。

### 初始化

- 每个任务采样多个带随机格式 prompt 的轨迹。
- 选每个任务 reward 最高的轨迹，再在全体中保留高于阈值且排名前 70% 的样本。
- 移除格式提示后 SFT，得到后续 iSFT/iDPO 的初始化模型。

### iSFT

- 每个任务生成 N 条完整双 Agent 轨迹。
- 选 reward 最高者；按固定阈值过滤，再保留前 70%。
- IE 任务从上一轮继续训练；Debate 任务每个 SFT iteration 从初始模型重启，以避免小数据过拟合。

### iDPO

- 将对话轮次视为树节点。
- 每轮选择一个可扩展节点；从 reward 最高的前 10 个候选中按 softmax 采样。
- 每个选中节点 rollout 3 条轨迹，共搜索 8 轮，即理论上 24 条 rollout。
- 子节点估计 reward 超过 `theta_dpo_filter` 且差值超过 `theta_dpo_diff` 时构造 chosen/rejected。
- 论文保留按较高 reward 排序后的前 50% pair。
- 论文的 iDPO 使用 RPO（DPO + NLL，alpha=1）；仓库通过 `run_rpo.py` 给 `DPOTrainer` 设置 `rpo_alpha=1.0`。

### iSFT-DPO

- 每个外层 iteration 先 SFT，再 DPO。
- IE 的 SFT 从上一轮 DPO 结果继续；Debate 的 SFT 从初始模型重启。
- DPO 从当轮 SFT 模型继续，使用标准 DPO，而不是 RPO。

## 3. 作者是否留下双 Agent 接口

答案是“留下了低层接口，但没有完成独立训练闭环”。

已经存在的接口：

- `train/inference.py::inference` 接受 `model_first`、`model_second`、`url_first`、`url_second`、两个 tokenizer path。
- `train/datagenerate.py::vllm_data_generate` 将这些参数继续传递。
- `vllm_data_generate_once` 会创建两个不同的 `VllmAgent` 对象；它们有独立 system prompt 和 memory。
- 非 vLLM 的 `data_generate` 甚至可以分别加载两个 model path 到两个 device。

没有完成的部分：

- `train/sft.py`、`train/dpo.py`、`train/sft_dpo.py` 始终把两边设为同一个 `"Llama-3"`，并部署同一个 `model_path`。
- 每轮只有一个 `model_path` 和一个 checkpoint 目录，没有 `alice_model_path` / `bob_model_path`。
- SFT 数据构造把同一完整对话复制为 `conversationA` 和 `conversationB`，用于训练一个共享 policy。
- DPO pair 没有按“当前行动者是 Alice 还是 Bob”路由到两个独立数据集。
- MCTS rollout 只有一个 `model_url`，所以无法在轮次切换时调用两个不同模型。

因此当前代码中的“两 Agent”只在对象状态和 prompt 上独立，参数并不独立。

## 4. Qwen-0.5B 的硬性兼容问题

以下位置不能只改模型路径：

1. `agent/agent.py`、`train/datagenerate.py`、`train/monte_carlo_deploy.py` 手写了 Llama-3 的 `<|start_header_id|>`、`<|end_header_id|>`、`<|eot_id|>` 模板。
2. `model/llm.py` 用 Llama-3 token 和字符串切割生成结果。
3. `alignment-handbook/mask/mask.py` 写死 Llama-3 token ID，用它识别 Alice/Bob 并做 loss mask。换 Qwen 后可能没有设置 `response_token_ids`，直接报错；即便不报错也会训练错误 token。
4. `reward/reward.py` 与 `reward/deploy_reward.py` 写死 `pad_token_id=128002`。
5. `reward/deploy_reward.py::RewardModel` 写死了作者机器上的 Llama-3 8B 路径。
6. 所有 launcher 把 served model name 写死为 `Llama-3`。
7. 训练 YAML 写死 Llama 路径、bf16、Flash Attention 2 和作者目录。

正确做法：始终以 `AutoTokenizer.from_pretrained(...).apply_chat_template(...)` 和 tokenizer 自身的 `pad_token_id/eos_token_id` 为准，不在业务代码中维护模型专属 token ID。

## 5. 双独立模型的正确数据语义

“独立 Agent”应定义为：Alice 与 Bob 从相同的 Qwen 初始权重复制开始，但拥有不同模型参数、优化器、checkpoint 和推理 endpoint；训练后不共享权重。reward base model 可以共享，因为它只是冻结的评分参照，不是 Agent。

### iSFT 数据拆分

对一条联合轨迹：

- Alice 数据只对 Alice 的回复计算 loss；Bob 的话作为对方输入。
- Bob 数据只对 Bob 的回复计算 loss；Alice 的话作为对方输入。
- 不应像当前代码一样把所有轮次都标成 `assistant` 再依赖 Llama token ID mask。
- 建议显式保存 `speaker`，构建每个 Agent 的训练样本时，将自己的目标轮标为 assistant，将对方已发生的轮次放入上下文并 mask。

### iDPO 数据拆分

- 一个树节点的 child 表示下一位行动 Agent 的候选回复。
- chosen/rejected pair 必须路由到该行动者的数据集。
- Alice DPO 更新只使用 Alice action pair；Bob 同理。
- 下一轮生成时同时加载 `alice/iteration_t` 与 `bob/iteration_t`。

### 联合 reward

仍对完整联合轨迹打一个系统 reward。这符合“共同完成任务”的目标，但会产生 credit assignment 噪声。建议第一版保持论文 joint reward，之后再做 per-agent 或 turn-level reward 作为额外消融，不要一开始同时改变过多变量。

## 6. 日志与“输出每一次对话”

当前 iSFT 原始 JSONL 已保存每个 task 的全部 `results`，每个 result 含 `conversation`、答案、token 数与 prompt；终端也逐轮 print。但格式不足以严谨复现。

建议每条轨迹保存：

- `run_id`、方法、数据集、split、全局 seed、task seed、iteration、task_id、trajectory_id。
- Alice/Bob 的模型 ID、checkpoint hash、endpoint、采样参数。
- 原始 question、两边 context、两边 system prompt。
- 每轮结构化字段：`turn_index`、`speaker`、`content`、`token_count`、`parsed_answer`、`finish_reason`。
- 轨迹级 `R_task`、归一化 token 项、`R_loss`、总 reward、是否入选 SFT。
- 训练数据中对应的 Alice/Bob 样本 ID。

iSFT 可以输出一份人类可读 transcript（每条轨迹完整列出）。iDPO 和 hybrid 即使“不输出内容”，仍必须在本地保存训练所需 chosen/rejected 和最小审计元数据；只关闭终端 transcript/最终报告展开即可，不能完全不留数据。

## 7. Seed 设计与论文对比

当前代码只在 Alignment Handbook 的训练 recipe 里设 `seed: 42`。数据分割与生成使用全局 `random`、`numpy.random`，多线程完成顺序也不确定，因此所谓 single seed 目前并不真正可复现。

必须加入：

- CLI/config 中的 `seed`。
- 启动时固定 Python、NumPy、Torch、Transformers 和 CUDA seed。
- 用 `(global_seed, iteration, task_id, trajectory_id)` 派生局部 RNG；不要让 thread 调度改变随机序列。
- 数据集随机分配、prompt pool 采样、MCTS 节点 softmax 采样、vLLM sampling seed 都使用派生 seed。
- 结果按 `task_id/trajectory_id` 排序后写出。

不建议“只有与论文差距大才跑 multi-seed”，因为这是按结果决定样本量。资源有限时可以：

1. 将 seed 42 明确定义为 pilot，仅用于排错和估算差距。
2. 预先定义触发阈值，例如任一主指标绝对差超过 5 个百分点，或 token 数相对差超过 20%，则追加 seed 43、44。
3. 对最终要写入结论的代表任务，无论 pilot 结果如何都跑 3 seeds，报告 mean ± std 和每个 seed 原始值。

## 8. 推荐实验矩阵

### 最小可行矩阵

- Information Exchange：2WikiMultiHopQA。它最依赖双方信息交换，论文改进最大。
- Debate：ARC-C。答案可 exact match，训练与评估比 MATH 稳定，且论文在小模型上仍有明显提升。
- 方法：Base、iSFT、iDPO、iSFT-DPO。
- 架构：共享 Qwen policy（软件对照）与 Alice/Bob 独立 policy（目标设定）。
- 先 seed 42 pilot；正式结果 seed 42/43/44。

### 完整矩阵

- IE：HotpotQA、2WMHQA、TriviaQA、CBT。
- Debate：MATH、GSM8K、ARC-C、MMLU。
- 每个方法同时报告 task metric、平均生成 token、完成率、平均轮数和无效格式率。

### 与论文比较的方式

不能只比较绝对值。至少分三层：

1. 论文原始 Llama 数值：作为历史参考。
2. Qwen-0.5B base/MAD/AutoForm：同模型、同数据、同评估代码的公平基线。
3. Qwen OPTIMA 变体：报告相对 base 的提升与 token 节省。

论文 Llama 3 8B 的主表中，2WMHQA 为：iSFT 72.4 F1 / 61.2 tokens，iDPO 66.1 / 35.9，iSFT-DPO 74.2 / 54.9；ARC-C 为：74.1 / 92.2，74.5 / 97.8，77.1 / 88.0。论文 Llama 3.2 3B 的对应值为 2WMHQA：65.2 / 47.7，57.0 / 65.4，66.8 / 51.4；ARC-C：62.7 / 156.2，63.1 / 132.7，61.6 / 141.4。0.5B 的绝对结果明显低于这些值并不自动说明复现失败。

## 9. 论文与仓库配置的已发现差异

- 论文称每个任务完成六轮优化；iSFT recipe 是 6，iDPO 多数 recipe 是 5（可解释为初始化后 5 个 DPO iteration），hybrid 多数为 3 个 SFT+DPO cycle，但 ARC 的 iDPO/hybrid recipe 只写 2。
- 论文 iSFT MATH 的 `lambda_loss=0.9`，仓库 `train/sft_recipes/math.yaml` 为 `lambda2=0.8`。
- 论文 hybrid MMLU 的 `theta_sft=0.6`，仓库为 `initial_episilon=0.7`。
- 多个 hybrid recipe 的 DPO filter 阈值为 0.43/0.45，而论文表中除 ARC 0.45 外通常为 0.4。
- 代码对 token 数 <= 40 的轨迹强制施加 `token_score=-1`，论文公式没有说明这个额外规则。
- SFT 数据选择对 MATH/ARC 有去掉顶部 10%、保留 10%-80% 的特殊分支；论文只说明 top 70%，且代码分支检查的是 `prompt_type == "arc"`，实际调用传入通常是 `debate`，ARC 特例可能根本未触发。
- 论文说用 base model loss；服务端 reward model 路径被写死为 Llama-3 8B，即使训练模型改成 Qwen 也仍会用 Llama，除非修改。

这些差异必须在实验清单中标记为“paper-faithful”或“code-faithful”，不能混用后仍声称严格复现。

## 10. 代码地图：每个部分的功能

### 根目录入口

- `sft_script.py`：读取 iSFT recipe、建输出目录、调用 `sft_train_v2`。
- `dpo_script.py`：读取 iDPO recipe、调用 `dpo_train_v2`。
- `sft_dpo_script.py`：读取 hybrid recipe、调用 `sft_dpo_train_v2`。
- `inference_main.py`：单次推理/数据生成 CLI；按 dataset type 创建 loader，再调用统一 inference。
- `inference_script.py`：遍历 checkpoint 根目录，部署每个 checkpoint 并生成测试结果。
- `reward_main.py`：raw trajectory -> reward -> clean -> SFT/DPO Hugging Face dataset 的总入口。
- `stats_main.py`：计算结果 F1/accuracy 和平均 token。
- `multi_sc_data_generate_script.py`：每题生成 50 条轨迹，用于 self-consistency。
- `multi_sc_analysis_main.py`：分析多采样随采样数变化的性能与 coverage。
- `ppl_deploy.py`：启动 Ray Serve 的冻结 LM-loss 服务。
- `README.md` / `README_revise.md`：安装与运行说明；后者是较旧/不完整副本。
- `requirements.txt`：依赖列表，但缺少若干实际使用包的明确 pin，例如 datasets、sympy、TRL/accelerate 由 vendored handbook 间接提供。

### `message/`、`model/`、`agent/`

- `message/message.py`：Pydantic 消息对象，字段为 role/content。
- `model/llm.py`：本地 Transformers 推理封装；`Llama3` 实现完全绑定 Llama-3 格式。
- `agent/agent.py::Agent`：本地模型 Agent，维护独立 memory。
- `agent/agent.py::VllmAgent`：通过 OpenAI-compatible vLLM endpoint 推理；`step` 更新 memory，`no_memory_step` 不更新。两个对象可以指向不同 endpoint，这是现有独立 Agent 接口的基础。

### `utils/`

- `utils/config.py`：模型与数据集路径常量；当前全部为空，仓库不能开箱运行。
- `utils/prompt_template.py`：IE、数学 solver/critic、ARC/MMLU solver/critic prompt；也包含用 GPT-4o 生成 format prompt pool 的工具。
- `utils/prompts_*.jsonl`：初始化阶段的格式多样性 prompt pool。
- `utils/utils_token.py`：按发言轮次交替使用两个 tokenizer 计算 token 数。
- `utils/utils.py`：将 JSONL 按相同 question 聚合结果的小工具。

### `dataloader/`

- `DataloaderForHotpotQA`：把 supporting facts 与干扰 context 分给两个 Agent；分配中含随机性。
- `DataloaderForMWHQA`：交替拆 supporting facts，随机拆其他 context。
- `DataloaderForTrivalQA`：随机分配搜索摘要；类名/配置一直拼成 `Trival` 而非 `Trivia`。
- `DataloaderForCBT`：把故事前后半段分给两个 Agent。
- `DataloaderForGSM8K`、`DataloaderForMATH`：无私有 context，返回题目与规范答案。
- `DataloaderForARC`、`DataloaderForMMLU`：构造选项题；MMLU 试图排除与 ARC 重合问题并固定 shuffle seed 42。
- `DataloaderForMix`：轮转混合数据集。
- `process_dataloader_for_sft`：每题取 reward 最优轨迹，阈值与排名过滤，构造 SFT 数据。
- `process_dpo_format_to_dataset`：把 MCTS chosen/rejected JSONL 转为 HF DatasetDict。
- `data_clean`：若单条回复同时冒充 Alice/Bob 或重复姓名，则把 reward 减 10。

### `train/datagenerate.py`

- `data_generate`：旧的本地双模型生成路径，接口天然允许两个模型，但只写了 HotpotQA。
- `vllm_data_generate`：加载两个 tokenizer，选择 prompt/loader，恢复已有 task，构造并行工作项。
- `vllm_data_generate_once`：创建 Alice/Bob 两个 Agent，按 explore_count 生成多条轨迹并写 JSONL。
- `conversation`：Alice/Bob 轮流发言，最多 10 轮；只有连续两次解析到相同答案才结束。
- `find_all_linear_names`：旧的 LoRA target-module 辅助函数，主路径未使用。

### `reward/`

- `reward/reward.py`：本地 reward、F1/exact/math 评分、结果统计和旧的 vLLM logprob reward 路径。
- `reward/deploy_reward.py`：Ray Serve 冻结语言模型 loss 服务；并行给轨迹写 `correct_score/token_score/ppl_score/reward`。
- 奖励服务实际返回的是平均 cross-entropy loss，而变量常写作 ppl；reward 使用 `lambda2 / max_loss`，命名不精确但与论文代码意图一致。

### `train/sft.py`

每轮：部署当前模型的 8 个 vLLM 服务 -> 生成轨迹 -> 启动 reward 服务 -> 打分/清洗/选 SFT 数据 -> 改写训练 YAML -> 调用 Alignment Handbook SFT -> 更新单一 model path。

### `train/monte_carlo.py` 与 `train/monte_carlo_deploy.py`

- `treeNode`：存 parent/children、轮到的 agent、平均 value、min/max child。
- `MonteCarloTreeDeploy`：8 次 search，每次选节点后 rollout 3 条；回传完整轨迹 reward。
- `generate`：从同父节点的 max/min child 构造 DPO pair，按 chosen reward 排序并截断。
- 输出四类 JSONL：完整 rollout、DPO pair、root format pair、整棵树审计 record。

### `train/dpo.py`

每轮：部署当前单一模型 -> 抽样 100 条估计 token 上限分位数 -> MCTS 生成 pair -> 转 HF dataset -> 调用 `run_rpo.py` -> 更新单一 DPO checkpoint。

### `train/sft_dpo.py`

每个外层循环完整执行一次 SFT 阶段，再部署 SFT checkpoint 做 MCTS/DPO，最后更新到 DPO checkpoint。它实现的是交替训练，不是把两种 loss 在同一步相加。

### `analysis/`

- 按采样轮数计算 best/majority-style 结果、coverage 和 token 预算。
- `get_best_division`、`get_best_answer`、`multi_sc_analysis` 用于论文 inference scaling 曲线。

### `alignment-handbook/`

- vendored Hugging Face Alignment Handbook，提供 SFT/DPO/RPO trainer、配置和 DeepSpeed 启动。
- `scripts/run_sft.py` 应用 chat template，并用自定义 `Mask` 只对当前角色回复算 loss。
- `scripts/run_dpo.py` 是标准 DPO。
- `scripts/run_rpo.py` 强制 `rpo_alpha=1.0`，实现论文 iDPO 的 DPO+NLL。
- `mask/mask.py` 是最关键的 Llama 专属代码，必须为 Qwen 重写。
- `recipes/Llama3-8b/**` 保存各任务训练超参；大量路径是作者机器绝对路径。
- `tests/`、`src/alignment/`、`chapters/` 主要是上游 handbook 的测试、配置工具和文档，并非 OPTIMA 算法本体。

### `scrips/`

拼写错误的 legacy 目录，内容是较旧的 launcher 副本，缺少后来加入的环境参数。应标记 deprecated，避免与根目录入口混用。

## 11. 需要优先修复的实现风险

1. `Mask` 的 Llama token ID 硬编码是 Qwen 迁移的第一阻塞点。
2. `reward_batch_based_on_deploy` 对 list exact-match 写了 `.lower` 而不是 `.lower()`，而且拿整个 `result["answer"]` 调 `.strip()`；该分支会异常后把正确率记 0。
3. `DataloaderForMMLU` 在过滤后没有更新 `self.total`，可能越界或循环逻辑错误。
4. `DataloaderForTrivalQA` 一边按原长度循环，一边从 list 删除随机元素，最终可能对空 list `random.choice`。
5. `DataloaderForHotpotQA` 只取每个 title 的第一句，而 supporting fact 可能不是第一句，会丢关键信息。
6. `process_dataloader_for_sft` 在阈值判断前就把所有 best result 加入列表，虽然后面再次过滤，但统计 count/平均值逻辑与选择逻辑不完全一致；count 为 0 时会除零。
7. SFT 数据把双方所有 utterance 都设为 assistant，依赖脆弱的姓名 token mask；更换 tokenizer 后必须改为显式角色 mask。
8. MCTS root pair 使用循环遗留的 `nodeType`，在边界情况下可能未定义或角色错误。
9. MCTS `search` 为去重会从 `all_nodes` 删除节点，改变后续候选集合；与论文伪代码的“排除候选”不完全等价。
10. iDPO 的 token 上限用 100 条样本的 85 分位（hybrid 为 80 分位），论文未说明该额外启发式。
11. 多处服务 readiness loop 没有 timeout；服务失败会永久挂起。
12. 使用 `pkill -f`、`source ~/.bashrc`、`conda activate`、固定 8 GPU/端口，只适用于特定 Linux 环境；当前 Windows 工作区不能直接运行。
13. `mid_yaml_root_path[21:]` 依赖固定字符串长度，路径一变就失效。
14. 多线程写入虽有锁，但 task 输出顺序不稳定；resume 与 seed 审计不可靠。
15. 当前仓库没有 checkpoints、results、my_datasets；`utils/config.py` 的数据/模型路径全部为空。

## 12. 从第一步开始的修改顺序

### Phase 0：冻结实验定义

- 确定准确模型 ID、硬件、任务子集、数据版本、最大上下文、全参或 LoRA。
- 明确“独立 Agent”是独立参数，而不仅是独立 memory。
- 建立 paper-faithful 与 code-faithful 两套配置标签。

### Phase 1：配置与可复现基础设施

- 用 dataclass/YAML 统一管理 model、tokenizer、dataset、ports、GPU、seed。
- 删除作者绝对路径和 `[21:]` 路径切片。
- 加 timeout、错误传播、进程清理和 run manifest。
- 所有输出采用 `runs/{run_id}/{method}/{dataset}/{seed}/...`。

### Phase 2：模型无关的 chat 与 loss mask

- 删除所有手写 Llama chat template 和 token ID。
- 通过 Qwen tokenizer 原生 chat template 生成输入。
- 写 speaker-aware collator：通过结构化 turn metadata mask，而不是搜索 token ID。
- reward 的 pad/eos 从 tokenizer 读取；冻结 reward model 路径配置化。

### Phase 3：双模型推理闭环

- 配置 `alice.model_path/url/name` 和 `bob.model_path/url/name`。
- conversation 与 MCTS 每轮根据 speaker 选择 endpoint/tokenizer。
- token count 按实际发言者 tokenizer 计算。
- 加结构化 transcript 与稳定 task/trajectory ID。

### Phase 4：双模型 iSFT

- 联合生成、联合 reward、联合选择。
- 从入选轨迹生成 Alice-only loss 数据与 Bob-only loss 数据。
- 分别训练并保存两个 checkpoint。
- 每轮同时更新两个路径；Debate 的 reset 策略对两边一致执行。

### Phase 5：双模型 iDPO

- MCTS 节点记录 acting_agent。
- rollout 交替调用 Alice/Bob endpoint。
- pair 按 acting_agent 分流为两个 DPO dataset。
- 分别训练 Alice/Bob，并为每边设置正确的 reference model。

### Phase 6：hybrid

- 每轮先执行双模型 iSFT，再用两份 SFT checkpoint 做双模型 iDPO。
- 保持论文 IE/Debate 的 restart 规则。

### Phase 7：验证与正式实验

- 单元测试：parser、终止条件、speaker mask、pair 路由、reward 分解、seed 重现。
- 20 个样本 smoke test，并人工检查全部 iSFT transcript。
- 100-500 样本 pilot，确认训练 loss 与无效输出率。
- 完整 single-seed pilot；最后代表任务 3 seeds。

## 13. 是否现在就能开始训练

不能。当前工作区缺少：

- Qwen 模型或 Hugging Face 可下载配置。
- 所有 benchmark 的本地路径/版本。
- checkpoints、results、my_datasets。
- 可运行 vLLM/训练的 Linux CUDA 环境说明与 GPU 数量。
- 对“Qwen 0.5B”的准确模型 ID。

先完成 Phase 0-3 和 smoke test，再开始 iSFT；否则很可能产生表面可运行、实际 mask/角色错误的数据，后续 iDPO 与 hybrid 结果将不可解释。
