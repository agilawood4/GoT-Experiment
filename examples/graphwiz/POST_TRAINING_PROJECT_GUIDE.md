## GraphWiz 后训练项目说明（小白友好版）

这份文档的目标是：把你当第一次接触该项目的人，帮助你在最短时间内理解“这个后训练工程到底在做什么、为什么这么做、怎么一步步跑通”。

---

## 1. 先用一句话理解项目

你现在这套代码做的是：

1. 先用 GoT/GoO 跑图推理任务，得到每条题目的完整推理轨迹（trajectory）  
2. 再把轨迹转换成训练数据（SFT / DPO / RL）  
3. 最后用 TRL 跑三条后训练线，让模型回答策略更稳定、更准确

可以把它想成一条流水线：

`评测运行 -> 轨迹与奖励 -> 数据导出 -> 三种训练`

---

## 2. 这个工程解决了什么问题

你原来主要依赖远程模型调用，且每次跑完任务只是拿到指标，不方便直接做后训练。  
这版改造后，解决了三个核心问题：

- **模型后端可切换**：保留远程，同时支持本地 Qwen（HF/safetensors）
- **轨迹可训练化**：每题执行过程可导出成结构化数据
- **后训练可落地**：SFT / DPO / RL 三条线都能跑最小闭环

---

## 3. 核心设计原则（为什么这样实现）

### 原则 A：最小侵入

不去大改你已有九类任务逻辑、路由和评测框架，避免破坏已验证流程。

### 原则 B：最大复用

直接复用已有信号：

- `ground_truth`
- `final_validator`
- `search_score`
- `token/cost`

这些本来就是你系统内最可靠的监督来源。

### 原则 C：先跑通，再升级

先做工程稳定的“最小闭环”（小样本可验收），再扩到更复杂训练（更多任务、更大样本、更强 RL）。

---

## 4. 模块结构（你要认识的关键脚本）

下面按“从上到下”列出你最需要看的文件：

### 4.1 运行与轨迹

- `examples/graphwiz/graphwiz_eval_generic.py`  
  评测入口，负责跑样本并触发轨迹导出和奖励计算。
- `examples/graphwiz/trajectory_exporter.py`  
  把执行图和模型 query 历史整理成 `trajectories.jsonl/parquet`。
- `examples/graphwiz/reward_builder.py`  
  根据正确性、合法性、分支质量、token 成本等计算 reward。

### 4.2 数据导出

- `examples/graphwiz/export_sft_data.py`  
  筛选高质量样本，导出 SFT 格式。
- `examples/graphwiz/export_pref_data.py`  
  从同题多轨迹构造 chosen/rejected，导出 DPO 格式。
- `examples/graphwiz/export_rl_data.py`  
  导出 RL 所需 `prompt/response/reward/metadata`。
- `examples/graphwiz/verify_post_training_pipeline.py`  
  检查导出文件是否齐全、字段是否基本正确。

### 4.3 训练脚本（三条线）

- `examples/graphwiz/train_sft_trl.py`：SFT
- `examples/graphwiz/train_dpo_trl.py`：DPO
- `examples/graphwiz/train_rl_weighted_sft.py`：稳定 RL 主线（推荐）
- `examples/graphwiz/train_rl_grpo_trl.py`：可选 GRPO（环境支持时）

### 4.4 模型后端

- `graph_of_thoughts/language_models/factory.py`  
  按配置决定创建远程后端还是本地 HF 后端。
- `graph_of_thoughts/language_models/local_hf_model.py`  
  本地 Qwen 推理实现（支持 safetensors）。

---

## 5. 数据到底长什么样（非常关键）

### 5.1 轨迹文件（`trajectories.jsonl`）

每一行是一道题的一次执行记录，包含：

- 题目与路由信息
- 分支/聚合/修复/最终回答等阶段信息
- correctness / validator / search_score
- token 与 cost
- reward 与奖励分解项

### 5.2 SFT 文件（`graphwiz_sft.jsonl`）

每条包含：

- `instruction`
- `input`
- `output`
- `metadata`

用于“告诉模型标准答案应该怎么答”。

### 5.3 DPO 文件（`graphwiz_pref.jsonl`）

每条包含：

- `instruction`
- `input`
- `chosen`
- `rejected`

用于“告诉模型哪种回答更好”。

### 5.4 RL 文件（`graphwiz_rl.jsonl`）

每条包含：

- `prompt`
- `response`
- `reward`
- `metadata`
- 可选 `step_rewards`

用于“按奖励强弱强化策略”。

---

## 6. 奖励是怎么计算的（直白版）

默认是“加分项 - 扣分项”的规则奖励：

`R = 正确性 + 合法性 + 分支质量 + 分支一致性 - 修复惩罚 - token惩罚`

更具体地说：

- 最终答案正确：大幅加分
- 最终格式合法：加分
- 分支思路质量高：加分
- 分支间一致：加分
- 触发修复流程太多：扣分
- token 过多：扣分

这个设计的意义是：  
不仅鼓励“答对”，也鼓励“答得稳、答得省”。

---

## 7. 三条训练线分别做什么（给小白的理解）

### 7.1 SFT（最容易理解）

把高质量样本当“老师答案”喂给模型，模型学习“应该怎么回答”。

### 7.2 DPO（偏好学习）

给模型看同一题的好回答和差回答，让模型学会偏向好回答。

### 7.3 RL（奖励驱动）

根据 reward 强弱来优化策略。  
当前稳定主线是 `train_rl_weighted_sft.py`，本质是用 reward 当样本权重做策略优化，工程上更稳，适合第一阶段落地。

---

## 8. 一步一步跑通（最小端到端）

下面是你可以照抄执行的最小流程。  
建议第一次都用小样本，先验收“通路”而不是追求最终指标。

### 0）安装依赖

```bash
pip install -U datasets trl peft transformers accelerate
```

### 1）生成轨迹（建议先跑两次，便于 DPO 造 pair）

```bash
python examples/graphwiz/graphwiz_eval_generic.py --source test --subset connectivity --lm_name qwen_local_3b --max_samples 8 --budget 100 --prefer_local 1 --data_root ./data --use_cache 0
python examples/graphwiz/graphwiz_eval_generic.py --source test --subset connectivity --lm_name qwen_local_3b --max_samples 8 --budget 100 --prefer_local 1 --data_root ./data --use_cache 0
```

记两个运行目录：`<run_dir_A>` 和 `<run_dir_B>`。

### 2）导出 SFT / DPO / RL 数据

```bash
python examples/graphwiz/export_sft_data.py --input examples/graphwiz/results/<run_dir_A> --output examples/graphwiz/data/graphwiz_sft.jsonl --min_reward 0.4
python examples/graphwiz/export_pref_data.py --inputs examples/graphwiz/results/<run_dir_A>,examples/graphwiz/results/<run_dir_B> --output examples/graphwiz/data/graphwiz_pref.jsonl --min_reward_gap 0.05
python examples/graphwiz/export_rl_data.py --input examples/graphwiz/results/<run_dir_A> --output examples/graphwiz/data/graphwiz_rl.jsonl --include_step_rewards 1
```

### 3）验证导出结果

```bash
python examples/graphwiz/verify_post_training_pipeline.py --run_dir examples/graphwiz/results/<run_dir_A> --sft_path examples/graphwiz/data/graphwiz_sft.jsonl --pref_path examples/graphwiz/data/graphwiz_pref.jsonl --rl_path examples/graphwiz/data/graphwiz_rl.jsonl
```

### 4）跑三条训练线（都先小样本）

```bash
python examples/graphwiz/train_sft_trl.py --model_name_or_path ./models/Qwen2.5-3B-Instruct --dataset_path examples/graphwiz/data/graphwiz_sft.jsonl --output_dir ./outputs/trl_sft_qwen25_3b_smoke --max_train_samples 32 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1
python examples/graphwiz/train_dpo_trl.py --model_name_or_path ./models/Qwen2.5-3B-Instruct --dataset_path examples/graphwiz/data/graphwiz_pref.jsonl --output_dir ./outputs/trl_dpo_qwen25_3b_smoke --max_train_samples 32 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1
python examples/graphwiz/train_rl_weighted_sft.py --model_name_or_path ./models/Qwen2.5-3B-Instruct --dataset_path examples/graphwiz/data/graphwiz_rl.jsonl --output_dir ./outputs/trl_rl_weighted_qwen25_3b_smoke --max_train_samples 32 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1
```

### 5）可选：GRPO（环境支持才跑）

```bash
python examples/graphwiz/train_rl_grpo_trl.py --model_name_or_path ./models/Qwen2.5-3B-Instruct --dataset_path examples/graphwiz/data/graphwiz_rl.jsonl --output_dir ./outputs/trl_grpo_qwen25_3b_smoke --max_train_samples 32 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1
```

如果报 GRPO 导入错误，直接忽略，继续用稳定主线 `train_rl_weighted_sft.py`。

---

## 9. 新手最常见问题与排查

### 问题 1：DPO 导出后几乎没有 pair

原因通常是两次运行差异太小。  
处理方法：

- 使用两个不同 run 目录，不要同一个目录重复传两次
- 适当调温度或随机性，让同题有更明显质量差异
- 先把 `--min_reward_gap` 调小一点（如 0.01~0.05）

### 问题 2：导出的数据条数很少

可能是：

- `max_samples` 太小
- `min_reward` 太高
- 当前模型在该任务上成功率低

建议先降低筛选门槛，先验证流程，再逐步提质量。

### 问题 3：RL/GRPO 脚本报 TRL 版本问题

这是常见环境问题。  
优先跑稳定 RL 主线：`train_rl_weighted_sft.py`。

### 问题 4：显存不够

先这样做：

- 减小 `max_train_samples`
- `per_device_train_batch_size=1`
- 增大 `gradient_accumulation_steps`
- 降低 `max_seq_length`

---

## 10. 你接下来该怎么扩量

当小闭环跑通后，按这个顺序放大：

1. 从单一 `connectivity` 扩到多子任务  
2. 增加每次 run 的样本数  
3. 累积多次 run 构造更丰富 DPO pair  
4. 固定验证集，观察 SFT/DPO/RL 的真实增益  
5. 再进入更强的 RL 方案（如更完整在线优化）

---

## 11. 一句话总结

这套后训练工程的本质是：  
**把 GoT 运行过程“数据化 + 奖励化 + 训练化”，在不破坏原有系统的前提下，先快速获得可执行、可迭代、可扩展的节点级回答优化闭环。**
