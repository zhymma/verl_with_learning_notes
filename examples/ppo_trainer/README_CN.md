# PPO 训练器 (Proximal Policy Optimization)

> 基于 Actor-Critic 架构的稳定可靠的强化学习训练

---

## 📋 概述

**PPO (Proximal Policy Optimization)** 是最广泛使用的强化学习算法之一，由 OpenAI 于 2017 年提出。它在 LLM 训练中表现出色，特别适合需要精细控制和稳定训练的场景。

### 核心特点

- ✅ **Actor-Critic 架构**：同时训练策略模型（Actor）和价值模型（Critic）
- ✅ **GAE 优势估计**：平衡偏差和方差，获得更稳定的梯度
- ✅ **Clipped Objective**：防止策略更新过大，避免训练崩溃
- ✅ **训练稳定性高**：适合长序列和过程导向任务
- ✅ **KL 散度控制**：防止策略偏离参考模型太远

### 适用场景

| 场景 | 说明 | 推荐度 |
|------|------|--------|
| **对话质量优化** | 需要精细控制生成质量 | ⭐⭐⭐⭐⭐ |
| **长文本生成** | 需要过程级别的优化 | ⭐⭐⭐⭐⭐ |
| **RLHF 训练** | 对齐人类偏好 | ⭐⭐⭐⭐⭐ |
| **数学推理** | 需要 step-by-step 优化 | ⭐⭐⭐⭐ |
| **代码生成** | 可以，但 GRPO 也不错 | ⭐⭐⭐ |
| **快速原型** | 需要训练 Critic，较慢 | ⭐⭐ |

### PPO vs GRPO 对比

| 特性 | PPO | GRPO |
|------|-----|------|
| **Critic 模型** | ✅ 需要 | ❌ 不需要 |
| **GPU 显存** | 更多（Actor + Critic） | 更少（只有 Actor） |
| **训练稳定性** | 更高（有价值函数） | 依赖采样数量 |
| **优势估计** | GAE（时序差分） | 组内均值归一化 |
| **适用场景** | 过程导向、长序列 | 结果导向、数学推理 |
| **训练速度** | 较慢 | 较快 |

---

## 🔧 前置条件

### 硬件要求

```
最低配置：
- GPU: 2 张 24GB GPU（如 RTX 3090）
- 内存: 64GB
- 存储: 100GB

推荐配置：
- GPU: 4-8 张 40GB GPU（如 A100/H100）
- 内存: 128GB+
- 存储: 500GB+
```

### 软件依赖

```bash
# 安装 verl（包含 vLLM）
pip install -e .[test,vllm]

# 或使用 SGLang
pip install -e .[test,sglang]

# 验证安装
python -c "import verl; print(verl.__version__)"
```

### 数据准备

PPO 训练需要准备 Parquet 格式的数据集：

```bash
# 1. 处理 GSM8K 数据集
python examples/data_preprocess/gsm8k.py \
    --local_save_dir ~/data/gsm8k

# 2. 验证数据格式
python learning_notes/01_快速上手/check_data.py ~/data/gsm8k/train.parquet

# 3. 确认文件存在
ls ~/data/gsm8k/
# 输出: train.parquet  test.parquet
```

### 模型准备

确保能访问 HuggingFace 模型：

```bash
# 设置 HuggingFace token（如果需要）
export HF_TOKEN=your_token_here

# 或使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com

# 预下载模型（可选，推荐）
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

---

## 🚀 快速开始

### 最简单的例子（单机 2 GPU）

```bash
# 使用 Gemma 2B 模型在 GSM8K 上训练
bash examples/ppo_trainer/run_gemma.sh

# 自定义参数
bash examples/ppo_trainer/run_gemma.sh \
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=20 \
    actor_rollout_ref.actor.optim.lr=5e-7
```

**预期输出：**
```
[2026-01-28 10:00:00] Initializing Ray...
[2026-01-28 10:00:05] Creating Actor worker pool...
[2026-01-28 10:00:10] Creating Critic worker pool...
[2026-01-28 10:00:15] Creating Rollout worker pool...
[2026-01-28 10:00:20] Starting training...

Epoch 0:
  rollout: 100%|████████| 512/512 [00:30<00:00]
  compute_values: 100%|████████| 512/512 [00:10<00:00]
  train_actor: 100%|████████| 4/4 [00:20<00:00]
  train_critic: 100%|████████| 4/4 [00:20<00:00]
  metrics: reward_mean=0.35, kl=0.002, actor_loss=0.234

✅ 训练完成！模型保存到: ./checkpoints/gemma2b_function_rm/
```

### 使用 Qwen 模型（推荐）

```bash
# Qwen2.5-0.5B（最小模型，快速测试）
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    critic.optim.lr=1e-5 \
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=15

# Qwen2-7B（更强性能）
bash examples/ppo_trainer/run_qwen2-7b_seq_balance.sh
```

### 使用 Reward Model

```bash
# 使用独立的 Reward Model（而非 rule-based reward）
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/hh_rlhf/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    critic.model.path=Qwen/Qwen2.5-7B-Instruct \
    reward_model.enable=True \
    reward_model.path=OpenAssistant/reward-model-deberta-v3-large \
    trainer.n_gpus_per_node=8

# 或参考现有脚本
bash examples/ppo_trainer/run_qwen2-7b_rm.sh
```

---

## 📖 详细配置

### 核心配置参数

PPO 训练的配置通过 Hydra 管理，可以在命令行覆盖 YAML 配置。

#### 1. 算法配置 (`algorithm.*`)

```yaml
algorithm:
  adv_estimator: gae              # 必须设置为 gae（PPO 算法）
  gamma: 0.99                     # 折扣因子（未来奖励权重）
  lam: 0.95                       # GAE lambda（bias-variance tradeoff）
  use_kl_in_reward: False         # 是否在 reward 中加 KL penalty
  kl_penalty: kl                  # KL penalty 类型: kl, abs, mse
  kl_ctrl:
    type: fixed                   # KL 控制器: fixed 或 adaptive
    kl_coef: 0.001                # KL 系数（初始值）
```

**参数详解：**

- **gamma（折扣因子）**：控制未来奖励的权重
  - `0.99`（默认）：重视长期奖励
  - `0.95`：更注重近期奖励
  - `1.0`：Monte Carlo，无折扣

- **lam（GAE lambda）**：平衡偏差和方差
  - `0.95`（默认）：标准选择
  - `1.0`：低偏差，高方差
  - `0.0`：高偏差，低方差（1-step TD）

#### 2. 数据配置 (`data.*`)

```yaml
data:
  train_files: ~/data/gsm8k/train.parquet    # 训练数据路径
  val_files: ~/data/gsm8k/test.parquet       # 验证数据路径
  train_batch_size: 512                      # 全局 batch size（prompt 数量）
  max_prompt_length: 1024                    # 最大 prompt 长度
  max_response_length: 512                   # 最大 response 长度
  filter_overlong_prompts: True              # 过滤过长的 prompt
  truncation: error                          # 截断策略: error, left, right
```

**重要说明：**

- `train_batch_size`：决定每轮生成多少条数据
  - 小模型（<7B）：256-512
  - 大模型（7B-70B）：128-256
  - 超大模型（>70B）：64-128

- **响应数量** = `train_batch_size × rollout.n`
  - PPO 通常 `rollout.n=1`（每个 prompt 一条响应）
  - GRPO 需要 `rollout.n>1`（每个 prompt 多条响应）

#### 3. Actor 配置 (`actor_rollout_ref.actor.*`)

```yaml
actor_rollout_ref:
  actor:
    # 优化器配置
    optim:
      lr: 1e-6                          # 学习率
      weight_decay: 0.01                # 权重衰减
      warmup_steps: 0                   # warmup 步数

    # PPO 参数
    ppo_mini_batch_size: 128            # PPO mini-batch 大小
    ppo_micro_batch_size_per_gpu: 4     # 每张 GPU 的 micro-batch
    ppo_epochs: 1                       # PPO 更新轮数
    clip_ratio: 0.2                     # PPO clipping 范围

    # KL 散度控制
    use_kl_loss: False                  # 是否使用 KL loss
    kl_loss_coef: 0.001                 # KL loss 系数
    kl_loss_type: kl                    # KL 计算类型

    # FSDP 配置
    fsdp_config:
      param_offload: False              # 参数卸载到 CPU
      optimizer_offload: False          # 优化器卸载到 CPU
      gradient_checkpointing: True      # 梯度检查点（省显存）
```

**关键参数：**

- **ppo_mini_batch_size**：决定 PPO 更新的 batch 大小
  - 必须能被 `train_batch_size × rollout.n` 整除
  - 示例：`train_batch_size=512, rollout.n=1 → ppo_mini_batch_size=128`
  - 更大的值 → 更稳定，但更慢

- **ppo_epochs**：对同一批数据更新几次
  - `1`（默认）：每批数据只用一次（on-policy）
  - `2-4`：复用数据，提高样本效率
  - 过大可能导致过拟合

- **clip_ratio**：PPO clipping 的范围
  - `0.2`（默认）：标准选择
  - 更小（0.1）：更保守的更新
  - 更大（0.3）：更激进的更新

#### 4. Critic 配置 (`critic.*`)

```yaml
critic:
  model:
    path: Qwen/Qwen2.5-7B-Instruct      # Critic 模型路径（通常与 Actor 相同）
    enable_gradient_checkpointing: True # 梯度检查点

  optim:
    lr: 1e-5                            # Critic 学习率（通常比 Actor 大 10x）

  ppo_mini_batch_size: 128              # Critic mini-batch（通常与 Actor 相同）
  ppo_micro_batch_size_per_gpu: 4       # 每张 GPU 的 micro-batch
  ppo_epochs: 1                         # Critic 更新轮数（默认同 Actor）
```

**重要：**

- Critic 学习率通常是 Actor 的 **10 倍**
  - Actor: `1e-6`, Critic: `1e-5`
  - Actor: `5e-7`, Critic: `5e-6`

- Critic 模型通常与 Actor 使用 **相同架构**
  - 初始化：从预训练模型加载
  - 训练：value head 会被添加到模型上

#### 5. Rollout 配置 (`actor_rollout_ref.rollout.*`)

```yaml
actor_rollout_ref:
  rollout:
    name: vllm                          # 推理引擎: vllm, sglang, hf
    tensor_model_parallel_size: 2       # TP（张量并行）大小
    gpu_memory_utilization: 0.4         # GPU 显存利用率

    # 生成参数
    temperature: 1.0                    # 采样温度
    top_p: 1.0                          # nucleus sampling
    top_k: -1                           # top-k sampling
    n: 1                                # 每个 prompt 生成几条（PPO=1）

    log_prob_micro_batch_size_per_gpu: 4  # 计算 log_prob 的 batch size
```

**Rollout Engine 选择：**

| 引擎 | 速度 | 显存 | 功能 | 推荐场景 |
|------|------|------|------|----------|
| **vLLM** | ⭐⭐⭐⭐⭐ | 中等 | 全面 | 首选 |
| **SGLang** | ⭐⭐⭐⭐⭐ | 较低 | 多轮对话 | Agent、多轮 |
| **TRT-LLM** | ⭐⭐⭐⭐⭐ | 最低 | 高性能 | 生产环境 |
| **HF** | ⭐⭐ | 较高 | 基础 | 调试 |

#### 6. Trainer 配置 (`trainer.*`)

```yaml
trainer:
  n_gpus_per_node: 2                    # 每个节点的 GPU 数量
  nnodes: 1                             # 节点数量
  total_epochs: 15                      # 总训练轮数
  save_freq: 5                          # 每 N 轮保存一次
  test_freq: 5                          # 每 N 轮测试一次

  # 日志配置
  logger: '["console","wandb"]'         # 日志工具
  project_name: verl_ppo                # 项目名称
  experiment_name: qwen2.5_gsm8k        # 实验名称

  # Critic warmup
  critic_warmup: 0                      # Critic 预热步数
```

---

## 💡 运行示例

### 示例 1：Qwen2.5-0.5B 快速测试

```bash
# 最小配置，适合快速验证
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    critic.ppo_micro_batch_size=2 \
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=15 \
    trainer.logger=console

# 预期结果（15 epochs 后）：
# - 训练时间: ~30 分钟
# - GSM8K 准确率: ~56.7%（从预训练的 36.4% 提升）
# - 模型保存: ./checkpoints/
```

### 示例 2：Qwen2-7B 完整训练

```bash
# 推荐配置，获得更好效果
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    critic.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    critic.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=20

# 或直接使用预设脚本
bash examples/ppo_trainer/run_qwen2-7b_seq_balance.sh
```

### 示例 3：使用 Reward Model（RLHF）

```bash
# 人类偏好对齐训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/hh_rlhf/train.parquet \
    data.val_files=$HOME/data/hh_rlhf/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    critic.model.path=Qwen/Qwen2.5-7B-Instruct \
    reward_model.enable=True \
    reward_model.path=OpenAssistant/reward-model-deberta-v3-large \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.n_gpus_per_node=8

# 参考脚本
bash examples/ppo_trainer/run_qwen2-7b_rm.sh
```

### 示例 4：启用 KL Loss（推荐）

```bash
# 使用 KL Loss 而非 KL Reward
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    critic.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=kl \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=8

# KL Loss 优势：
# ✅ 梯度更直接，训练更稳定
# ✅ 不影响 reward 设计
# ✅ 更容易调参
```

### 示例 5：多节点分布式训练

```bash
# 使用 4 个节点，每个节点 8 GPU（总共 32 GPU）
# 节点 0 (主节点)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.train_batch_size=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2-72B-Instruct \
    critic.model.path=Qwen/Qwen2-72B-Instruct \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.node_rank=0 \
    trainer.master_addr=192.168.1.100 \
    trainer.master_port=29500

# 节点 1-3（工作节点）
# 将 trainer.node_rank 设置为 1, 2, 3
# master_addr 保持一致
```

### 示例 6：使用 Megatron-LM 后端

```bash
# 超大模型训练（使用张量并行 + 流水线并行）
bash examples/ppo_trainer/run_qwen2-7b_math_megatron.sh

# 自定义 Megatron 配置
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron_config.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron_config.pipeline_model_parallel_size=2 \
    critic.strategy=megatron \
    critic.megatron_config.tensor_model_parallel_size=4 \
    trainer.n_gpus_per_node=8
```

---

## 🐛 常见问题

### Q1: OOM（显存不足）怎么办？

**症状：**
```
CUDA out of memory. Tried to allocate XXX GiB
```

**解决方案：**

```bash
# 方法 1: 减小 batch size
data.train_batch_size=256  # 从 512 减小到 256
actor_rollout_ref.actor.ppo_mini_batch_size=64  # 相应减小

# 方法 2: 减小 micro_batch_size
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2  # 从 4 减小到 2
critic.ppo_micro_batch_size_per_gpu=2

# 方法 3: 启用梯度检查点
actor_rollout_ref.actor.fsdp_config.gradient_checkpointing=True
critic.model.enable_gradient_checkpointing=True

# 方法 4: 启用参数卸载（会变慢）
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# 方法 5: 减小 rollout GPU 显存占用
actor_rollout_ref.rollout.gpu_memory_utilization=0.3  # 从 0.4 减小

# 方法 6: 增大张量并行
actor_rollout_ref.rollout.tensor_model_parallel_size=4  # 从 2 增大到 4
```

### Q2: 训练不稳定，loss 爆炸怎么办？

**症状：**
```
actor_loss: nan
或
actor_loss: 1e10
```

**解决方案：**

```bash
# 方法 1: 降低学习率
actor_rollout_ref.actor.optim.lr=5e-7  # 从 1e-6 降低
critic.optim.lr=5e-6

# 方法 2: 减小 clip_ratio
actor_rollout_ref.actor.clip_ratio=0.1  # 从 0.2 减小

# 方法 3: 启用梯度裁剪
actor_rollout_ref.actor.grad_clip=1.0
critic.grad_clip=1.0

# 方法 4: 增加 KL 约束
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01  # 增大系数

# 方法 5: 减小 ppo_epochs
actor_rollout_ref.actor.ppo_epochs=1  # 避免过度优化
```

### Q3: Reward 始终为 0 怎么办？

**检查步骤：**

```bash
# 1. 检查数据格式
python learning_notes/02_数据准备/data_quality_check.py ~/data/gsm8k/train.parquet

# 2. 检查 reward_model 字段
python -c "
import pandas as pd
df = pd.read_parquet('~/data/gsm8k/train.parquet')
print(df.iloc[0]['reward_model'])
# 应该输出: {'style': 'rule', 'ground_truth': '...'}
"

# 3. 检查 data_source 是否正确
python -c "
import pandas as pd
df = pd.read_parquet('~/data/gsm8k/train.parquet')
print(df.iloc[0]['data_source'])
# 应该输出: openai/gsm8k
"

# 4. 确认 Reward 函数已注册
# 查看 verl/trainer/ppo/reward_score/gsm8k.py
# 确保 data_source 匹配

# 5. 查看训练日志中的 reward 计算
# 应该看到类似：
# [RewardManager] Computing rewards for data_source=openai/gsm8k
# [GSM8K Reward] Correct: 123/512, Accuracy: 0.24
```

### Q4: Critic loss 不下降怎么办？

**可能原因：**

```bash
# 原因 1: Critic 学习率太小
critic.optim.lr=1e-5  # 应该是 Actor 的 10 倍

# 原因 2: Critic warmup 不足
trainer.critic_warmup=10  # 先单独训练 Critic

# 原因 3: Reward signal 太弱
# 检查 reward 的分布
# 应该看到 reward_mean 在变化，不是始终为 0

# 原因 4: batch size 太小
data.train_batch_size=512  # 增大 batch size
```

### Q5: 如何选择 GAE 的 gamma 和 lam 参数？

**推荐配置：**

```bash
# 标准配置（适合大多数任务）
algorithm.gamma=0.99
algorithm.lam=0.95

# 短序列任务（如分类）
algorithm.gamma=0.95
algorithm.lam=0.9

# 长序列任务（如长文本生成）
algorithm.gamma=0.99
algorithm.lam=0.97

# 调试技巧：
# - 如果训练不稳定（高方差）：减小 lam（如 0.9）
# - 如果收敛慢（高偏差）：增大 lam（如 0.98）
```

### Q6: vLLM 和 SGLang 如何选择？

**选择指南：**

```bash
# 使用 vLLM（推荐）
actor_rollout_ref.rollout.name=vllm
# 优势：成熟稳定，性能优秀，社区支持好
# 劣势：多轮对话支持一般

# 使用 SGLang
actor_rollout_ref.rollout.name=sglang
# 优势：多轮对话支持更好，Agent RL 推荐
# 劣势：较新，文档相对少

# 如果任务是单轮 → vLLM
# 如果任务是多轮/Agent → SGLang
```

### Q7: 如何监控训练进度？

**使用 TensorBoard：**

```bash
# 训练时启用 tensorboard
trainer.logger='["console","tensorboard"]'

# 在另一个终端启动 TensorBoard
tensorboard --logdir ./runs/

# 访问 http://localhost:6006
```

**使用 W&B（推荐）：**

```bash
# 1. 安装 wandb
pip install wandb

# 2. 登录
wandb login

# 3. 启用 wandb
trainer.logger='["console","wandb"]' \
trainer.project_name='my_ppo_project' \
trainer.experiment_name='qwen2.5_gsm8k_v1'

# 4. 查看训练曲线
# 访问 https://wandb.ai/<your-username>/my_ppo_project
```

**关键指标：**

```yaml
# 需要关注的指标：
- reward_mean: 平均奖励（应该上升）
- actor_loss: Actor 损失（应该下降）
- critic_loss: Critic 损失（应该下降）
- kl_divergence: KL 散度（应该保持在合理范围，如 < 0.1）
- ppo_ratio: PPO ratio（应该在 [0.8, 1.2] 之间）
- grad_norm: 梯度范数（不应该爆炸）

# 健康的训练曲线：
# - reward_mean: 稳步上升
# - kl_divergence: 缓慢增长，但不超过 0.1-0.2
# - actor_loss/critic_loss: 稳步下降
```

---

## 📊 性能基准

### Qwen2.5-0.5B on GSM8K

```
预训练模型准确率: 36.4%
PPO 训练后准确率: 56.7%
训练时间: ~30 分钟（2x RTX 3090）
配置: batch_size=256, epochs=15

命令:
bash examples/ppo_trainer/run_gemma.sh \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct
```

### Qwen2-7B on GSM8K

```
预训练模型准确率: ~65%
PPO 训练后准确率: ~75%
训练时间: ~2 小时（8x A100）
配置: batch_size=512, epochs=20

命令:
bash examples/ppo_trainer/run_qwen2-7b_seq_balance.sh
```

---

## 🔗 参考资料

### 官方文档

- [PPO 算法原理](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [verl 文档](https://verl.readthedocs.io/)

### 学习笔记

- [03_RL算法/PPO_详解.md](../../learning_notes/03_RL算法/PPO_详解.md) - PPO 算法源码级详解
- [03_RL算法/03_RL算法概览.md](../../learning_notes/03_RL算法/03_RL算法概览.md) - 算法对比与选择
- [01_快速上手/ray_trainer_详解.md](../../learning_notes/01_快速上手/ray_trainer_详解.md) - 训练流程详解

### 相关示例

- `examples/grpo_trainer/` - GRPO 训练示例（无 Critic）
- `examples/sft/` - SFT 训练（PPO 的前置步骤）
- `examples/data_preprocess/` - 数据预处理

### 论文

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - InstructGPT (PPO + RLHF)
- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) - PPO 在 LLM 的早期应用

---

**创建时间**: 2026-01-28
**适用版本**: verl v0.2+
**维护者**: verl team
