# GRPO 训练器 (Group Relative Policy Optimization)

> 无需 Critic 模型的高效强化学习训练

---

## 📋 概述

**GRPO (Group Relative Policy Optimization)** 是 DeepSeek 在 2024 年提出的创新强化学习算法。它通过**组内相对比较**消除了对 Critic 模型的需求，大幅降低了训练成本和显存占用。

### 核心特点

- ✅ **无需 Critic**：不需要训练价值函数模型，节省 50% 显存
- ✅ **组采样**：为每个 prompt 生成多个响应，组内比较
- ✅ **相对优势**：使用组内均值归一化，自动形成 baseline
- ✅ **训练高效**：更快的训练速度，更少的 GPU 需求
- ✅ **适合数学推理**：在 GSM8K、MATH 等任务上表现出色

### 适用场景

| 场景 | 说明 | 推荐度 |
|------|------|--------|
| **数学推理** | GSM8K、MATH 等结果导向任务 | ⭐⭐⭐⭐⭐ |
| **代码生成** | 通过测试即可判断正确性 | ⭐⭐⭐⭐⭐ |
| **快速原型** | 不需要训练 Critic，快速验证想法 | ⭐⭐⭐⭐⭐ |
| **显存受限** | GPU 显存不足，无法同时训练 Actor + Critic | ⭐⭐⭐⭐⭐ |
| **问答任务** | 明确的对错标准 | ⭐⭐⭐⭐ |
| **长文本生成** | 需要过程级优化，GRPO 效果一般 | ⭐⭐ |
| **对话质量** | 需要精细控制，建议用 PPO | ⭐⭐ |

### GRPO vs PPO 对比

| 特性 | GRPO | PPO |
|------|------|-----|
| **Critic 模型** | ❌ 不需要 | ✅ 需要 |
| **GPU 显存** | 更少（只有 Actor） | 更多（Actor + Critic） |
| **训练速度** | 更快 | 较慢 |
| **Baseline** | 组内样本均值 | Critic 的 V(s) |
| **优势估计** | 归一化的相对 reward | GAE（时序差分） |
| **每个 prompt 采样数** | n > 1（通常 4-8） | n = 1 |
| **适用场景** | 结果导向（数学、代码） | 过程导向（对话、长文本） |
| **训练稳定性** | 依赖采样数 n | 更稳定 |

---

## 🔧 前置条件

### 硬件要求

GRPO 比 PPO 的硬件需求更低（无需 Critic）：

```
最低配置：
- GPU: 1 张 24GB GPU（如 RTX 3090）
- 内存: 32GB
- 存储: 50GB

推荐配置：
- GPU: 2-4 张 40GB GPU（如 A100）
- 内存: 64GB+
- 存储: 200GB+
```

### 软件依赖

```bash
# 安装 verl（包含 vLLM）
pip install -e .[test,vllm]

# 或使用 SGLang（多轮对话推荐）
pip install -e .[test,sglang]

# 验证安装
python -c "import verl; print(verl.__version__)"
```

### 数据准备

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

---

## 🚀 快速开始

### 最简单的例子（单机 2 GPU）

```bash
# 使用 Qwen2.5-3B 在 GSM8K 上快速训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=15

# 注意 GRPO 的关键配置：
# - algorithm.adv_estimator=grpo  （设置为 GRPO 算法）
# - actor_rollout_ref.rollout.n=4  （每个 prompt 生成 4 条响应）
# - actor_rollout_ref.actor.use_kl_loss=True  （使用 KL loss 而非 KL reward）
# - 没有 critic 配置！
```

**预期输出：**
```
[2026-01-28 10:00:00] Initializing Ray...
[2026-01-28 10:00:05] Creating Actor worker pool...
[2026-01-28 10:00:10] Creating Rollout worker pool...
[2026-01-28 10:00:15] Starting GRPO training...

Epoch 0:
  rollout: 100%|████████| 256/256 [00:40<00:00]  # 256 prompts × 4 responses = 1024 samples
  train_actor: 100%|████████| 4/4 [00:15<00:00]
  metrics: reward_mean=0.28, kl=0.003, actor_loss=0.187, advantage_std=1.0

✅ 训练完成！模型保存到: ./checkpoints/qwen2.5-3b_grpo/
```

### 使用推荐配置（Qwen3-8B）

```bash
# 直接运行预设脚本
bash examples/grpo_trainer/run_qwen3-8b.sh

# 或自定义参数
bash examples/grpo_trainer/run_qwen3-8b.sh \
    data.train_batch_size=512 \
    actor_rollout_ref.rollout.n=8 \
    trainer.total_epochs=20
```

### 使用 LoRA 训练（节省显存）

```bash
# LoRA 训练（显存占用更少）
bash examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh

# 或自定义 LoRA 配置
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.lora.enable=True \
    actor_rollout_ref.actor.lora.r=16 \
    actor_rollout_ref.actor.lora.alpha=32 \
    actor_rollout_ref.actor.lora.target_modules='["q_proj","v_proj"]' \
    actor_rollout_ref.rollout.n=4 \
    trainer.n_gpus_per_node=2
```

---

## 📖 详细配置

### GRPO 核心配置

#### 1. 算法配置 (`algorithm.*`)

```yaml
algorithm:
  adv_estimator: grpo             # 必须设置为 grpo
  norm_adv_by_std_in_grpo: True   # 使用标准差归一化优势值（推荐）
  use_kl_in_reward: False         # GRPO 不使用 KL reward（用 KL loss 代替）
```

**重要：**
- GRPO **不使用** `algorithm.gamma` 和 `algorithm.lam`（这是 GAE 的参数）
- GRPO **不使用** `algorithm.use_kl_in_reward`（使用 KL loss 代替）

#### 2. 组采样配置 (`actor_rollout_ref.rollout.n`)

这是 GRPO 最关键的参数：

```yaml
actor_rollout_ref:
  rollout:
    n: 4                          # 每个 prompt 生成几条响应（组大小）
```

**如何选择 n：**

| n 值 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| **n=2** | 快速实验 | 速度快 | 统计不稳定 |
| **n=4** | 标准配置（推荐） | 平衡速度和稳定性 | - |
| **n=8** | 高质量训练 | 更稳定的梯度 | 更慢，显存占用更多 |
| **n=16** | 研究级训练 | 最稳定 | 非常慢 |

**计算公式：**
```
总响应数 = train_batch_size × n

示例：
train_batch_size=256, n=4 → 1024 条响应
train_batch_size=128, n=8 → 1024 条响应
```

#### 3. Actor 配置（无 Critic！）

```yaml
actor_rollout_ref:
  actor:
    # 优化器配置
    optim:
      lr: 1e-6                    # 学习率（GRPO 可以稍高于 PPO）

    # GRPO 特有参数
    ppo_mini_batch_size: 256      # mini-batch 大小
    ppo_epochs: 1                 # 训练轮数
    clip_ratio: 0.2               # Clipping 范围
    loss_agg_mode: token-mean     # 损失聚合方式（推荐）

    # KL 散度控制（必须启用）
    use_kl_loss: True             # 必须设置为 True
    kl_loss_coef: 0.001           # KL 系数
    kl_loss_type: kl              # KL 类型
```

**loss_agg_mode 详解：**

| 模式 | 计算方式 | 适用场景 |
|------|---------|---------|
| `token-mean` | 所有 token 平均（推荐） | 标准选择，稳定 |
| `seq-mean-token-sum` | 先按序列求和，再平均 | 长序列任务 |
| `seq-mean-token-mean` | 序列级别平均（原论文） | 短序列任务 |

**原论文使用 `seq-mean-token-mean`，但 verl 推荐 `token-mean` 以获得更好的稳定性。**

#### 4. 数据配置

```yaml
data:
  train_files: ~/data/gsm8k/train.parquet
  val_files: ~/data/gsm8k/test.parquet
  train_batch_size: 256           # Prompt 数量
  max_prompt_length: 1024
  max_response_length: 512
```

**关键：**
- `train_batch_size` 是 **prompt 数量**
- 实际训练的响应数 = `train_batch_size × rollout.n`
- 示例：`batch_size=256, n=4 → 1024 条响应`

#### 5. 完整配置示例

```bash
python3 -m verl.trainer.main_ppo \
    # ========== 算法配置 ==========
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \

    # ========== 数据配置 ==========
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \

    # ========== 模型配置 ==========
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \

    # ========== Rollout 配置（组采样）==========
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \

    # ========== Actor 配置 ==========
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \

    # ========== KL 控制（GRPO 必须）==========
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=kl \

    # ========== Trainer 配置 ==========
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=20 \
    trainer.save_freq=5 \
    trainer.logger='["console","wandb"]'
```

---

## 💡 运行示例

### 示例 1：Qwen2.5-3B 快速测试

```bash
# 最小配置，2 GPU，30 分钟完成
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=10 \
    trainer.logger=console

# 预期结果：
# - 训练时间: ~30 分钟
# - GSM8K 准确率: 60-65%
# - 显存占用: ~18GB/GPU
```

### 示例 2：Qwen2-7B 标准训练

```bash
# 标准配置，4 GPU
bash examples/grpo_trainer/run_qwen2-7b_math.sh

# 或自定义
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.train_batch_size=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.use_kl_loss=True \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=20

# 预期结果：
# - 训练时间: ~1.5 小时
# - GSM8K 准确率: 70-75%
```

### 示例 3：Qwen3-8B 高质量训练

```bash
# 推荐配置，8 GPU
bash examples/grpo_trainer/run_qwen3-8b.sh

# 关键配置：
# - train_batch_size=512
# - rollout.n=8（更多采样）
# - total_epochs=30

# 预期结果：
# - 训练时间: ~3 小时
# - GSM8K 准确率: 75-80%
```

### 示例 4：使用 Megatron-LM 训练超大模型

```bash
# DeepSeek-Math-671B（需要多节点）
bash examples/grpo_trainer/run_deepseek671b_math_megatron_96gb.sh

# 或 Qwen3-235B
bash examples/grpo_trainer/run_qwen3-235b_megatron_96gb.sh

# 配置要点：
# - Megatron-LM backend
# - tensor_model_parallel_size=8
# - pipeline_model_parallel_size=8
# - 需要 64+ GPU
```

### 示例 5：多模态 VLM 训练

```bash
# Qwen2.5-VL-7B（视觉语言模型）
bash examples/grpo_trainer/run_qwen2_5_vl-7b.sh

# 或 Qwen3-VL-8B
bash examples/grpo_trainer/run_qwen3_vl-8b-megatron.sh

# 需要准备包含图像的数据
# 参考 examples/data_preprocess/geo3k.py
```

### 示例 6：DrGRPO（减少长度偏差）

```bash
# DrGRPO 配置（推荐用于长 CoT 任务）
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.loss_scale_factor=512 \
    actor_rollout_ref.actor.use_kl_loss=False \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.rollout.n=4 \
    trainer.n_gpus_per_node=4

# DrGRPO 关键区别：
# - loss_agg_mode=seq-mean-token-sum-norm（取消序列级平均）
# - norm_adv_by_std_in_grpo=False（取消标准差归一化）
# - use_kl_loss=False（不使用 KL loss）
# - loss_scale_factor=512（固定归一化常数）
```

---

## 🎯 GRPO 核心原理

### 优势计算公式

**传统 GRPO（组内归一化）：**

```python
# 对每个 prompt，生成 n 个响应
responses = [resp_1, resp_2, ..., resp_n]
rewards = [r_1, r_2, ..., r_n]

# 计算组内统计量
mean = sum(rewards) / n
std = sqrt(sum((r_i - mean)^2) / n)

# 归一化优势值
advantage_i = (r_i - mean) / std

# 使用 advantage_i 更新策略
```

**DrGRPO（全局归一化）：**

```python
# 不使用组内标准差
advantage_i = r_i - mean

# 使用固定的全局归一化因子
loss_i = -advantage_i * log_ratio_i / scale_factor
```

### 与 PPO 的对比

**PPO（GAE）：**
```python
# 需要 Critic 模型
values = critic_model(states)

# 计算 TD-error
delta_t = reward_t + gamma * value_{t+1} - value_t

# 计算 GAE 优势
advantage_t = delta_t + gamma * lambda * advantage_{t+1}
```

**GRPO：**
```python
# 不需要 Critic
# 只需要同一组内的其他响应

# 组内归一化
advantage_i = (reward_i - group_mean) / group_std
```

**关键区别：**
- PPO：依赖 Critic 提供 baseline
- GRPO：依赖组内其他样本提供 baseline

---

## 🐛 常见问题

### Q1: GRPO vs PPO 如何选择？

**选择 GRPO 的场景：**
```
✅ 数学推理任务（GSM8K、MATH）
✅ 代码生成任务（HumanEval、MBPP）
✅ 明确的对错标准（问答、选择题）
✅ GPU 显存受限
✅ 快速实验原型
✅ 结果导向的任务
```

**选择 PPO 的场景：**
```
✅ 对话质量优化
✅ 长文本生成
✅ 需要过程级优化
✅ 复杂的 reward shaping
✅ RLHF 人类偏好对齐
✅ 需要更稳定的训练
```

**决策树：**
```
任务是否有明确的对错标准？
├─ Yes → 是否是数学/代码任务？
│   ├─ Yes → 使用 GRPO（推荐）
│   └─ No → GRPO 或 PPO 都可以
└─ No → 是否需要过程级优化？
    ├─ Yes → 使用 PPO
    └─ No → 使用 GRPO（更快）
```

### Q2: rollout.n 应该设置为多少？

**推荐配置：**

```bash
# 快速实验（不推荐用于最终训练）
actor_rollout_ref.rollout.n=2

# 标准配置（推荐）
actor_rollout_ref.rollout.n=4

# 高质量训练
actor_rollout_ref.rollout.n=8

# 研究级别
actor_rollout_ref.rollout.n=16
```

**权衡考虑：**

| n | 优势 | 劣势 | 适用场景 |
|---|------|------|---------|
| 2 | 快速 | 不稳定 | 调试 |
| 4 | 平衡 | - | 生产环境（推荐） |
| 8 | 稳定 | 慢 2x | 高质量训练 |
| 16 | 非常稳定 | 慢 4x | 研究 |

**实验建议：**
```bash
# 第一次尝试：n=4
# 如果训练不稳定：增大到 n=8
# 如果速度太慢：减小到 n=2（但可能需要调整其他参数）
```

### Q3: Advantage 标准差为 0 怎么办？

**症状：**
```
Warning: advantage std is 0, setting to 1
或
All rewards in the group are the same
```

**原因：**
- 组内所有响应的 reward 完全相同
- 可能是 reward 函数设计问题

**解决方案：**

```bash
# 方法 1: 增大 rollout.n（增加多样性）
actor_rollout_ref.rollout.n=8  # 从 4 增大到 8

# 方法 2: 调整采样参数（增加随机性）
actor_rollout_ref.rollout.temperature=1.0  # 增大温度
actor_rollout_ref.rollout.top_p=0.95  # 不要设置为 1.0

# 方法 3: 检查 reward 函数
# 确保 reward 不是二元的（0 或 1），应该有细粒度的评分

# 方法 4: 使用 reward shaping
# 添加中间步骤的奖励，详见 learning_notes/04_Reward设计/
```

### Q4: GRPO 训练不稳定怎么办？

**症状：**
```
- Loss 震荡严重
- Reward 不上升
- KL 散度爆炸
```

**解决方案：**

```bash
# 方法 1: 增大 rollout.n（更稳定的梯度）
actor_rollout_ref.rollout.n=8  # 从 4 增大

# 方法 2: 降低学习率
actor_rollout_ref.actor.optim.lr=5e-7  # 从 1e-6 降低

# 方法 3: 增大 KL 约束
actor_rollout_ref.actor.kl_loss_coef=0.01  # 从 0.001 增大

# 方法 4: 减小 clip_ratio
actor_rollout_ref.actor.clip_ratio=0.1  # 从 0.2 减小

# 方法 5: 减小 batch size
data.train_batch_size=128  # 从 256 减小

# 方法 6: 使用 DrGRPO
algorithm.norm_adv_by_std_in_grpo=False
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
```

### Q5: GRPO 的显存占用如何？

**显存占用对比（Qwen2-7B，单 GPU）：**

```
PPO:
- Actor 训练: ~20GB
- Critic 训练: ~20GB
- Rollout: ~15GB
- 总计（峰值）: ~40GB
- 需要: 2-4 张 40GB GPU

GRPO:
- Actor 训练: ~20GB
- Rollout: ~15GB
- 总计（峰值）: ~25GB
- 需要: 1-2 张 40GB GPU

节省: ~40%
```

**优化技巧：**

```bash
# 1. 减小 rollout GPU 显存占用
actor_rollout_ref.rollout.gpu_memory_utilization=0.3

# 2. 启用梯度检查点
actor_rollout_ref.actor.fsdp_config.gradient_checkpointing=True

# 3. 使用 LoRA
actor_rollout_ref.actor.lora.enable=True
actor_rollout_ref.actor.lora.r=16

# 4. 增大张量并行
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

### Q6: loss_agg_mode 如何选择？

**三种模式对比：**

```python
# 假设一个 batch 有 2 条序列
responses = [
    [token1, token2, token3],  # 长度 3
    [token1, token2, token3, token4, token5]  # 长度 5
]
advantages = [
    [adv1, adv2, adv3],
    [adv4, adv5, adv6, adv7, adv8]
]
ratios = [
    [ratio1, ratio2, ratio3],
    [ratio4, ratio5, ratio6, ratio7, ratio8]
]

# 模式 1: token-mean（推荐）
loss = mean([-adv1*ratio1, -adv2*ratio2, ..., -adv8*ratio8])
# 所有 token 平等对待

# 模式 2: seq-mean-token-sum
seq_loss_1 = sum([-adv1*ratio1, -adv2*ratio2, -adv3*ratio3])
seq_loss_2 = sum([-adv4*ratio4, ..., -adv8*ratio8])
loss = mean([seq_loss_1, seq_loss_2])
# 先按序列求和，再平均（偏向长序列）

# 模式 3: seq-mean-token-mean（原论文）
seq_loss_1 = mean([-adv1*ratio1, -adv2*ratio2, -adv3*ratio3])
seq_loss_2 = mean([-adv4*ratio4, ..., -adv8*ratio8])
loss = mean([seq_loss_1, seq_loss_2])
# 每个序列平等对待（可能不稳定）
```

**推荐：**
- **一般任务**：`token-mean`（默认，最稳定）
- **长序列任务**：`seq-mean-token-sum`
- **短序列任务**：`seq-mean-token-mean`（原论文）
- **减少长度偏差**：`seq-mean-token-sum-norm`（DrGRPO）

### Q7: GRPO 需要 Critic warmup 吗？

**不需要！**

```bash
# GRPO 没有 Critic，所以不需要 warmup
trainer.critic_warmup=0  # 保持为 0

# 但如果想让 Actor 先适应数据，可以：
# 1. 先做 SFT
bash examples/sft/run_qwen2_5_7b.sh

# 2. 然后从 SFT checkpoint 开始 GRPO
actor_rollout_ref.model.path=./checkpoints/sft_qwen2.5_7b/
```

### Q8: 如何监控 GRPO 训练？

**关键指标：**

```yaml
# 1. Reward 相关
reward_mean: 平均奖励（应该上升）
reward_std: 奖励标准差（组内差异）

# 2. Advantage 相关
advantage_mean: 应该接近 0（归一化后）
advantage_std: 应该接近 1（归一化后）

# 3. 损失相关
actor_loss: Actor 损失（应该下降）
kl_divergence: KL 散度（应该 < 0.1）

# 4. PPO 相关
ppo_ratio_mean: 应该接近 1.0
ppo_ratio_clipped: 被 clip 的比例（应该 < 30%）

# 5. 梯度相关
grad_norm: 梯度范数（不应该爆炸）
```

**健康的训练曲线：**
```
Epoch 0:  reward_mean=0.25, kl=0.003
Epoch 5:  reward_mean=0.45, kl=0.008
Epoch 10: reward_mean=0.60, kl=0.015
Epoch 15: reward_mean=0.68, kl=0.020

✅ reward 稳步上升
✅ kl 缓慢增长但不爆炸
```

**异常情况：**
```
Epoch 0:  reward_mean=0.25, kl=0.003
Epoch 5:  reward_mean=0.28, kl=0.150  ❌ KL 太大
Epoch 10: reward_mean=0.20, kl=0.250  ❌ Reward 下降

→ 学习率太高，或 KL 约束太弱
→ 减小 lr 或增大 kl_loss_coef
```

---

## 📊 性能基准

### Qwen2.5-3B on GSM8K

```
预训练模型准确率: ~50%
GRPO 训练后准确率: ~65%
训练时间: ~30 分钟（2x A100）
配置: batch_size=128, n=4, epochs=10

命令:
bash examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh
```

### Qwen2-7B on GSM8K

```
预训练模型准确率: ~65%
GRPO 训练后准确率: ~78%
训练时间: ~1.5 小时（4x A100）
配置: batch_size=512, n=4, epochs=20

命令:
bash examples/grpo_trainer/run_qwen2-7b_math.sh
```

### Qwen3-8B on GSM8K

```
预训练模型准确率: ~70%
GRPO 训练后准确率: ~82%
训练时间: ~3 小时（8x A100）
配置: batch_size=512, n=8, epochs=30

命令:
bash examples/grpo_trainer/run_qwen3-8b.sh
```

---

## 🔗 参考资料

### 官方文档

- [GRPO 论文（DeepSeekMath）](https://arxiv.org/pdf/2402.03300)
- [DrGRPO 论文](https://arxiv.org/pdf/2503.20783)
- [verl 文档](https://verl.readthedocs.io/)

### 学习笔记

- [03_RL算法/GRPO_详解.md](../../learning_notes/03_RL算法/GRPO_详解.md) - GRPO 算法源码级详解
- [03_RL算法/03_RL算法概览.md](../../learning_notes/03_RL算法/03_RL算法概览.md) - 算法对比与选择
- [01_快速上手/ray_trainer_详解.md](../../learning_notes/01_快速上手/ray_trainer_详解.md) - 训练流程详解

### 相关示例

- `examples/ppo_trainer/` - PPO 训练示例（有 Critic）
- `examples/rloo_trainer/` - RLOO 训练示例（另一种无 Critic 算法）
- `examples/data_preprocess/` - 数据预处理

---

**创建时间**: 2026-01-28
**适用版本**: verl v0.2+
**维护者**: verl team
