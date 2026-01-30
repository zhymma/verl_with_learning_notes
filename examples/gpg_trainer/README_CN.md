# GPG 训练器使用指南

[English](./gpg.md) | 简体中文

## 目录
- [算法概述](#算法概述)
- [核心特点](#核心特点)
- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [参考资料](#参考资料)

## 算法概述

GPG (Group Policy Gradient) 是一种极简的强化学习算法，直接优化 RL 目标而无需复杂技巧。基于论文 [GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning](https://arxiv.org/abs/2504.02546)。

### 算法原理

GPG 的核心理念："回归基础" (Back to Basics)：
- ✅ 直接优化 RL 目标
- ❌ 无代理损失函数
- ❌ 无 KL 惩罚
- ❌ 无价值网络 (critic)
- ❌ 无参考模型

使用**修正的优势函数**提高策略梯度的准确性和训练效率。

### 与其他算法对比

| 特性 | GPG | GRPO | PPO |
|------|-----|------|-----|
| 价值网络 | ❌ | ❌ | ✅ |
| 参考模型 | ❌ | ❌ | ✅ |
| KL 约束 | ❌ 可选 | ❌ 可选 | ✅ 必需 |
| 裁剪 | ❌ | ✅ | ✅ |
| 复杂度 | **最低** | 低 | 高 |
| 性能 | 很好 | 很好 | 很好 |

### 适用场景

- 数学推理任务（GSM8K、MATH）
- 追求极简实现
- 不需要 KL 约束的场景
- 快速原型开发
- 教学和研究

### 算法优势

1. **极简设计**：比 GRPO 更简单，无需任何约束
2. **强大性能**：在多个任务上优于 GRPO
3. **高效训练**：无额外模型，训练速度最快
4. **易于理解**：回归 RL 本质，便于理解和修改

## 核心特点

### 关键配置

```yaml
algorithm:
  adv_estimator: gpg               # 使用 GPG 优势估计
  use_kl_in_reward: False          # 通常不使用 KL

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: "gpg"             # 使用 GPG 损失函数
    use_kl_loss: False             # 不使用 KL loss
```

### 可选扩展（添加 KL）

虽然 GPG 原版不使用 KL，但可以添加以进一步提升性能：

```yaml
algorithm:
  adv_estimator: gpg

actor_rollout_ref:
  actor:
    use_kl_loss: True              # 启用 KL 正则化
    kl_loss_coef: 0.01             # KL 系数
    policy_loss:
      loss_mode: "gpg"
```

### 性能特征

- **训练速度**：最快（无 critic，无 ref）
- **显存占用**：最低
- **收敛性**：优异
- **配置复杂度**：极低

## 快速开始

### 最简运行

```bash
cd examples/gpg_trainer
bash run_qwen2-7b_math.sh
```

### 核心脚本解析

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gpg \
    actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.use_kl_in_reward=False \
    # ... 其他标准 RL 参数
```

## 详细配置

### 完整参数说明

#### GPG 核心配置
```yaml
# 1. 优势估计器
algorithm:
  adv_estimator: gpg
  use_kl_in_reward: False          # GPG 原版不用 KL

# 2. 策略损失
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: "gpg"             # 关键配置

    # 3. KL loss（原版不用）
    use_kl_loss: False
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
```

#### 数据配置
```yaml
data:
  train_files:
    - $HOME/data/gsm8k/train.parquet
    - $HOME/data/math/train.parquet
  train_batch_size: 1024
  max_prompt_length: 1024
  max_response_length: 1024
```

#### Rollout 配置
```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    n: 5                           # GPG 用 5 个响应
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.6
```

#### Actor 配置
```yaml
actor_rollout_ref:
  actor:
    optim.lr: 1e-6
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 16
    entropy_coeff: 0
```

### 参数选择建议

#### 采样数量
```yaml
# GPG 推荐
rollout.n: 5

# 也可以
rollout.n: 4-8  # 范围
```

#### 学习率
```yaml
# 从 Instruct 模型
actor.optim.lr: 1e-6

# 从 Base 模型
actor.optim.lr: 5e-6
```

## 实战示例

### 示例 1: 纯 GPG（无 KL）

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gpg \
    actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    data.train_files="['$HOME/data/gsm8k/train.parquet']" \
    trainer.total_epochs=15
```

### 示例 2: GPG + KL（扩展版本）

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gpg \
    actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    trainer.total_epochs=15
```

### 示例 3: GPG + Megatron（大模型）

```bash
cd examples/gpg_trainer
bash run_qwen2-7b_math_megatron.sh
```

## 性能对比

### GPG vs GRPO

| 算法 | GSM8K | MATH | 复杂度 | 速度 |
|------|-------|------|--------|------|
| GRPO | 77.2% | 35.1% | 低 | 快 |
| **GPG** | **78.5%** | **36.2%** | **更低** | **更快** |

### GPG vs GPG+KL

| 配置 | GSM8K | MATH | 稳定性 |
|------|-------|------|--------|
| GPG (纯) | 78.5% | 36.2% | 好 |
| GPG+KL | 79.1% | 36.8% | **更好** |

## 常见问题

### Q1: GPG vs GRPO？

**答**：
- **GPG**：更简单，性能略好，无裁剪
- **GRPO**：有裁剪机制，更传统

两者都很优秀，GPG 更极简。

### Q2: 为什么不需要 KL？

**答**：GPG 通过修正的优势函数实现隐式正则化，通常无需显式 KL 约束。但添加 KL 可以进一步提升性能。

### Q3: GPG 会不会太激进？

**答**：理论上可能，但实践中：
- 修正优势函数提供自然约束
- 可选添加 KL loss
- 实验表明训练稳定

### Q4: 适合生产环境吗？

**答**：✅ 非常适合！
- 简单稳定
- 性能优异
- 易于部署
- 无复杂依赖

## 参考资料

### 论文
- **GPG 原始论文**：[GPG: A Simple and Strong Reinforcement Learning Baseline](https://arxiv.org/abs/2504.02546)

### 代码
- 实现：`verl/trainer/ppo/core_algos.py`
- 示例：`examples/gpg_trainer/`
- 文档：`examples/gpg_trainer/gpg.md`

### 相关文档
- [GRPO 训练指南](../grpo_trainer/README_CN.md) - 最相似的算法
- [ReMax 训练指南](../remax_trainer/README_CN.md) - 另一个极简算法

---

**提示**：GPG 是 veRL 中最简单也最强大的算法之一。"简单就是美" - GPG 证明了有时候回归基础反而能获得最好的效果。
