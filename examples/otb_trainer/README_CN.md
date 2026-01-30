# OTB 训练器使用指南

[English](README.md) | 简体中文

## 目录
- [算法概述](#算法概述)
- [核心特点](#核心特点)
- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [参考资料](#参考资料)

## 算法概述

OTB (Optimal Token Baseline) 是一种使用最优token级基线的强化学习算法。

### 算法原理

OTB 计算每个 token 的最优基线：
```python
# 传统方法：使用统一基线
baseline = mean(all_rewards)

# OTB：每个 token 有自己的基线
baseline[token_i] = optimal_baseline_for_token_i
```

通过计算和利用更精确的 token 级基线来减少方差。

### 与其他算法对比

| 特性 | OTB | GRPO | RLOO |
|------|-----|------|------|
| 基线粒度 | Token级 | 序列级 | 序列级 |
| 方差减少 | 很好 | 好 | 很好 |
| 计算开销 | 稍高 | 低 | 中 |
| 实现复杂度 | 中 | 低 | 中 |

### 适用场景

- 需要精确方差控制的任务
- token级奖励建模
- 研究和算法对比
- 长序列任务

## 核心特点

### 关键配置

```yaml
algorithm:
  adv_estimator: optimal_token_baseline  # OTB 算法

actor_rollout_ref:
  actor:
    calculate_sum_pi_squared: True       # OTB 需要
```

## 快速开始

### 最简运行

```bash
cd examples/otb_trainer
bash run_qwen2_5-7b.sh
```

### 核心配置

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=optimal_token_baseline \
    actor_rollout_ref.actor.calculate_sum_pi_squared=True \
    actor_rollout_ref.rollout.n=8 \
    # ... 其他参数
```

## 详细配置

### 完整参数说明

#### OTB 核心配置
```yaml
# 1. 算法选择
algorithm:
  adv_estimator: optimal_token_baseline

# 2. Actor 配置
actor_rollout_ref:
  actor:
    calculate_sum_pi_squared: True     # OTB 必需
    use_dynamic_bsz: False             # 通常关闭
    use_kl_loss: False
    entropy_coeff: 0
```

#### 数据配置
```yaml
data:
  train_batch_size: 128              # OTB 通常用较小 batch
  max_prompt_length: 1024
  max_response_length: 2048          # 支持长序列
```

#### Rollout 配置
```yaml
actor_rollout_ref:
  rollout:
    n: 8                             # OTB 需要更多样本
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.75
```

### 参数选择建议

#### 采样数量
```yaml
# OTB 推荐值
rollout.n: 8

# 更多样本可能更好
rollout.n: 12-16
```

#### Batch Size
```yaml
# OTB 通常用较小 batch size
train_batch_size: 128-256
```

## 实战示例

### 示例 1: Qwen2.5-7B OTB

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=optimal_token_baseline \
    data.train_batch_size=128 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B \
    actor_rollout_ref.actor.calculate_sum_pi_squared=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.rollout.n=8 \
    trainer.total_epochs=15
```

### 示例 2: GSM8K+MATH 混合训练

```bash
gsm8k_train=$HOME/data/gsm8k/train.parquet
math_train=$HOME/data/math/train.parquet
train_files="['$gsm8k_train', '$math_train']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=optimal_token_baseline \
    data.train_files="$train_files" \
    actor_rollout_ref.actor.calculate_sum_pi_squared=True \
    actor_rollout_ref.rollout.n=8 \
    trainer.total_epochs=15
```

## 性能对比

### OTB vs 其他算法

| 算法 | GSM8K | 方差 | 计算开销 |
|------|-------|------|---------|
| GRPO | 77.2% | 基准 | 基准 |
| RLOO | 75.8% | -10% | +10% |
| **OTB** | 76.5% | -15% | +15% |

### 特点总结

- ✅ 更精确的方差控制
- ✅ Token级基线建模
- ⚠️ 计算开销稍高
- ⚠️ 实现较复杂

## 常见问题

### Q1: OTB 的优势是什么？

**答**：
- Token 级最优基线
- 更精确的方差减少
- 理论上更优

### Q2: 为什么性能不如 GRPO？

**答**：
- 理论优势不一定转化为实际性能
- 实现细节影响大
- GRPO 的简单性更实用

### Q3: 什么时候用 OTB？

**答**：
- 研究目的
- 算法对比
- 理论验证

**生产环境推荐 GRPO/RLOO**。

### Q4: calculate_sum_pi_squared 是什么？

**答**：
```python
# OTB 需要计算策略概率的平方和
sum_pi_squared = sum(pi[i]^2 for all i)
# 用于计算最优 token 基线
```

## 参考资料

### 代码
- 实现：`verl/trainer/ppo/core_algos.py`
- 示例：`examples/otb_trainer/`

### 相关文档
- [GRPO 训练指南](../grpo_trainer/README_CN.md) - 更推荐
- [RLOO 训练指南](../rloo_trainer/README_CN.md) - 更推荐

---

**提示**：OTB 是一个理论导向的算法，适合研究和对比实验。生产环境推荐使用 GRPO 或 RLOO，它们更简单且性能通常更好。
