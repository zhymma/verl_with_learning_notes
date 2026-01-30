# GSPO 训练器使用指南

[English](README.md) | 简体中文

## 目录
- [算法概述](#算法概述)
- [核心特点](#核心特点)
- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [参考资料](#参考资料)

## 算法概述

GSPO (Guided Sampling Policy Optimization) 是一种使用极小裁剪比率的引导采样策略优化算法。

### 算法原理

GSPO 的核心特点：
- **极小裁剪比率**：clip_ratio_low = 3e-4, clip_ratio_high = 4e-4
- **精细的策略更新**：使用非常小的裁剪限制精确控制策略变化
- **稳定性优先**：通过极小更新步长确保训练稳定

### 与其他算法对比

| 特性 | GSPO | GMPO | GRPO |
|------|------|------|------|
| Clip Ratio | **极小** (3e-4) | 中等 (0.4) | 标准 (0.2) |
| 更新步长 | 最小 | 中等 | 标准 |
| 稳定性 | 极高 | 高 | 高 |
| 收敛速度 | 慢 | 中 | 快 |

### 适用场景

- 需要极致稳定性的训练
- 大规模 MoE 模型
- 多任务学习
- 长序列训练
- 敏感任务

## 核心特点

### 关键配置

```yaml
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: gspo              # GSPO 损失模式
    clip_ratio_low: 3e-4           # 极小裁剪
    clip_ratio_high: 4e-4
    clip_ratio_c: 10.0             # 裁剪参数
```

## 快速开始

### 最简运行

```bash
cd examples/gspo_trainer
bash run_qwen30b_gspo.sh
```

### 核心配置

```bash
# 算法配置
adv_estimator=grpo
loss_mode=gspo

# GSPO 特有的极小裁剪
clip_ratio_low=3e-4
clip_ratio_high=4e-4
clip_ratio_c=10.0
```

## 详细配置

### 完整参数说明

#### GSPO 核心配置
```yaml
# 1. 算法选择
algorithm:
  adv_estimator: grpo
  use_kl_in_reward: False

# 2. 损失函数
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: gspo

    # 3. 极小裁剪比率
    clip_ratio_low: 3e-4           # 非常小！
    clip_ratio_high: 4e-4
    clip_ratio_c: 10.0

    # 4. KL loss（可选）
    use_kl_loss: False
    kl_loss_coef: 0.001
```

#### MoE 模型配置
```yaml
# Qwen3-30B MoE 配置
actor_rollout_ref:
  actor:
    megatron:
      tensor_model_parallel_size: 2
      expert_model_parallel_size: 8
      expert_tensor_parallel_size: 1
      param_offload: True
      grad_offload: True
      optimizer_offload: True
```

### 参数选择建议

#### Clip Ratio
```yaml
# GSPO 标准值（不建议改动）
clip_ratio_low: 3e-4
clip_ratio_high: 4e-4

# 如果训练过于保守，可稍微增大
clip_ratio_low: 5e-4
clip_ratio_high: 6e-4
```

#### 学习率
```yaml
# GSPO 通常用较低学习率配合极小裁剪
actor.optim.lr: 1e-6              # 推荐
```

## 实战示例

### 示例 1: Qwen3-30B GSPO

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Base \
    trainer.total_epochs=10
```

### 示例 2: GSPO + DAPO

```bash
cd examples/gspo_trainer
bash test_gspo_qwen30b_a3b_ep.sh
```

### 示例 3: 3B 模型测试

```bash
cd examples/gspo_trainer
bash test_gspo_3b_math.sh
```

## 性能对比

### GSPO vs GRPO

| 算法 | 稳定性 | 收敛速度 | 最终性能 |
|------|--------|----------|---------|
| GRPO | 高 | 快 | 基准 |
| GSPO | **极高** | 慢 | 相当或略好 |

### 适用模型规模

| 模型规模 | GRPO | GSPO |
|---------|------|------|
| < 10B | ✅ 推荐 | 可选 |
| 10-30B | ✅ 推荐 | ✅ 推荐 |
| > 30B | ✅ | ✅ **更推荐** |
| MoE | ✅ | ✅ **更推荐** |

## 常见问题

### Q1: GSPO 的极小裁剪有什么好处？

**答**：
- ✅ 极致稳定性
- ✅ 避免策略崩溃
- ✅ 适合大模型
- ⚠️ 收敛较慢

### Q2: GSPO vs GMPO 如何选择？

**答**：
- **GMPO**：MoE 模型 + 追求性能
- **GSPO**：MoE 模型 + 追求稳定

### Q3: clip_ratio 为什么这么小？

**答**：
```python
常规 GRPO: clip_ratio = 0.2     # 允许 20% 策略变化
GSPO:      clip_ratio = 3e-4    # 只允许 0.03% 策略变化

目的：极其保守的策略更新，确保稳定性
```

### Q4: 适合小模型吗？

**答**：不太推荐。
- 小模型用 GRPO 更高效
- GSPO 的优势在大模型和 MoE

## 参考资料

### 代码
- 实现：`verl/trainer/ppo/policy_loss.py`
- 示例：`examples/gspo_trainer/`

### 相关文档
- [GRPO 训练指南](../grpo_trainer/README_CN.md)
- [GMPO 训练指南](../gmpo_trainer/README_CN.md)
- [SAPO 训练指南](../sapo_trainer/README_CN.md)

---

**提示**：GSPO 是追求极致稳定性的选择，特别适合大规模 MoE 模型。如果训练过程对稳定性要求极高，GSPO 是最佳选择。
