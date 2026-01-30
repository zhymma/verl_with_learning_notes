# SAPO 训练器使用指南

[English](README.md) | 简体中文

## 目录
- [算法概述](#算法概述)
- [核心特点](#核心特点)
- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [参考资料](#参考资料)

## 算法概述

SAPO (Self-Adaptive Policy Optimization) 是一种自适应策略优化算法，使用平滑函数替代PPO的裁剪机制。基于论文 [SAPO: Self-Adaptive Policy Optimization for Large Language Models](https://arxiv.org/pdf/2511.20347)。

### 算法原理

SAPO 的核心创新：
1. **平滑函数替代裁剪**：使用可微分的平滑函数替代硬裁剪
2. **自适应 tau 参数**：为正负优势分别设置平滑参数（tau_pos, tau_neg）
3. **更稳定的训练**：避免裁剪带来的梯度不连续性

数学形式：
```
SAPO Loss = -min(r_t * A_t, smooth(r_t, tau) * A_t)
其中 smooth(r, tau) 是平滑函数
```

### 与其他算法对比

| 特性 | SAPO | PPO | GRPO |
|------|------|-----|------|
| 裁剪机制 | ❌ 平滑函数 | ✅ 硬裁剪 | ✅ 硬裁剪 |
| 价值网络 | 可选 | ✅ | ❌ |
| 参数敏感度 | 低 | 中 | 低 |
| 训练稳定性 | 很好 | 好 | 很好 |
| 梯度连续性 | ✅ | ❌ | ❌ |

### 适用场景

- 大规模模型训练（Qwen3-30B等）
- 需要长序列训练（多轮对话，数学推理）
- 追求训练稳定性
- MoE 模型训练
- 多任务学习场景

## 核心特点

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `loss_mode` | sapo | 损失函数模式 |
| `tau_pos` | 1.0 | 正优势的平滑参数 |
| `tau_neg` | 1.05 | 负优势的平滑参数 |
| `adv_estimator` | grpo | 基础优势估计器 |

### 关键配置

```yaml
algorithm:
  adv_estimator: grpo              # 使用 GRPO 优势估计

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: sapo              # 使用 SAPO 损失函数
    tau_pos: 1.0                   # 正优势平滑参数
    tau_neg: 1.05                  # 负优势平滑参数（略大于 tau_pos）
```

### SAPO vs Vanilla PPO

| 项目 | Vanilla | SAPO |
|------|---------|------|
| 损失函数 | 硬裁剪 | 平滑函数 |
| 梯度 | 不连续 | 连续 |
| 稳定性 | 好 | **更好** |
| MoE 友好 | 一般 | **很好** |

## 快速开始

### 最简运行（7B模型）

```bash
cd examples/sapo_trainer
bash run_qwen30b_sapo.sh
```

### 配置说明

```bash
# 算法配置
adv_estimator=grpo
loss_mode=sapo

# SAPO 特定参数
tau_pos=1.0
tau_neg=1.05

# 不使用裁剪（SAPO 用平滑函数替代）
clip_ratio_low=null
clip_ratio_high=null
```

## 详细配置

### 完整参数说明

#### SAPO 核心配置
```yaml
# 1. 算法选择
algorithm:
  adv_estimator: grpo              # 可以用 gae, grpo 等

# 2. 损失函数模式
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: sapo              # 关键：使用 SAPO

    # 3. 平滑参数
    tau_pos: 1.0                   # 正优势平滑度
    tau_neg: 1.05                  # 负优势平滑度（稍大）

    # 4. 不使用裁剪
    # clip_ratio_low: null         # SAPO 不需要裁剪
    # clip_ratio_high: null
```

#### 数据配置（长序列）
```yaml
data:
  max_prompt_length: 2048          # 支持长提示词
  max_response_length: 8192        # 支持长响应
  return_raw_chat: True            # 多轮对话

  # 超长缓冲区配置
  enable_overlong_buffer: True
  overlong_buffer_len: 4096
  overlong_penalty_factor: 1.0
```

#### 大模型配置（Qwen3-30B）
```yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-30B-A3B-Base

  # Megatron 并行配置（MoE 模型）
  actor:
    megatron:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 8    # MoE 专家并行
      expert_tensor_parallel_size: 1
      param_offload: True
      grad_offload: True
      optimizer_offload: True
```

### 参数选择建议

#### tau_pos 和 tau_neg
```python
# 论文推荐值（Qwen3-30B）
tau_pos = 1.0
tau_neg = 1.05

# 更保守（更接近裁剪）
tau_pos = 0.5
tau_neg = 0.6

# 更激进（更平滑）
tau_pos = 1.5
tau_neg = 1.8

# 规律：tau_neg 略大于 tau_pos
tau_neg = tau_pos * 1.05
```

#### 基础优势估计器选择
```yaml
# 推荐：GRPO（简单高效）
algorithm.adv_estimator: grpo

# 也可以：GAE（需要 critic）
algorithm.adv_estimator: gae
trainer.critic_warmup: 10

# 或：RLOO
algorithm.adv_estimator: rloo
```

## 实战示例

### 示例 1: Qwen3-30B MoE 模型

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=sapo \
    actor_rollout_ref.actor.tau_pos=1.0 \
    actor_rollout_ref.actor.tau_neg=1.05 \
    actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Base \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
    data.max_response_length=8192 \
    trainer.total_epochs=10
```

### 示例 2: 7B 模型 SAPO（FSDP）

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=sapo \
    actor_rollout_ref.actor.tau_pos=1.0 \
    actor_rollout_ref.actor.tau_neg=1.05 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    data.train_batch_size=256 \
    trainer.n_gpus_per_node=8
```

### 示例 3: SAPO + GAE（带 Critic）

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.actor.policy_loss.loss_mode=sapo \
    critic.policy_loss.loss_mode=sapo \
    trainer.critic_warmup=10 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct
```

## 性能对比

### SAPO vs Vanilla（Qwen3-30B）

| 配置 | 收敛性 | 稳定性 | 最终性能 |
|------|--------|--------|---------|
| Vanilla (clip) | 基准 | 基准 | 基准 |
| **SAPO** | **更快** | **更好** | **+2-3%** |

### MoE 模型友好性

| 模型类型 | Vanilla | SAPO |
|---------|---------|------|
| Dense | ✅ 好 | ✅ 好 |
| MoE | ⚠️ 不稳定 | ✅ **稳定** |

## 常见问题

### Q1: SAPO vs PPO/GRPO？

**简答**：
- **SAPO = GRPO/PPO + 平滑函数**
- 训练更稳定，特别是 MoE 模型
- 性能通常略好于 vanilla

### Q2: tau_pos 和 tau_neg 如何调？

**推荐**：
```yaml
# 默认值（通常最优）
tau_pos: 1.0
tau_neg: 1.05

# 如果训练不稳定，降低 tau
tau_pos: 0.5
tau_neg: 0.6

# 如果训练过于保守，增大 tau
tau_pos: 1.5
tau_neg: 1.8
```

### Q3: SAPO 需要特殊数据吗？

**答**：不需要。SAPO 是优化算法的改进，与数据无关。

### Q4: 可以和其他技术结合吗？

**答**：可以！
- SAPO + GRPO ✅
- SAPO + GAE ✅
- SAPO + RLOO ✅
- SAPO + MTP ✅
- SAPO + DAPO ✅

## 参考资料

### 论文
- **SAPO 原始论文**：[Self-Adaptive Policy Optimization](https://arxiv.org/pdf/2511.20347)

### 代码
- 实现：`verl/trainer/ppo/policy_loss.py`
- 示例：`examples/sapo_trainer/`

### 相关文档
- [GRPO 训练指南](../grpo_trainer/README_CN.md)
- [PPO 训练指南](../ppo_trainer/README_CN.md)

---

**提示**：SAPO 特别适合大规模 MoE 模型训练。对于 Qwen3-30B 等模型，推荐使用 SAPO 以获得更稳定的训练过程。
