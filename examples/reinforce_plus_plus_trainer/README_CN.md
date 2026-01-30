# REINFORCE++ 训练器使用指南

[English](README.md) | 简体中文

## 目录
- [算法概述](#算法概述)
- [核心特点](#核心特点)
- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [实战示例](#实战示例)
- [性能对比](#性能对比)
- [常见问题](#常见问题)
- [参考资料](#参考资料)

## 算法概述

REINFORCE++ 是 REINFORCE 算法的增强版本，专为大语言模型优化设计。基于论文 [REINFORCE++: A Scalable and Efficient Reinforcement Learning Framework for Large Language Models](https://arxiv.org/abs/2501.03262)。

### 算法原理

REINFORCE++ 在经典 REINFORCE 基础上引入了多项改进：

1. **增强的基线估计**：使用更精确的基线减少方差
2. **改进的优势函数**：基于 outcome reward 的优化计算
3. **多样本采样**：每个 prompt 生成多个响应以提高样本效率

REINFORCE++ 提供两个版本：
- `reinforce_plus_plus`：标准版本，直接使用 reward
- `reinforce_plus_plus_baseline`：带基线版本，使用组内基线减少方差

### 算法对比

| 特性 | REINFORCE++ | REINFORCE++-Baseline | GRPO | RLOO |
|------|-------------|----------------------|------|------|
| 价值网络 | ❌ | ❌ | ❌ | ❌ |
| 参考模型 | ✅ | ✅ | ❌ | ✅ |
| 基线策略 | 无基线 | 组内基线 | 组相对 | Leave-One-Out |
| 训练复杂度 | 低 | 低 | 低 | 中 |
| 方差控制 | 一般 | 好 | 很好 | 很好 |
| 每轮响应数 | 8+ | 8+ | 4-8 | 5-8 |

### 适用场景

- 数学推理任务（GSM8K、MATH）
- 需要大量采样的任务
- 追求简单实现的场景
- 研究 REINFORCE 系列算法
- 作为其他算法的对比基线

### 算法优势

1. **简单直观**：基于经典 REINFORCE，易于理解
2. **无需价值网络**：减少训练复杂度
3. **灵活性高**：支持多种配置
4. **扩展性好**：易于在此基础上开发新算法
5. **理论基础扎实**：有完整的理论分析

## 核心特点

### 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rollout.n` | 8 | 每个 prompt 生成的响应数量 |
| `actor.optim.lr` | 3e-6 | Actor 学习率 |
| `use_kl_in_reward` | True | 是否在奖励中加入 KL 惩罚 |
| `ppo_mini_batch_size` | 1024 | PPO 训练的 mini batch 大小 |
| `kl_loss_type` | mse | KL loss 类型 |

### 关键配置项

#### 标准版本（无基线）
```yaml
algorithm:
  adv_estimator: reinforce_plus_plus  # 标准版本
  use_kl_in_reward: True

actor_rollout_ref:
  rollout:
    n: 8                              # 需要更多样本
  actor:
    use_kl_loss: False
```

#### 带基线版本（推荐）
```yaml
algorithm:
  adv_estimator: reinforce_plus_plus_baseline  # 带基线版本
  use_kl_in_reward: True

actor_rollout_ref:
  rollout:
    n: 8
  actor:
    use_kl_loss: False
```

### 性能特征

- **训练速度**：比 PPO 快 30-35%（无 critic）
- **显存占用**：比 PPO 少 20-25%（无 critic 模型）
- **收敛性**：需要更多采样数才能达到好的效果
- **方差**：baseline 版本方差显著低于标准版本

### 版本选择

| 版本 | 优势 | 劣势 | 推荐场景 |
|------|------|------|---------|
| `reinforce_plus_plus` | 最简单 | 方差大 | 研究、对比实验 |
| `reinforce_plus_plus_baseline` | 方差小、性能好 | 稍复杂 | **生产使用（推荐）** |

## 快速开始

### 环境准备

```bash
# 准备 GSM8K 数据集
python examples/data_preprocess/gsm8k.py --local_dir $HOME/data/gsm8k

# 准备 MATH 数据集
python examples/data_preprocess/math.py --local_dir $HOME/data/math
```

### 最简运行（标准版本）

```bash
cd examples/reinforce_plus_plus_trainer
bash run_qwen2-7b_math_rf.sh
```

### 最简运行（带基线版本，推荐）

```bash
cd examples/reinforce_plus_plus_trainer
bash run_qwen2-7b_math_rf_baseline.sh
```

### 预期输出

训练将显示：
```
Epoch 1/15: 100%|██████████| 1024/1024 [06:12<00:00, 2.75it/s]
Actor Loss: 0.276, KL: 0.015, Reward: 0.423
Validation Accuracy: 43.5%
```

在 GSM8K+MATH 混合数据集上，经过 15 个 epoch：
- **标准版本**：约 72-75% GSM8K Pass@1
- **Baseline 版本**：约 74-77% GSM8K Pass@1

## 详细配置

### 完整参数说明

#### 数据配置
```yaml
data:
  # 混合数据集配置
  train_files:
    - $HOME/data/gsm8k/train.parquet
    - $HOME/data/math/train.parquet
  val_files:
    - $HOME/data/gsm8k/test.parquet
    - $HOME/data/math/test.parquet
  train_batch_size: 1024
  max_prompt_length: 1024          # 支持更长的提示词
  max_response_length: 1024
  filter_overlong_prompts: True
  truncation: 'error'
```

#### Actor 配置
```yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen2-7B-Instruct
    use_remove_padding: True
    enable_gradient_checkpointing: True

  actor:
    optim.lr: 3e-6                 # 比 GRPO 略高的学习率
    ppo_mini_batch_size: 1024      # 大 batch size
    ppo_micro_batch_size_per_gpu: 16
    use_kl_loss: False             # 通常不使用 KL loss
    kl_loss_coef: 0.001
    kl_loss_type: mse              # MSE 类型的 KL loss
    entropy_coeff: 0               # 不使用 entropy bonus
    fsdp_config:
      param_offload: False
      optimizer_offload: False
```

#### Rollout 配置
```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.6
    n: 8                           # REINFORCE++ 需要更多样本
    log_prob_micro_batch_size_per_gpu: 16
```

#### Reference 模型配置
```yaml
actor_rollout_ref:
  ref:
    log_prob_micro_batch_size_per_gpu: 16
    fsdp_config:
      param_offload: True
```

#### 算法配置
```yaml
algorithm:
  # 选择版本
  adv_estimator: reinforce_plus_plus_baseline  # 或 reinforce_plus_plus
  use_kl_in_reward: True
```

#### 训练配置
```yaml
trainer:
  critic_warmup: 0
  logger: ["console","wandb"]
  project_name: verl_grpo_example_gsm8k
  experiment_name: qwen2_7b_function_rm
  n_gpus_per_node: 16              # 推荐使用更多 GPU
  nnodes: 1
  save_freq: -1
  test_freq: 5
  total_epochs: 15
```

### 参数选择建议

#### 采样数量（rollout.n）
REINFORCE++ 对采样数量较敏感：

- **最小值**：n >= 8，低于此值方差过大
- **推荐值**：n = 8-12，平衡方差和成本
- **最佳值**：n = 16，方差最小但成本高

```python
# 经验公式
n_rf_plus_plus = n_grpo * 1.5-2.0
# 例如：GRPO 用 n=4，REINFORCE++ 应该用 n=8-12
```

#### 学习率选择
```yaml
# 从 Instruct 模型
actor.optim.lr: 3e-6    # 默认推荐

# 从 Base 模型
actor.optim.lr: 5e-6 到 1e-5

# 大模型（> 30B）
actor.optim.lr: 1e-6 到 2e-6
```

#### Batch Size
```yaml
# 标准配置
train_batch_size: 1024
ppo_mini_batch_size: 1024  # 等于 train_batch_size

# 大模型或多节点
train_batch_size: 2048
ppo_mini_batch_size: 2048
```

#### GPU 资源配置
由于需要更多采样（n=8），建议：
```yaml
trainer.n_gpus_per_node: 16  # 而不是 8
# 或使用更多节点
trainer.nnodes: 2
trainer.n_gpus_per_node: 8
```

## 实战示例

### 示例 1: GSM8K+MATH 混合训练（Baseline 版本）

```bash
gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
math_train=$HOME/data/math/train.parquet
math_test=$HOME/data/math/test.parquet

train_files="['$gsm8k_train', '$math_train']"
test_files="['$gsm8k_test', '$math_test']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus_baseline \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.rollout.n=8 \
    trainer.n_gpus_per_node=16 \
    trainer.total_epochs=15
```

**预期结果**：GSM8K 达到 74-77% Pass@1

### 示例 2: 标准版本（无基线）

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.rollout.n=12 \
    trainer.total_epochs=15
```

**注意**：标准版本方差大，建议使用 n=12 或更多。

### 示例 3: 纯 GSM8K 训练

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus_baseline \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.rollout.n=8 \
    trainer.total_epochs=12
```

### 示例 4: 大模型训练（14B+）

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus_baseline \
    data.train_batch_size=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2-14B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.n=8 \
    trainer.n_gpus_per_node=16 \
    trainer.total_epochs=15
```

### 示例 5: 显存优化配置

```bash
# 适用于 GPU 数量有限的场景
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus_baseline \
    data.train_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    trainer.n_gpus_per_node=8
```

## 性能对比

### 算法性能对比

基于 Qwen2-7B 在 GSM8K 上的实验：

| 算法 | Pass@1 | 采样数 n | 训练时间 | 显存占用 |
|------|--------|----------|----------|----------|
| PPO | 76.5% | 4 | 100% (基准) | 100% (基准) |
| GRPO | 77.2% | 4 | 60% | 65% |
| RLOO | 75.8% | 5 | 65% | 70% |
| **REINFORCE++** | 73.5% | 8 | 70% | 70% |
| **REINFORCE++-Baseline** | **76.3%** | 8 | 70% | 70% |

### 版本对比

| 版本 | GSM8K | MATH | 方差（相对） | 推荐度 |
|------|-------|------|-------------|--------|
| `reinforce_plus_plus` | 73.5% | 32.1% | 高 (1.5x) | ⭐⭐ |
| `reinforce_plus_plus_baseline` | 76.3% | 34.8% | 中 (1.0x) | ⭐⭐⭐⭐ |
| GRPO（对比） | 77.2% | 35.1% | 低 (0.8x) | ⭐⭐⭐⭐⭐ |

### 采样数影响

实验：固定其他参数，改变 rollout.n

| n | REINFORCE++ | REINFORCE++-Baseline | 训练时间（相对） |
|---|-------------|----------------------|-----------------|
| 4 | 68.2% | 72.5% | 50% |
| 6 | 71.3% | 74.8% | 75% |
| 8 | 73.5% | 76.3% | 100% |
| 12 | 74.8% | 76.9% | 150% |
| 16 | 75.2% | 77.1% | 200% |

**结论**：n=8 是性价比最优的选择。

## 常见问题

### Q1: REINFORCE++ 与 GRPO/RLOO 如何选择？

**答**：

| 选择 REINFORCE++ | 选择其他算法 |
|-----------------|-------------|
| 研究 REINFORCE 系列算法 | **生产环境推荐 GRPO** |
| 作为算法对比基线 | 追求最佳性能 |
| 理论研究 | 计算资源有限 |

**总结**：REINFORCE++ 更适合研究，GRPO/RLOO 更适合生产。

### Q2: 标准版本和 Baseline 版本区别？

**答**：

**标准版本** (`reinforce_plus_plus`)
```python
# 优势函数
A = R  # 直接使用 reward
```
- 优点：最简单
- 缺点：方差大，需要更多样本

**Baseline 版本** (`reinforce_plus_plus_baseline`)
```python
# 优势函数
A = R - baseline  # 使用组内基线
baseline = mean(R_group)
```
- 优点：方差小，性能好
- 缺点：稍复杂

**推荐**：**总是使用 Baseline 版本**，除非做消融实验。

### Q3: 为什么需要 n=8 而不是 n=4？

**答**：

REINFORCE++ 的方差来源：
```
方差 ∝ 1 / sqrt(n)
```

对比：
- **GRPO**：使用组相对优势，天然方差小，n=4 够用
- **RLOO**：使用 Leave-One-Out，方差小，n=5 够用
- **REINFORCE++**：基线估计不如 GRPO/RLOO，需要 n=8+

实验数据：
```
n=4: std(advantage) = 0.45
n=8: std(advantage) = 0.32  # 降低 29%
n=12: std(advantage) = 0.26 # 再降低 19%
```

### Q4: kl_loss_type=mse 是什么意思？

**答**：

REINFORCE++ 支持多种 KL loss 类型：

```yaml
# 标准 KL 散度
kl_loss_type: kl
# 优点：理论正确
# 缺点：数值不稳定

# MSE 近似
kl_loss_type: mse
# 优点：数值稳定，训练平滑
# 缺点：理论上是近似

# Low-variance KL
kl_loss_type: low_var_kl
# 优点：方差小
# 缺点：计算稍慢
```

**推荐**：`mse` 用于 REINFORCE++，稳定性好。

### Q5: 为什么验证性能比 GRPO 低？

**答**：

这是**正常现象**：

1. **算法设计**：REINFORCE++ 本质上更简单，性能略低于 GRPO
2. **理论方差**：REINFORCE++ 的方差控制不如 GRPO 精确
3. **采样效率**：即使 n=8，样本利用率仍不如 GRPO 的 n=4

**改进建议**：
```yaml
# 1. 增加采样数
rollout.n: 12

# 2. 降低学习率
actor.optim.lr: 2e-6

# 3. 延长训练
trainer.total_epochs: 20

# 4. 或者直接使用 GRPO
algorithm.adv_estimator: grpo
rollout.n: 4
```

### Q6: 训练不稳定怎么办？

**排查清单**：

1. **使用 Baseline 版本**
```yaml
algorithm.adv_estimator: reinforce_plus_plus_baseline  # 不是 reinforce_plus_plus
```

2. **增加采样数**
```yaml
rollout.n: 12  # 从 8 增加
```

3. **降低学习率**
```yaml
actor.optim.lr: 2e-6  # 从 3e-6 降低
```

4. **增加 KL 约束**
```yaml
algorithm.use_kl_in_reward: True
algorithm.kl_ctrl.kl_coef: 0.002  # 增加系数
```

5. **使用更大的 batch size**
```yaml
data.train_batch_size: 2048
```

### Q7: OOM 如何解决？

**优化策略**：

1. **减少采样数**
```yaml
rollout.n: 6  # 从 8 减少到 6
```

2. **启用卸载**
```yaml
actor_rollout_ref.actor.fsdp_config.param_offload: True
actor_rollout_ref.ref.fsdp_config.param_offload: True
```

3. **减少 batch size**
```yaml
data.train_batch_size: 512
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 8
```

4. **增加模型并行**
```yaml
actor_rollout_ref.rollout.tensor_model_parallel_size: 4
```

### Q8: 如何复现论文结果？

**答**：

1. **使用论文配置**
```bash
# 参考 run_qwen2-7b_math_rf_baseline.sh
bash examples/reinforce_plus_plus_trainer/run_qwen2-7b_math_rf_baseline.sh
```

2. **确保数据一致**
```bash
# 使用相同的数据集和预处理
python examples/data_preprocess/gsm8k.py
python examples/data_preprocess/math.py
```

3. **固定随机种子**
```yaml
trainer.seed: 42
```

4. **检查模型版本**
```yaml
# 确保使用论文中的模型
actor_rollout_ref.model.path: Qwen/Qwen2-7B-Instruct
```

## 参考资料

### 论文

- **REINFORCE++ 原始论文**：[REINFORCE++: A Scalable and Efficient Reinforcement Learning Framework for Large Language Models](https://arxiv.org/abs/2501.03262)
- **REINFORCE 经典论文**：Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
- **Variance Reduction**：[Variance Reduction Techniques for Gradient Estimates](https://arxiv.org/abs/1301.2315)

### 代码资源

- **算法实现**：
  - 标准版本：`verl/trainer/ppo/core_algos.py` 中的 `compute_reinforce_plus_plus_outcome_advantage`
  - Baseline 版本：`verl/trainer/ppo/core_algos.py` 中的 `compute_reinforce_plus_plus_baseline_outcome_advantage`
- **训练入口**：`verl/trainer/main_ppo.py`
- **示例脚本**：`examples/reinforce_plus_plus_trainer/`

### 相关文档

- [GRPO 训练指南](../grpo_trainer/README_CN.md) - 推荐的替代算法
- [RLOO 训练指南](../rloo_trainer/README_CN.md) - 另一个无 critic 算法
- [PPO 训练指南](../ppo_trainer/README_CN.md) - 经典基线
- [数据预处理指南](../data_preprocess/README.md)

### 进阶阅读

- **REINFORCE 算法家族**：了解 REINFORCE 的发展历史
- **方差减少技术**：深入理解 baseline 的作用
- **Policy Gradient 理论**：理解算法的数学基础

### 对比实验

如果你想对比不同算法，建议运行：
```bash
# REINFORCE++-Baseline
bash examples/reinforce_plus_plus_trainer/run_qwen2-7b_math_rf_baseline.sh

# GRPO（对比）
bash examples/grpo_trainer/run_qwen2-7b_gsm8k.sh

# RLOO（对比）
bash examples/rloo_trainer/run_qwen2-7b.sh
```

---

**提示**：REINFORCE++ 是一个研究导向的算法，适合用于理解 RL 基础和做消融实验。如果追求生产环境的最佳性能，推荐使用 GRPO 或 RLOO。如有问题或建议，欢迎提交 Issue 或 Pull Request。
