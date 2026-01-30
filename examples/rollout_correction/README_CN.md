# Rollout Correction - 离策略修正技术

[English](README.md) | 简体中文

## 概述

Rollout Correction 示例展示了如何在 RL 训练中使用离策略修正技术，包括重要性采样 (Importance Sampling, IS) 和拒绝采样 (Rejection Sampling, RS)。这些技术可以提高样本效率，减少策略漂移的负面影响，并改善训练稳定性。

本示例提供了两个完整的配置：
1. **基本 RLOO + 序列级 IS**: 使用自归一化的序列级重要性采样
2. **PPO-clip + 多重 RS**: 结合 token 级和序列级的多重拒绝采样标准

## 主要特点

- **重要性采样 (IS)**: 修正离策略数据的分布偏差
  - Token 级 IS: 逐 token 计算权重
  - 序列级 IS: 整体序列权重
  - 自归一化选项: 确保权重均值为 1.0

- **拒绝采样 (RS)**: 过滤极端的离策略样本
  - `token_k1`: Token 级别的阈值过滤
  - `seq_max_k2`: 序列级别的极端值检测
  - 多重 RS 标准组合

- **Bypass 模式**: 重用 rollout 阶段的 log_prob，避免重复计算
- **灵活的损失类型**: 支持 REINFORCE 和 PPO-clip

## 适用场景

Rollout Correction 技术适用于以下场景：

1. **离策略学习**: 当训练策略与行为策略不一致时
2. **数据重用**: 重用历史生成的数据进行训练
3. **大批次训练**: 生成大批次数据后进行多轮训练
4. **策略漂移**: 防止在多轮更新后策略偏离过大
5. **样本效率**: 提高样本利用率，减少重新生成的需求
6. **分布式训练**: 异步或延迟更新的场景

## 技术原理

### 重要性采样 (Importance Sampling)

重要性采样用于修正使用旧策略生成的数据来训练新策略时的分布偏差：

```
IS权重 = π_new(a|s) / π_old(a|s)
```

#### Token 级 IS

```
w_token = exp(log π_new - log π_old)
```

每个 token 独立计算权重，适用于细粒度控制。

#### 序列级 IS

```
w_sequence = Π_t w_token[t] = exp(Σ_t (log π_new[t] - log π_old[t]))
```

整个序列使用单一权重，适用于整体评估。

#### 自归一化 (Batch Normalization)

```
w_normalized = w / (Σ w / N)
```

确保批次内权重的均值为 1.0，减少方差。

### 拒绝采样 (Rejection Sampling)

拒绝采样通过丢弃极端的离策略样本来提高稳定性：

#### Token K1 标准

```
lower_threshold <= w_token <= upper_threshold
```

拒绝 token 级权重超出阈值范围的样本。

#### Sequence Max K2 标准

```
max_t(w_token[t]) <= threshold
```

拒绝包含极端 token 权重的序列。

### Bypass 模式

在 bypass 模式下，重用 rollout 阶段计算的 log_prob 作为"旧策略"的 log_prob，避免额外的前向传播。这在以下情况下特别有用：

- 单轮 PPO 更新 (`ppo_epochs=1`)
- 想要最大化计算效率
- rollout 和训练之间策略变化不大

## 快速开始

### 环境准备

1. 安装 veRL 及其依赖：

```bash
# 安装 veRL 和 vLLM
pip install -e .[test,vllm]
```

2. 准备数据和模型：

```bash
# 准备训练数据
export TRAIN_FILE="data/train.parquet"
export TEST_FILE="data/test.parquet"

# 设置模型路径
export MODEL_PATH="Qwen/Qwen2.5-7B"
```

### 示例 1: RLOO + 序列级 IS

运行基本的 RLOO 训练，使用序列级重要性采样：

```bash
bash examples/rollout_correction/run_with_rollout_corr.sh
```

这个示例：
- 使用 RLOO 优势估计器
- 应用序列级自归一化 IS
- 使用 REINFORCE 损失
- 启用 bypass 模式以提高效率

### 示例 2: PPO-clip + 多重 RS

运行 PPO 训练，使用多重拒绝采样标准：

```bash
bash examples/rollout_correction/run_with_rollout_corr_multi_rs.sh
```

这个示例：
- 使用 GRPO 优势估计器
- 应用 token 级 IS（不自归一化）
- 使用多重 RS 标准（token_k1 + seq_max_k2）
- 使用 PPO-clip 损失

## 详细配置说明

### 1. 重要性采样 (IS) 配置

#### IS 模式

```bash
rollout_is="sequence"                     # 选项: "token", "sequence", "null"
```

- **`token`**: Token 级 IS，每个 token 独立权重
- **`sequence`**: 序列级 IS，整个序列单一权重
- **`null`**: 不使用 IS

#### IS 阈值

```bash
rollout_is_threshold=2.0                  # 权重裁剪阈值
```

权重会被裁剪到 `[1/threshold, threshold]` 范围内：
```
w_clipped = clip(w, 1/2.0, 2.0) = clip(w, 0.5, 2.0)
```

**推荐值**:
- `2.0`: 保守，适合稳定训练
- `5.0`: 适中
- `10.0`: 激进，允许更大的权重变化

#### IS 批次归一化

```bash
rollout_is_batch_normalize="true"        # 自归一化
```

- **`true`**: 归一化使批次内权重均值为 1.0（推荐用于 REINFORCE）
- **`false`**: 保持原始裁剪后的权重（适用于 PPO-clip）

### 2. 拒绝采样 (RS) 配置

#### RS 模式

```bash
# 单一标准
rollout_rs="token_k1"

# 多重标准 (用逗号分隔)
rollout_rs="token_k1,seq_max_k2"

# 不使用 RS
rollout_rs="null"
```

可用的 RS 标准：
- **`token_k1`**: Token 级阈值过滤
- **`seq_max_k2`**: 序列最大值过滤
- **`sequence`**: 序列级阈值过滤
- 可以组合多个标准

#### RS 阈值

```bash
# 单一标准 (上界)
rollout_rs_threshold="2.0"

# token_k1 (下界_上界)
rollout_rs_threshold="0.6_1.6"

# 多重标准 (用逗号分隔，与 RS 模式对应)
rollout_rs_threshold="0.6_1.6,2.5"
```

**Token K1 阈值示例**:
```bash
rollout_rs_threshold="0.6_1.6"
# 拒绝任何 token 权重不在 [0.6, 1.6] 范围内的序列
```

**Seq Max K2 阈值示例**:
```bash
rollout_rs_threshold="2.5"
# 拒绝包含任何权重 > 2.5 的 token 的序列
```

### 3. Bypass 模式和损失类型

#### Bypass 模式

```bash
bypass_mode="true"                       # 启用 bypass 模式
```

- **`true`**: 重用 rollout 的 log_prob 作为"旧策略"log_prob
- **`false`**: 在训练前重新计算旧策略 log_prob（标准 PPO）

**何时使用 bypass 模式**:
- ✅ `ppo_epochs=1` (单轮更新)
- ✅ rollout 和训练之间策略变化很小
- ✅ 追求最大计算效率
- ❌ `ppo_epochs>1` (多轮更新，策略会在轮次间变化)

#### 损失类型

```bash
loss_type="reinforce"                    # 选项: "reinforce", "ppo_clip"
```

- **`reinforce`**: REINFORCE 损失，显式使用 IS 权重
  ```
  L = -w * A * log π
  ```
  适合与自归一化 IS 结合使用

- **`ppo_clip`**: PPO-clip 损失，使用比率进行裁剪
  ```
  L = -min(r*A, clip(r, 1-ε, 1+ε)*A)
  r = π_new / π_old
  ```
  适合标准 PPO 训练

### 4. 算法配置

```bash
adv_estimator=rloo                       # RLOO, GRPO, GAE 等
gamma=1.0                                # 折扣因子
```

Rollout correction 可以与任何优势估计器结合使用：
- **RLOO**: 推荐与序列级 IS 结合
- **GRPO**: 推荐与 token 级 IS 或 RS 结合
- **GAE**: 适用于较长的时间范围

### 5. 日志和监控

启用详细的 rollout correction 指标：

```bash
trainer.logger='["console","wandb"]'
trainer.project_name="rollout_corr_experiment"
```

重要指标：
- `rollout_corr/rollout_is_mean`: IS 权重均值（归一化前应接近 1.0）
- `rollout_corr/rollout_is_batch_norm_factor`: 批次归一化因子
- `rollout_corr/rollout_is_eff_sample_size`: 有效样本大小（应 > 0.5）
- `rollout_corr/rollout_rs_masked_fraction`: RS 拒绝的样本比例
- `rollout_corr/rollout_rs_token_k1_mean`: Token K1 RS 统计
- `rollout_corr/rollout_rs_seq_max_k2_mean`: Seq Max K2 RS 统计

## 配置组合建议

### 配置 1: RLOO + 序列级自归一化 IS

```bash
adv_estimator=rloo
rollout_is="sequence"
rollout_is_threshold=2.0
rollout_is_batch_normalize="true"
rollout_rs="null"
bypass_mode="true"
loss_type="reinforce"
ppo_epochs=1
```

**特点**:
- 适合 RLOO 算法
- 自归一化确保稳定性
- 最高计算效率（bypass 模式 + 无 RS）

**适用场景**:
- 标准 RLOO 训练
- 追求简单和效率
- 策略漂移不严重

### 配置 2: GRPO + Token 级 IS + 多重 RS

```bash
adv_estimator=grpo
rollout_is="token"
rollout_is_threshold=2.0
rollout_is_batch_normalize="false"
rollout_rs="token_k1,seq_max_k2"
rollout_rs_threshold="0.6_1.6,2.5"
bypass_mode="true"
loss_type="ppo_clip"
ppo_epochs=1
```

**特点**:
- 细粒度的 token 级控制
- 多重 RS 标准提供强大的过滤
- PPO-clip 损失提供额外稳定性

**适用场景**:
- 需要精细控制
- 离策略程度较大
- 数据质量参差不齐

### 配置 3: GAE + 序列级 IS + Token RS

```bash
adv_estimator=gae
gamma=0.99
lam=0.95
rollout_is="sequence"
rollout_is_threshold=3.0
rollout_is_batch_normalize="true"
rollout_rs="token_k1"
rollout_rs_threshold="0.5_2.0"
bypass_mode="false"                     # 多轮更新需要关闭
loss_type="ppo_clip"
ppo_epochs=4
```

**特点**:
- 标准 PPO + 时序差分学习
- 多轮更新（不使用 bypass）
- 结合 IS 和 RS 的优势

**适用场景**:
- 标准 PPO 训练
- 需要多轮策略更新
- 长时间范围的任务

### 配置 4: 纯 RS 过滤 (无 IS)

```bash
adv_estimator=grpo
rollout_is="null"                       # 不使用 IS
rollout_rs="token_k1,seq_max_k2"
rollout_rs_threshold="0.7_1.4,2.0"      # 更严格的阈值
bypass_mode="true"
loss_type="ppo_clip"
ppo_epochs=1
```

**特点**:
- 只过滤极端样本，不修正权重
- 简单直观
- 适合分布偏移不大的场景

**适用场景**:
- 离策略程度较小
- 只需要过滤异常值
- 追求简单性

## 运行示例

### 示例 1: 基本 RLOO with Seq IS

```bash
#!/usr/bin/env bash
set -xeuo pipefail

MODEL_PATH="Qwen/Qwen2.5-7B"
TRAIN_FILE="data/train.parquet"
TEST_FILE="data/test.parquet"

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.train_batch_size=128 \
    algorithm.adv_estimator=rloo \
    algorithm.gamma=1.0 \
    algorithm.rollout_correction.rollout_is="sequence" \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize="true" \
    algorithm.rollout_correction.rollout_rs="null" \
    algorithm.rollout_correction.bypass_mode="true" \
    algorithm.rollout_correction.loss_type="reinforce" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="rollout_corr_rloo" \
    trainer.total_epochs=10
```

### 示例 2: GRPO with Multi-RS

```bash
#!/usr/bin/env bash
set -xeuo pipefail

MODEL_PATH="Qwen/Qwen2.5-7B"
TRAIN_FILE="data/train.parquet"
TEST_FILE="data/test.parquet"

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.train_batch_size=128 \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.rollout_correction.rollout_is="token" \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize="false" \
    algorithm.rollout_correction.rollout_rs="token_k1,seq_max_k2" \
    algorithm.rollout_correction.rollout_rs_threshold="0.6_1.6,2.5" \
    algorithm.rollout_correction.bypass_mode="true" \
    algorithm.rollout_correction.loss_type="ppo_clip" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="rollout_corr_multi_rs" \
    trainer.total_epochs=5
```

### 示例 3: 标准 PPO with IS + RS (多轮更新)

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.gamma=0.99 \
    algorithm.lam=0.95 \
    algorithm.rollout_correction.rollout_is="sequence" \
    algorithm.rollout_correction.rollout_is_threshold=3.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize="true" \
    algorithm.rollout_correction.rollout_rs="token_k1" \
    algorithm.rollout_correction.rollout_rs_threshold="0.5_2.0" \
    algorithm.rollout_correction.bypass_mode="false" \     # 重要!
    algorithm.rollout_correction.loss_type="ppo_clip" \
    actor_rollout_ref.actor.ppo_epochs=4 \                 # 多轮更新
    # ... 其他配置
```

### 示例 4: 调试 IS 权重

启用详细日志以调试 IS 权重：

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.rollout_correction.rollout_is="token" \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize="true" \
    trainer.logger='["console","wandb","tensorboard"]' \
    # ... 其他配置
```

在 WandB 中监控：
- `rollout_corr/rollout_is_mean`: 应在 0.8-1.2 之间
- `rollout_corr/rollout_is_eff_sample_size`: 应 > 0.5（越高越好）
- `rollout_corr/rollout_is_max`: 应接近阈值

## 常见问题

### 1. IS 权重均值偏离 1.0

**问题**: `rollout_corr/rollout_is_mean` 远离 1.0。

**原因**:
- 策略漂移过大
- 阈值设置不合理
- 未启用批次归一化

**解决方案**:
```bash
# 启用批次归一化
rollout_is_batch_normalize="true"

# 减小阈值
rollout_is_threshold=1.5                    # 更保守

# 减小学习率以降低策略漂移
actor_rollout_ref.actor.optim.lr=1e-6

# 增加更新频率（减少离策略程度）
# 更频繁地重新生成 rollout 数据
```

### 2. 有效样本大小 (ESS) 过低

**问题**: `rollout_corr/rollout_is_eff_sample_size` < 0.3。

**含义**: 大部分样本的权重接近 0，训练效率低下。

**解决方案**:
```bash
# 方案 1: 使用拒绝采样过滤极端样本
rollout_rs="token_k1"
rollout_rs_threshold="0.6_1.6"

# 方案 2: 增加 IS 阈值
rollout_is_threshold=5.0                    # 允许更大的权重

# 方案 3: 减小策略更新步长
actor_rollout_ref.actor.optim.lr=3e-7

# 方案 4: 更频繁地重新生成数据
```

### 3. RS 拒绝率过高

**问题**: `rollout_corr/rollout_rs_masked_fraction` > 0.5。

**含义**: 超过一半的样本被拒绝，样本效率低下。

**解决方案**:
```bash
# 放宽 RS 阈值
rollout_rs_threshold="0.5_2.0"              # 更宽松的范围

# 只使用 seq_max_k2 而非 token_k1
rollout_rs="seq_max_k2"                     # 只过滤极端情况

# 或完全禁用 RS
rollout_rs="null"

# 改为使用 IS 修正
rollout_is="sequence"
rollout_is_batch_normalize="true"
```

### 4. Bypass 模式与多轮更新冲突

**问题**: 使用 `bypass_mode=true` 但 `ppo_epochs > 1` 导致性能下降。

**原因**: Bypass 模式假设策略不变，但多轮更新会改变策略。

**解决方案**:
```bash
# 方案 1: 禁用 bypass 模式
bypass_mode="false"
ppo_epochs=4

# 方案 2: 使用单轮更新
bypass_mode="true"
ppo_epochs=1
```

### 5. 如何选择 IS 阈值？

**推荐值**:
```bash
# 保守 (高稳定性，低灵活性)
rollout_is_threshold=1.5

# 平衡 (推荐起点)
rollout_is_threshold=2.0

# 适中
rollout_is_threshold=3.0

# 激进 (高灵活性，可能不稳定)
rollout_is_threshold=5.0
```

**经验法则**:
- 如果 ESS < 0.5，增加阈值
- 如果训练不稳定，减小阈值
- 监控 `rollout_is_max`，它应该接近但不总是等于阈值

### 6. Token 级 vs 序列级 IS？

**Token 级 IS**:
```bash
rollout_is="token"
```
- 优点: 细粒度控制，更灵活
- 缺点: 可能增加方差
- 适合: 需要精细控制的场景，与 token 级 RS 配合

**序列级 IS**:
```bash
rollout_is="sequence"
```
- 优点: 低方差，更稳定
- 缺点: 粗粒度，可能过度修正
- 适合: RLOO、REINFORCE 等序列级算法

### 7. 何时使用 RS？

**使用 RS 的场景**:
- ✅ 离策略程度很大
- ✅ 数据质量参差不齐
- ✅ 有明确的"坏样本"定义
- ✅ 追求训练稳定性

**不使用 RS 的场景**:
- ❌ 样本效率是首要目标
- ❌ 离策略程度很小
- ❌ 数据已经经过质量控制

### 8. 如何调试 Rollout Correction？

**调试步骤**:

1. **检查基本 IS 统计**:
```bash
# 监控这些指标
rollout_corr/rollout_is_mean              # 应接近 1.0
rollout_corr/rollout_is_std               # 标准差，越小越好
rollout_corr/rollout_is_max               # 应接近阈值
rollout_corr/rollout_is_eff_sample_size   # 应 > 0.5
```

2. **检查 RS 统计**:
```bash
rollout_corr/rollout_rs_masked_fraction   # 拒绝率，应 < 0.3
rollout_corr/rollout_rs_token_k1_mean     # Token K1 平均值
rollout_corr/rollout_rs_seq_max_k2_mean   # Seq Max K2 平均值
```

3. **对比有无 Rollout Correction**:
```bash
# 运行基线（无 rollout correction）
python3 -m verl.trainer.main_ppo \
    algorithm.rollout_correction.rollout_is="null" \
    algorithm.rollout_correction.rollout_rs="null" \
    # ...

# 运行带 rollout correction 的版本
# 比较性能、稳定性、样本效率
```

4. **逐步启用功能**:
```bash
# 步骤 1: 只使用 IS
rollout_is="sequence"
rollout_rs="null"

# 步骤 2: 添加 RS
rollout_rs="token_k1"

# 步骤 3: 添加多重 RS
rollout_rs="token_k1,seq_max_k2"
```

## 性能影响

### 计算开销

| 配置 | 额外开销 | 说明 |
|------|---------|------|
| IS only | ~5-10% | 需要计算 log_prob 差异 |
| RS only | ~1-2% | 只需要权重比较 |
| IS + RS | ~5-10% | RS 开销可忽略 |
| Bypass 模式 | 0% | 重用已有 log_prob |
| 非 Bypass 模式 | +20-30% | 需要额外的 reference 前向传播 |

### 样本效率

- **IS**: 可以提高 1.5-3x 样本效率（取决于离策略程度）
- **RS**: 可能降低样本效率（由于拒绝样本），但提高质量
- **组合**: 平衡样本数量和质量

### 训练稳定性

- **IS**: 显著提高稳定性，特别是在离策略学习中
- **RS**: 防止极端样本破坏训练
- **批次归一化**: 减少方差，提高稳定性

## 参考资料

### 官方文档

- [Rollout Correction 文档](https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr.md)
- [Rollout Correction 数学原理](https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr_math.md)

### 相关论文

1. **Importance Sampling**:
   - [Importance Weighted Actor-Learner Architectures (IMPALA)](https://arxiv.org/abs/1802.01561)

2. **Off-Policy RL**:
   - [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

3. **Rejection Sampling in RL**:
   - [Safe Policy Optimization with Local Generalized Linear Models](https://arxiv.org/abs/1907.01752)

### 相关示例

- `examples/rloo_trainer/`: RLOO 训练
- `examples/ppo_trainer/`: 标准 PPO 训练
- `examples/grpo_trainer/`: GRPO 训练

### 实现文件

- `verl/trainer/ppo/core_algos.py`: Rollout correction 实现
- `verl/trainer/config/algorithm/`: 算法配置
- `verl/protocol.py`: 数据协议

## 总结

Rollout Correction 提供了强大的离策略修正工具：

1. **重要性采样 (IS)**: 修正分布偏差，提高样本效率
2. **拒绝采样 (RS)**: 过滤极端样本，提高训练稳定性
3. **Bypass 模式**: 提高计算效率
4. **灵活配置**: 支持多种组合和参数设置

通过合理使用这些技术，您可以：
- 提高样本利用率
- 重用历史数据
- 稳定离策略学习
- 适应分布式和异步训练场景

建议从简单配置开始（如示例 1），根据具体需求逐步添加复杂功能。

## 支持与反馈

如果遇到问题或有改进建议：

1. 查看 [官方文档](https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr.md)
2. 提交 [GitHub Issue](https://github.com/volcengine/verl/issues)
3. 参考数学原理文档理解细节

祝训练顺利！
