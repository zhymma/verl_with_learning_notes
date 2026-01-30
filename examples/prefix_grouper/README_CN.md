# PrefixGrouper - GRPO 前缀分组优化

[English](README.md) | 简体中文

## 概述

**PrefixGrouper** 是一种优化技术，通过将具有相同提示（prompt）的样本分组，减少 GRPO 训练中的冗余计算。这是一个即插即用的高效 GRPO 训练工具，只需最小的代码修改即可实现计算减少、设备内存降低和训练加速。

> 官方仓库: [https://github.com/johncaged/PrefixGrouper](https://github.com/johncaged/PrefixGrouper)

## 技术原理

在当前主流的 GRPO 训练流程中，策略模型训练主要涉及复制前缀（通常是问题、多模态输入等）G 次。因此，当训练数据前缀足够长（例如长上下文推理、图像/长视频推理）时，训练期间的冗余计算变得不可忽视。

**PrefixGrouper** 将原始的冗余自注意力操作分解为：
- **前缀自注意力** (Prefix Self-Attention)
- **后缀拼接注意力** (Suffix Concat-Attention)

<h3 align="center">
    <img src="https://raw.githubusercontent.com/johncaged/PrefixGrouper/main/assets/images/method.jpg">
</h3>

## 主要特点

- **即插即用**: 无需修改模型代码
- **显著加速**: 长上下文场景下加速高达 1.7x
- **内存优化**: 减少设备内存消耗
- **自动适配**: 自动 patch transformers 的注意力函数

## 适用场景

PrefixGrouper 特别适用于：

1. **长上下文推理**: 提示长度 > 2K tokens
2. **多模态输入**: 图像、视频等长前缀
3. **GRPO/RLOO 训练**: 每个提示生成多个响应
4. **批量生成**: `rollout.n > 1` 的场景

## 安装

```bash
pip install prefix_grouper
```

## 限制和兼容性

### 当前限制

- ✅ 仅支持 **FSDP worker**（Megatron worker 暂不支持）
- ❌ 不兼容 `use_dynamic_bsz=True`
- ❌ 不兼容 `use_remove_padding=True` (Flash Attention V2 变长)
- ❌ 不兼容 `use_fused_kernels=True`
- ❌ 不兼容 Ulysses 序列并行 (`use_ulysses_sp=True`) 和 ring-attention

### 批次平衡支持

现在支持 `balance_batch=True` 的组级平衡，它将具有相同 uid 的样本保持在同一个 rank 上。但这要求 `batch_size % (world_size * rollout.n) == 0`。

**示例**:
- `world_size=8`, `rollout.n=4` → `batch_size` 必须是 32 的倍数

## 快速开始

### 1. 在配置中启用 PrefixGrouper

在训练配置中设置 `use_prefix_grouper=True`：

```yaml
actor_rollout_ref:
  actor:
    use_prefix_grouper: True
  model:
    use_remove_padding: False       # 必须禁用
```

可选启用批次平衡以获得更好的负载分布：

```yaml
trainer:
  balance_batch: True               # 现在支持组级平衡
```

### 2. 运行训练

使用提供的示例脚本：

```bash
bash examples/prefix_grouper/run_qwen3_prefix_grouper.sh
```

## 详细配置说明

### 基本配置

```bash
#!/usr/bin/env bash

MODEL_PATH="Qwen/Qwen2.5-4B-Instruct"
TRAIN_FILE="data/train.parquet"
TEST_FILE="data/test.parquet"

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.max_prompt_length=4096 \              # 长上下文
    data.max_response_length=2048 \
    data.train_batch_size=256 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_prefix_grouper=True \      # 启用 PrefixGrouper
    actor_rollout_ref.model.use_remove_padding=False \     # 必须禁用
    actor_rollout_ref.rollout.n=4 \                        # 每个提示4个响应
    algorithm.adv_estimator=grpo \
    trainer.balance_batch=True \                           # 启用批次平衡
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=5
```

### 关键参数

**必须设置**:
```bash
actor_rollout_ref.actor.use_prefix_grouper=True     # 启用 PrefixGrouper
actor_rollout_ref.model.use_remove_padding=False    # 禁用 padding 移除
```

**推荐设置**:
```bash
data.max_prompt_length=4096                         # 更长的提示更有收益
actor_rollout_ref.rollout.n=4                       # 多个响应以利用分组
trainer.balance_batch=True                          # 组级批次平衡
```

**不兼容的设置** (必须禁用):
```bash
actor_rollout_ref.actor.use_dynamic_bsz=False       # 禁用动态批次
actor_rollout_ref.model.use_remove_padding=False    # 禁用 padding 移除
actor_rollout_ref.actor.use_fused_kernels=False     # 禁用融合核
actor_rollout_ref.actor.use_ulysses_sp=False        # 禁用 Ulysses SP
```

## 工作原理

当设置 `use_prefix_grouper=True` 时，veRL 会自动 patch `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS` 中的注意力函数以支持 `prefix_grouper` 参数。无需修改模型代码。

Patch 包装每个注意力函数以：
1. 从 kwargs 中提取 `prefix_grouper`
2. 如果 `prefix_grouper` 为 None，调用原始注意力
3. 如果提供了 `prefix_grouper`，使用 PrefixGrouper 的优化注意力计算

### 优化原理

**标准 GRPO**:
```
对于每个提示生成的 n 个响应：
  - 完整前缀 (prefix) 重复 n 次
  - 每个响应计算完整的自注意力
  - 冗余计算: O(n * prefix_len²)
```

**PrefixGrouper**:
```
1. 前缀自注意力: 只计算一次
2. 后缀拼接注意力: n 个响应共享前缀表示
3. 优化计算: O(prefix_len²) + O(n * suffix_len²)
```

## 性能基准

**测试环境**: Qwen3-4B, 4×H800, `rollout.n=4`

### 4K 上下文长度

| 指标 | PrefixGrouper | 无PG | 加速比 |
|------|--------------|------|--------|
| `old_log_prob` | 1.31s | 1.70s | **1.30x** |
| `update_actor` | 4.80s | 6.07s | **1.26x** |
| `step` | 17.08s | 19.40s | **1.14x** |

### 8K 上下文长度

| 指标 | PrefixGrouper | 无PG | 加速比 |
|------|--------------|------|--------|
| `old_log_prob` | 1.69s | 2.63s | **1.56x** |
| `update_actor` | 5.98s | 10.18s | **1.70x** |
| `step` | 19.48s | 24.71s | **1.27x** |

**结论**: 随着上下文长度增加，加速效果更加明显。

## 运行示例

### 示例 1: 基本 GRPO + PrefixGrouper

```bash
#!/usr/bin/env bash
set -xeuo pipefail

MODEL_PATH="Qwen/Qwen2.5-4B-Instruct"
TRAIN_FILE="${HOME}/data/gsm8k/train.parquet"
TEST_FILE="${HOME}/data/gsm8k/test.parquet"

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.train_batch_size=128 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_prefix_grouper=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.rollout.n=4 \
    algorithm.adv_estimator=grpo \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=5
```

### 示例 2: 长上下文 + 批次平衡

```bash
python3 -m verl.trainer.main_ppo \
    data.max_prompt_length=8192 \              # 8K 上下文
    data.max_response_length=4096 \
    data.train_batch_size=256 \                # 确保是 32 的倍数
    actor_rollout_ref.actor.use_prefix_grouper=True \
    actor_rollout_ref.rollout.n=4 \
    trainer.n_gpus_per_node=8 \
    trainer.balance_batch=True \               # 启用批次平衡
    # ... 其他参数
```

### 示例 3: 多模态长前缀

```bash
# 适用于包含图像或视频的多模态任务
python3 -m verl.trainer.main_ppo \
    data.max_prompt_length=16384 \             # 超长前缀（包含视觉 tokens）
    data.max_response_length=1024 \
    actor_rollout_ref.actor.use_prefix_grouper=True \
    actor_rollout_ref.rollout.n=8 \            # 更多响应以利用分组
    # ... 其他参数
```

## 常见问题

### 1. 如何知道 PrefixGrouper 是否生效？

查看训练日志，应该看到类似的消息：
```
[INFO] PrefixGrouper enabled, patching attention functions
[INFO] Processing grouped prefixes: batch_size=128, n_groups=32
```

### 2. 我的加速效果不明显？

可能的原因：
- **前缀太短**: PrefixGrouper 在长前缀（>2K）时效果最好
- **n 太小**: `rollout.n` 应该 ≥ 4 以体现分组优势
- **不兼容的设置**: 检查是否启用了不兼容的功能

```bash
# 优化配置以获得更好的加速
data.max_prompt_length=4096     # 增加前缀长度
actor_rollout_ref.rollout.n=8   # 增加响应数量
```

### 3. balance_batch=True 报错？

确保批次大小满足约束：
```bash
batch_size % (world_size * rollout.n) == 0
```

**示例**:
```bash
# world_size=8, rollout.n=4
data.train_batch_size=256       # ✓ 256 % 32 = 0
data.train_batch_size=200       # ✗ 200 % 32 ≠ 0
```

### 4. 与 Megatron 后端兼容吗？

目前 PrefixGrouper 仅支持 FSDP worker，不支持 Megatron worker。

如果您需要使用 Megatron：
```bash
# 禁用 PrefixGrouper
actor_rollout_ref.actor.use_prefix_grouper=False
```

### 5. 可以与 Flash Attention 一起使用吗？

可以，但需要注意：
- ✅ 兼容标准 Flash Attention
- ❌ 不兼容 `use_remove_padding=True`（变长优化）

```bash
# 正确的配置
actor_rollout_ref.actor.use_prefix_grouper=True
actor_rollout_ref.model.use_remove_padding=False
```

### 6. 内存使用如何？

PrefixGrouper 通常会**降低**内存使用，因为：
- 前缀表示只计算一次
- 减少了冗余的中间激活

在长前缀场景下，内存节省可以达到 20-30%。

### 7. 如何选择最佳的 rollout.n？

**推荐值**:
```bash
# 短前缀 (<1K)
rollout.n=4

# 中等前缀 (1K-4K)
rollout.n=4-8

# 长前缀 (>4K)
rollout.n=8-16      # 更多响应以最大化分组优势
```

**权衡**:
- 更大的 n: 更好的分组效果，但每批次处理更多数据
- 更小的 n: 更快的批次迭代，但分组优势较小

## 性能优化建议

### 1. 最大化加速效果

```bash
# 使用长前缀
data.max_prompt_length=8192

# 使用多个响应
actor_rollout_ref.rollout.n=8

# 启用批次平衡
trainer.balance_batch=True

# 确保批次大小合适
data.train_batch_size=256       # 调整以满足约束
```

### 2. 内存优化

```bash
# PrefixGrouper 本身就节省内存
# 可以进一步优化：

# 使用梯度检查点
actor_rollout_ref.actor.use_activation_checkpointing=True

# 调整批次大小
data.train_batch_size=128       # 根据 GPU 内存调整
```

### 3. 调试和分析

```bash
# 启用详细日志
trainer.logger='["console","tensorboard"]'

# 比较有无 PrefixGrouper 的性能
# 运行两次实验并比较 step 时间
```

## 参考资料

### 官方资源

- **官方仓库**: [https://github.com/johncaged/PrefixGrouper](https://github.com/johncaged/PrefixGrouper)
- **论文/技术报告**: 查看官方仓库获取最新信息

### 相关示例

- `examples/grpo_trainer/`: GRPO 训练基础
- `examples/rloo_trainer/`: RLOO 训练（也可使用 PrefixGrouper）
- `examples/ppo_trainer/`: PPO 训练（PrefixGrouper 主要用于 GRPO/RLOO）

### 实现文件

- `prefix_grouper` package: PrefixGrouper 核心实现
- `verl/workers/fsdp_workers.py`: FSDP worker 集成
- `verl/trainer/config/`: 训练配置

## 总结

PrefixGrouper 是一个强大的优化工具，特别适合：

1. **长上下文 GRPO/RLOO 训练**: 提示长度 > 2K
2. **多响应场景**: `rollout.n ≥ 4`
3. **多模态任务**: 包含长前缀的图像/视频输入
4. **资源受限环境**: 节省计算和内存

通过简单的配置即可获得 1.1-1.7x 的加速，随着前缀长度增加，加速效果更加显著。

## 支持与反馈

如果遇到问题或有改进建议：

1. 查看 [PrefixGrouper 官方仓库](https://github.com/johncaged/PrefixGrouper)
2. 提交 [veRL GitHub Issue](https://github.com/volcengine/verl/issues)
3. 参考示例脚本进行配置

祝训练顺利！
