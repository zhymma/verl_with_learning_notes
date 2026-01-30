# Router Replay - MoE 模型路由重放

[English](README.md) | 简体中文

## 概述

Router Replay 是 Verl 框架中为专家混合（Mixture of Experts, MoE）模型设计的高级路由重放功能。它通过记录和重放路由决策，实现确定性训练，确保训练过程的可复现性和一致性。

在 MoE 模型中，路由器（Router）决定每个 token 分配给哪些专家。Router Replay 技术可以记录这些路由决策，并在后续的训练阶段重放，从而保持训练的确定性。

## 主要特点

### 多种操作模式

- **`disabled`**: 完全禁用路由重放功能（默认模式）
- **`R2`**: 标准路由重放模式，用于记录和重放路由决策
- **`R3`**: 专为强化学习工作流优化的 Rollout 专用路由重放模式

### 核心能力

- **无缝集成**: 与强化学习管道（包括 PPO、GRPO 等）完美集成
- **分布式训练支持**: 兼容多 GPU 和多节点训练环境
- **灵活配置**: 通过 YAML 文件或命令行参数轻松配置
- **确定性训练**: 确保 MoE 模型在多次运行中的一致性
- **调试友好**: 便于调试和分析 MoE 模型的路由行为

## 适用场景

Router Replay 技术适用于以下场景：

1. **MoE 模型训练**: 任何使用专家混合架构的模型（如 DeepSeek-V2、Qwen-MoE 等）
2. **可复现性研究**: 需要确保实验结果完全可复现
3. **分布式 RLHF**: 在多节点环境中保持路由一致性
4. **调试和分析**: 分析路由决策对模型性能的影响
5. **A/B 测试**: 在相同路由决策下比较不同的训练策略

## 技术原理

### MoE 路由机制

在 MoE 模型中：
1. 每个 token 经过路由器（Router）
2. 路由器计算每个专家的分数
3. 选择 Top-K 个专家处理该 token
4. 专家输出经过加权组合

由于路由选择涉及随机性或数值精度问题，可能导致：
- 不同运行之间的结果不一致
- 分布式训练中的同步问题
- 难以复现和调试

### Router Replay 解决方案

Router Replay 通过以下机制解决这些问题：

**R2 模式** (Record & Replay):
1. **记录阶段**: 在首次前向传播时记录路由决策
2. **重放阶段**: 在后续的前向传播（如 PPO 的多轮更新）中使用记录的路由

**R3 模式** (Rollout-specific Replay):
1. **Rollout 记录**: 在推理阶段记录路由决策
2. **训练重放**: 在训练阶段使用 rollout 的路由决策
3. **确保一致性**: Rollout 和训练使用完全相同的专家选择

## 快速开始

### 环境准备

1. 安装 veRL 及其依赖：

```bash
# 安装 veRL 和 vLLM（支持 Router Replay）
pip install -e .[test,vllm]

# 或使用 SGLang
pip install -e .[test,sglang]
```

2. 确保您的模型是 MoE 架构：

```bash
# 支持的 MoE 模型示例
# - DeepSeek-V2 系列
# - Qwen-MoE 系列
# - Mixtral 系列
# - 其他支持 MoE 的模型
```

### 配置方法

#### 方法 1: 配置文件方式

创建或修改训练配置文件：

```yaml
# 针对 Actor (训练)
actor:
  router_replay:
    mode: "R2"                    # 或 "R3", "disabled"
    record_file: null             # 可选：指定记录文件路径
    replay_file: null             # 可选：指定重放文件路径

# 针对 Rollout (推理)
rollout:
  enable_rollout_routing_replay: True  # 启用 rollout 路由重放
```

#### 方法 2: 命令行方式

通过命令行参数启用：

```bash
# 启用 R2 模式
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.router_replay.mode="R2" \
    # ... 其他参数

# 启用 R3 模式（推荐用于 RLHF）
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.router_replay.mode="R3" \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
    # ... 其他参数
```

## 详细配置说明

### RouterReplayConfig 参数

```yaml
router_replay:
  mode: "disabled"              # 模式: disabled, R2, R3
  record_file: null             # 记录文件路径（可选）
  replay_file: null             # 重放文件路径（可选）
```

#### Mode 参数

**`disabled`** (默认):
- 完全禁用路由重放
- 路由器正常工作，不记录或重放
- 适用于非 MoE 模型或不需要确定性的场景

**`R2`** (标准模式):
- 记录并重放路由决策
- 适用于标准 PPO 多轮更新场景
- 确保每轮更新使用相同的路由

使用场景：
```bash
# Actor 进行多轮 PPO 更新
ppo_epochs=4                    # 4 轮更新都使用相同路由
router_replay.mode="R2"         # 第1轮记录，第2-4轮重放
```

**`R3`** (Rollout 优化模式):
- 从 Rollout 阶段获取路由决策
- 训练阶段直接使用 Rollout 的路由
- 确保推理和训练的路由完全一致
- 适用于 RLHF 工作流

使用场景：
```bash
# 1. Rollout 生成数据并记录路由
# 2. 训练时重放 Rollout 的路由
router_replay.mode="R3"
enable_rollout_routing_replay=True
```

#### 文件路径参数（可选）

```yaml
router_replay:
  record_file: "path/to/router_record.pt"   # 记录文件
  replay_file: "path/to/router_replay.pt"   # 重放文件
```

通常情况下，这些参数可以保持为 `null`，系统会自动管理内存中的路由记录。只有在需要持久化存储或跨进程共享时才需要指定。

### Rollout 配置

```yaml
rollout:
  enable_rollout_routing_replay: True       # 启用 rollout 路由重放
```

这个参数告诉推理引擎（vLLM 或 SGLang）返回路由选择结果。

**注意**: 需要推理后端的支持：
- vLLM: 需要支持路由返回的版本（参考 [PR #28284](https://github.com/vllm-project/vllm/pull/28284)）
- SGLang: 需要支持路由返回的版本（参考 [commit bed301a](https://github.com/sgl-project/sglang/commit/bed301a5acaa9577c9aa706468bdf242f6a43051)）

## 运行示例

### 示例 1: 使用 R2 模式训练 DeepSeek-MoE

```bash
#!/usr/bin/env bash

MODEL_PATH="deepseek-ai/DeepSeek-V2-Lite"
TRAIN_FILE="data/train.parquet"
TEST_FILE="data/test.parquet"

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.router_replay.mode="R2" \
    actor_rollout_ref.actor.ppo_epochs=4 \
    algorithm.adv_estimator=gae \
    trainer.total_epochs=10 \
    # ... 其他参数
```

**工作流程**:
1. 第 1 次 PPO 更新：记录路由决策
2. 第 2-4 次 PPO 更新：重放相同的路由
3. 下一个批次：重新记录新的路由

### 示例 2: 使用 R3 模式进行 RLHF

```bash
#!/usr/bin/env bash

MODEL_PATH="Qwen/Qwen2.5-57B-A14B-Instruct"  # MoE 模型
TRAIN_FILE="data/train.parquet"
TEST_FILE="data/test.parquet"

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.router_replay.mode="R3" \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
    actor_rollout_ref.rollout.name=vllm \
    algorithm.adv_estimator=grpo \
    trainer.total_epochs=5 \
    # ... 其他参数
```

**工作流程**:
1. **Rollout 阶段**: vLLM 生成响应并记录路由决策
2. **训练阶段**: Actor 使用 Rollout 的路由进行前向传播
3. **一致性**: 确保推理和训练看到相同的专家激活

### 示例 3: 多节点 MoE 训练

```bash
#!/usr/bin/env bash

export NNODES=4
export NGPUS_PER_NODE=8

MODEL_PATH="deepseek-ai/DeepSeek-V2-Chat"
TRAIN_FILE="data/train.parquet"

bash examples/router_replay/run_qwen30_a3b_megatron_vllm.sh

# 脚本内容：
python3 -m verl.trainer.main_ppo \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    data.train_files="${TRAIN_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.router_replay.mode="R3" \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    # ... 其他参数
```

### 示例 4: 完整的 RLHF 工作流配置

**完整脚本** (`run_qwen30_a3b_megatron_vllm.sh`):

```bash
#!/usr/bin/env bash
set -xeuo pipefail

# Model and Data
MODEL_PATH="Qwen/Qwen2.5-57B-A14B-Instruct"
TRAIN_FILE="data/train.parquet"
TEST_FILE="data/test.parquet"

# MoE Router Replay Configuration
ROUTER_REPLAY_MODE="R3"
ENABLE_ROLLOUT_ROUTING_REPLAY=True

# Training Configuration
NNODES=4
NGPUS_PER_NODE=8
TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=64

# Parallelism Strategy
GEN_TP=8        # Rollout 张量并行
TRAIN_TP=4      # 训练张量并行
TRAIN_PP=2      # 流水线并行

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.actor.router_replay.mode="${ROUTER_REPLAY_MODE}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=${ENABLE_ROLLOUT_ROUTING_REPLAY} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    \
    algorithm.adv_estimator=grpo \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.total_epochs=10 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="moe_router_replay"
```

## 常见问题

### 1. 何时使用 R2 vs R3？

**使用 R2 模式**:
- ✅ 标准 PPO 训练（多轮更新）
- ✅ 不使用独立的 Rollout 引擎
- ✅ 简单的训练设置

**使用 R3 模式**:
- ✅ RLHF 工作流（Rollout + Training 分离）
- ✅ 使用 vLLM 或 SGLang 进行推理
- ✅ 需要确保推理和训练的严格一致性

### 2. Router Replay 对性能的影响？

**计算开销**:
- **R2 模式**: 几乎无额外开销（~1-2%）
- **R3 模式**: 无训练时开销，推理时需要记录路由（~2-3%）

**内存开销**:
- 路由信息通常很小（每个 token 几个整数）
- 对于 4K 序列，通常 <1MB 每个样本

### 3. 如何验证 Router Replay 是否生效？

**检查日志**:
```bash
# 查看训练日志，应该看到类似的消息
[INFO] Router Replay R3 mode enabled
[INFO] Recording routing decisions during rollout
[INFO] Replaying routing decisions during training
```

**验证确定性**:
```bash
# 运行两次相同的训练，结果应该完全一致
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.router_replay.mode="R2" \
    trainer.seed=42 \
    # ... 其他参数

# 第二次运行应该产生完全相同的结果
```

### 4. 推理后端不支持路由返回怎么办？

如果您的 vLLM/SGLang 版本不支持路由返回：

**选项 1**: 升级到支持的版本
```bash
# vLLM: 使用包含 PR #28284 的版本
pip install vllm>=0.x.x

# SGLang: 使用包含相应 commit 的版本
pip install sglang>=0.x.x
```

**选项 2**: 使用 R2 模式
```bash
# R2 模式不依赖推理后端的路由返回
actor_rollout_ref.actor.router_replay.mode="R2"
# 不需要设置 enable_rollout_routing_replay
```

**选项 3**: 禁用 Router Replay
```bash
# 如果不需要严格的确定性
actor_rollout_ref.actor.router_replay.mode="disabled"
```

### 5. 分布式训练中的同步问题？

Router Replay 自动处理分布式训练的同步：
- 路由决策在所有设备间同步
- 确保不同 rank 使用相同的路由
- 支持张量并行和流水线并行

**注意事项**:
```bash
# 确保所有节点使用相同的配置
actor_rollout_ref.actor.router_replay.mode="R3"
actor_rollout_ref.rollout.enable_rollout_routing_replay=True

# 在所有节点上保持一致
```

### 6. R3 模式下 Rollout 和训练的并行度不同？

这是正常的，Router Replay 会处理这种情况：

```bash
# Rollout 使用更大的张量并行
actor_rollout_ref.rollout.tensor_model_parallel_size=8

# 训练使用较小的张量并行
actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4

# Router Replay 会自动处理路由信息的重分布
```

### 7. 如何调试路由决策？

**启用详细日志**:
```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.router_replay.mode="R3" \
    trainer.logger='["console","tensorboard"]' \
    # 查看 TensorBoard 中的路由统计
```

**导出路由信息**:
```bash
# 指定记录文件以便分析
actor_rollout_ref.actor.router_replay.record_file="router_records.pt"

# 分析路由模式
python analyze_router.py --input router_records.pt
```

### 8. MoE 负载均衡与 Router Replay 的关系？

Router Replay 与负载均衡是独立的：
- Router Replay 确保确定性和一致性
- 负载均衡损失仍然正常工作
- 两者可以同时启用

```bash
# 同时启用 Router Replay 和负载均衡
actor_rollout_ref.actor.router_replay.mode="R3"
actor_rollout_ref.actor.use_load_balance_loss=True
actor_rollout_ref.actor.load_balance_coef=0.01
```

## 性能优化建议

### 1. 选择合适的模式

```bash
# 对于简单场景，使用 R2
actor_rollout_ref.actor.router_replay.mode="R2"

# 对于 RLHF，使用 R3
actor_rollout_ref.actor.router_replay.mode="R3"
```

### 2. 优化内存使用

Router Replay 的内存使用通常很小，但对于超长序列：

```bash
# 减少批次大小以节省内存
data.train_batch_size=128           # 而不是 256

# 使用更积极的序列长度限制
data.max_response_length=2048       # 而不是 4096
```

### 3. 确保版本兼容性

```bash
# 使用支持路由返回的 vLLM/SGLang 版本
pip install vllm>=0.x.x --upgrade

# 或使用最新的开发版本
pip install git+https://github.com/vllm-project/vllm.git
```

## 参考资料

### 推理后端支持

- **vLLM Router Replay PR**: [vllm#28284](https://github.com/vllm-project/vllm/pull/28284)
- **SGLang Router Support**: [commit bed301a](https://github.com/sgl-project/sglang/commit/bed301a5acaa9577c9aa706468bdf242f6a43051)

### 相关论文

1. **MoE 架构**:
   - [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668)
   - [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)

2. **DeepSeek-V2**:
   - [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)

### 相关示例

- `examples/ppo_trainer/`: 标准 PPO 训练
- `examples/grpo_trainer/`: GRPO 训练
- `examples/prefix_grouper/`: 前缀分组优化

### 实现文件

- `verl/workers/config/actor.py`: Router Replay 配置
- `verl/workers/actor/*.py`: Router Replay 实现
- `verl/trainer/config/`: 训练配置

## 总结

Router Replay 为 MoE 模型训练提供了关键的确定性保证：

1. **R2 模式**: 适合标准多轮 PPO 更新
2. **R3 模式**: 专为 RLHF 工作流优化
3. **无缝集成**: 与现有训练流程完美结合
4. **性能开销小**: 几乎无额外计算和内存开销
5. **分布式支持**: 自动处理多节点训练的同步

对于 MoE 模型的 RLHF 训练，强烈推荐启用 Router Replay 以确保训练的稳定性和可复现性。

## 支持与反馈

如果遇到问题或有改进建议：

1. 查看 [veRL 文档](https://verl.readthedocs.io/)
2. 提交 [GitHub Issue](https://github.com/volcengine/verl/issues)
3. 参考推理后端的相关 PR 和文档

祝训练顺利！
