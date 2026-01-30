# Split Placement - 模型分离部署示例

[English](README.md) | 简体中文

## 概述

Split Placement 示例展示了如何在 PPO 算法中实现模型的分离部署（Split Placement）。通过将不同的模型组件（Actor、Critic、Reference、Reward Model）部署到不同的 GPU 上，并让它们异步执行，可以提高资源利用率和训练效率。

本示例提供了 Split Placement 的基础实现。完整版的灵活部署功能将在不久的将来发布。

## 主要特点

- **模型分离**: Actor/Ref 和 Critic/RM 部署到不同的 GPU
- **异步执行**: 模型更新可以并行进行
- **资源优化**: 更好地利用多 GPU 资源
- **灵活配置**: 支持多种部署策略

## 适用场景

Split Placement 适用于：

1. **资源充足的环境**: 有足够的 GPU 可以分离部署模型
2. **Actor 和 Critic 大小相似**: 两者都需要相当的计算资源
3. **需要并行化**: 想要 Actor 和 Critic 同时更新
4. **多节点训练**: 跨多个节点分配模型

## 快速开始

### 最简步骤（推荐）

如果您只是想快速尝试 Split Placement，只需按照以下步骤：

**步骤 1**: 修改代码（步骤 2）
**步骤 2**: 执行示例（步骤 4）

### 完整步骤

#### 步骤 1: 将模型放置到不同的 GPU

指定部署和资源分配。在示例中，我们将 actor 和 reference 放置在前一半的 GPU 上，而将 critic 和 reward model（如果有）映射到后一半的 GPU 上。

```python
actor_rollout_ref_pool_id = 'actor_rollout_ref_pool'
critic_pool_id = 'critic_pool'

if config.trainer.nnodes // 2 == 0 and config.trainer.n_gpus_per_node // 2 > 0:
    # 单节点：将每个节点的 GPU 分成两半
    resource_pool_spec = {
        actor_rollout_ref_pool_id: [config.trainer.n_gpus_per_node // 2] * config.trainer.nnodes,
        critic_pool_id: [config.trainer.n_gpus_per_node // 2] * config.trainer.nnodes,
    }
else:
    # 多节点：将节点分成两组
    resource_pool_spec = {
        actor_rollout_ref_pool_id: [config.trainer.n_gpus_per_node] * (config.trainer.nnodes // 2),
        critic_pool_id: [config.trainer.n_gpus_per_node] * (config.trainer.nnodes // 2),
    }

print(f'resource_pool_spec: {resource_pool_spec}')

# 定义角色到资源池的映射
mapping = {
    Role.ActorRollout: actor_rollout_ref_pool_id,
    Role.Critic: critic_pool_id,
    Role.RefPolicy: actor_rollout_ref_pool_id,
}
mapping[Role.RewardModel] = critic_pool_id
```

**部署策略示例**:

**单节点 8 GPU**:
```python
# GPU 0-3: Actor + Ref
# GPU 4-7: Critic + RM
resource_pool_spec = {
    'actor_rollout_ref_pool': [4],
    'critic_pool': [4],
}
```

**2 节点 16 GPU**:
```python
# 选项 A: 每个节点分离
resource_pool_spec = {
    'actor_rollout_ref_pool': [4, 4],  # 每个节点 4 GPU
    'critic_pool': [4, 4],
}

# 选项 B: 节点分离
resource_pool_spec = {
    'actor_rollout_ref_pool': [8],     # 节点 0 的所有 8 GPU
    'critic_pool': [8],                # 节点 1 的所有 8 GPU
}
```

#### 步骤 2: 使模型异步执行

基于模型部署，我们需要使模型异步执行。

为此，您需要在模型操作的装饰器中关闭 `blocking` 标志（即设置 `blocking=False`）。

例如，我们希望 actor 更新和 critic 更新可以并行执行，那么我们需要在 `fsdp_workers.py` 中进行以下修改：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def update_actor(self, data: DataProto):
    ...

@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def update_critic(self, data: DataProto):
    ...
```

**说明**:
- `blocking=False`: 方法立即返回一个 future，不等待完成
- `blocking=True` (默认): 方法等待完成后才返回

我们也可以在 split placement 中并行化 `ref_log_prob`、`values` 和 `rewards` 的计算。为了简化本教程，我们在此示例中不这样做。

#### 步骤 3: 在单控制器进程中并行执行这些操作

要实现 actor 和 critic 更新的并行执行，我们唯一需要在 `ray_trainer.py` 中修改的是在单控制器进程中 `get` 并发的 `futures`。

```python
# 启动并发操作
critic_output = worker_group.update_critic(data)  # 返回 future
actor_output = worker_group.update_actor(data)    # 返回 future

# 并行等待完成
critic_output = critic_output.get()  # 阻塞直到 critic 完成
actor_output = actor_output.get()    # 阻塞直到 actor 完成
```

**工作流程**:
```
时间线:
t0: 启动 critic_update (异步)
t0: 启动 actor_update (异步)
t1: critic 和 actor 并行执行
t2: critic 完成，critic_output.get() 返回
t3: actor 完成，actor_output.get() 返回
```

#### 步骤 4: 运行 Split Placement 示例

```bash
cd examples/split_placement
bash run_deepseek7b_llm.sh
```

## 详细配置说明

### 1. 资源池配置

在 `main_ppo_split.py` 中配置资源池：

```python
# 示例：2 节点，每节点 8 GPU
resource_pool_spec = {
    'actor_rollout_ref_pool': [4, 4],  # 每个节点前 4 个 GPU
    'critic_pool': [4, 4],              # 每个节点后 4 个 GPU
}

# 或者：节点级分离
resource_pool_spec = {
    'actor_rollout_ref_pool': [8],      # 节点 0 的全部 GPU
    'critic_pool': [8],                 # 节点 1 的全部 GPU
}
```

### 2. 角色映射配置

```python
from verl.single_controller.ray.base import Role

mapping = {
    Role.ActorRollout: 'actor_rollout_ref_pool',  # Actor 和 Rollout
    Role.RefPolicy: 'actor_rollout_ref_pool',     # Reference Policy
    Role.Critic: 'critic_pool',                   # Critic
    Role.RewardModel: 'critic_pool',              # Reward Model (可选)
}
```

### 3. 异步操作配置

在 `verl/workers/fsdp_workers.py` 中修改：

```python
# Actor 更新 - 异步
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def update_actor(self, data: DataProto):
    # ... actor 更新逻辑
    return actor_output

# Critic 更新 - 异步
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def update_critic(self, data: DataProto):
    # ... critic 更新逻辑
    return critic_output

# Reference log prob - 可以保持同步或改为异步
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=True)
def compute_ref_log_prob(self, data: DataProto):
    # ... ref log prob 计算
    return ref_output
```

### 4. Trainer 配置

在 `verl/trainer/ray_trainer.py` 中修改：

```python
# 标准方式（同步）
critic_output = worker_group.update_critic(data).get()
actor_output = worker_group.update_actor(data).get()

# Split Placement 方式（异步）
critic_future = worker_group.update_critic(data)  # 不立即 get()
actor_future = worker_group.update_actor(data)    # 不立即 get()

# 并行等待
critic_output = critic_future.get()
actor_output = actor_future.get()
```

## 运行示例

### 示例 1: 基本 Split Placement (单节点)

```bash
#!/usr/bin/env bash

MODEL_PATH="deepseek-ai/deepseek-llm-7b-chat"
TRAIN_FILE="data/train.parquet"

python3 -m verl.trainer.main_ppo_split \
    data.train_files="${TRAIN_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    algorithm.adv_estimator=gae \
    # ... 其他参数
```

**GPU 分配**:
- GPU 0-3: Actor + Reference
- GPU 4-7: Critic + Reward Model

### 示例 2: 多节点 Split Placement

```bash
#!/usr/bin/env bash

export NNODES=2
export NGPUS_PER_NODE=8

python3 -m verl.trainer.main_ppo_split \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    # ... 其他参数
```

**GPU 分配选项**:

**选项 A - 每节点分离**:
- 节点 0: GPU 0-3 (Actor/Ref), GPU 4-7 (Critic)
- 节点 1: GPU 0-3 (Actor/Ref), GPU 4-7 (Critic)

**选项 B - 节点级分离**:
- 节点 0: GPU 0-7 (Actor/Ref)
- 节点 1: GPU 0-7 (Critic)

### 示例 3: 自定义资源分配

```python
# 不对称分配：Actor 需要更多资源
resource_pool_spec = {
    'actor_rollout_ref_pool': [6, 6],   # 每节点 6 GPU
    'critic_pool': [2, 2],               # 每节点 2 GPU
}

# 或者：Critic 需要更多资源
resource_pool_spec = {
    'actor_rollout_ref_pool': [3, 3],
    'critic_pool': [5, 5],
}
```

## 性能分析

### 理论加速

**标准 PPO (顺序执行)**:
```
总时间 = T_rollout + T_ref_log_prob + T_values + T_rewards + T_actor + T_critic
```

**Split Placement (并行执行)**:
```
总时间 = T_rollout + T_ref_log_prob + T_values + T_rewards + max(T_actor, T_critic)
```

**加速比**:
```
Speedup = (T_actor + T_critic) / max(T_actor, T_critic)
```

最佳情况（T_actor ≈ T_critic）:
```
Speedup ≈ 2x
```

### 实际性能

实际加速取决于多个因素：

1. **Actor 和 Critic 的相对大小**
   - 如果 Actor >> Critic，加速有限
   - 如果 Actor ≈ Critic，接近 2x 加速

2. **通信开销**
   - 单节点：低通信开销
   - 多节点：可能有网络延迟

3. **负载均衡**
   - 理想情况：两个池的 GPU 利用率都接近 100%
   - 不平衡：较小模型的 GPU 可能空闲

## 常见问题

### 1. 何时应该使用 Split Placement？

**适合的场景**:
- ✅ Actor 和 Critic 大小相似
- ✅ 有充足的 GPU 资源
- ✅ Actor 和 Critic 更新时间相当
- ✅ 需要最大化 GPU 利用率

**不适合的场景**:
- ❌ Actor >> Critic 或 Critic >> Actor
- ❌ GPU 资源有限
- ❌ 简单的训练设置

### 2. 如何验证 Split Placement 是否生效？

**检查日志**:
```bash
# 应该看到资源池创建信息
[INFO] Creating resource pool: actor_rollout_ref_pool with [4, 4] GPUs
[INFO] Creating resource pool: critic_pool with [4, 4] GPUs

# 应该看到并行执行信息
[INFO] Starting parallel actor and critic updates
[INFO] Actor update completed in 5.2s
[INFO] Critic update completed in 5.1s
```

**监控 GPU 使用**:
```bash
# 在训练期间运行
watch -n 1 nvidia-smi

# 应该看到两组 GPU 同时活跃
```

### 3. 如何调试 Split Placement？

**启用详细日志**:
```bash
export VERL_LOG_LEVEL=DEBUG
python3 -m verl.trainer.main_ppo_split ...
```

**检查资源分配**:
```python
# 在代码中添加打印
print(f'Resource pool spec: {resource_pool_spec}')
print(f'Role mapping: {mapping}')
```

**验证异步执行**:
```python
# 在 ray_trainer.py 中添加计时
import time
start = time.time()
critic_future = worker_group.update_critic(data)
actor_future = worker_group.update_actor(data)
print(f'Launch time: {time.time() - start}s')  # 应该很短

start = time.time()
critic_output = critic_future.get()
print(f'Critic time: {time.time() - start}s')

start = time.time()
actor_output = actor_future.get()
print(f'Actor time: {time.time() - start}s')  # 应该很短（已完成）
```

### 4. 资源分配不平衡怎么办？

如果一个池的 GPU 经常空闲：

**选项 1: 调整资源分配**
```python
# 如果 Actor 更慢，给它更多 GPU
resource_pool_spec = {
    'actor_rollout_ref_pool': [6, 6],
    'critic_pool': [2, 2],
}
```

**选项 2: 调整模型并行度**
```bash
# 增加较大模型的并行度
actor_rollout_ref.actor.fsdp_config.fsdp_size=4
critic.fsdp_config.fsdp_size=2
```

**选项 3: 回退到标准部署**
```bash
# 如果不平衡太严重，禁用 Split Placement
python3 -m verl.trainer.main_ppo ...  # 使用标准 main_ppo
```

### 5. 多节点 Split Placement 的网络要求？

**网络带宽**:
- 节点内: PCIe/NVLink (高带宽)
- 节点间: InfiniBand 或高速以太网 (推荐 ≥100 Gbps)

**延迟**:
- 低延迟网络（<10μs）效果最好
- 高延迟可能抵消并行化的收益

### 6. 可以并行化更多操作吗？

是的，您可以扩展 Split Placement：

```python
# 并行化 ref_log_prob、values、rewards
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def compute_ref_log_prob(self, data: DataProto):
    ...

@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def compute_values(self, data: DataProto):
    ...

@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def compute_rewards(self, data: DataProto):
    ...

# 在 trainer 中
ref_future = worker_group.compute_ref_log_prob(data)
values_future = worker_group.compute_values(data)
rewards_future = worker_group.compute_rewards(data)

# 并行等待所有
ref_output = ref_future.get()
values_output = values_future.get()
rewards_output = rewards_future.get()
```

## 最佳实践

### 1. 资源分配策略

```python
# 平衡分配（推荐起点）
resource_pool_spec = {
    'actor_rollout_ref_pool': [4, 4],
    'critic_pool': [4, 4],
}

# 根据实际性能调整
# - 监控 GPU 利用率
# - 调整分配以平衡负载
```

### 2. 异步操作选择

```python
# 一定要并行化的（收益最大）
update_actor: blocking=False
update_critic: blocking=False

# 可选并行化的
compute_ref_log_prob: blocking=False  # 如果时间显著
compute_values: blocking=False
compute_rewards: blocking=False

# 保持同步的（通常很快）
generate_sequences: blocking=True
```

### 3. 性能监控

```bash
# 监控 GPU 利用率
nvidia-smi dmon -s u

# 监控网络（多节点）
iftop -i eth0

# 分析 Ray 时间线
# 在 ray_trainer.py 中启用 timeline
```

## 参考资料

### 相关示例

- `examples/ppo_trainer/`: 标准 PPO 训练
- `examples/grpo_trainer/`: GRPO 训练

### 实现文件

- `examples/split_placement/main_ppo_split.py`: Split Placement 主程序
- `examples/split_placement/split_monkey_patch.py`: Monkey patch 实现
- `verl/single_controller/ray/base.py`: Ray 资源管理
- `verl/trainer/ray_trainer.py`: Ray 训练器
- `verl/workers/fsdp_workers.py`: FSDP worker 实现

## 总结

Split Placement 提供了一种灵活的模型部署策略：

1. **并行化**: Actor 和 Critic 同时更新
2. **资源优化**: 更好地利用多 GPU 环境
3. **可扩展**: 支持单节点和多节点
4. **灵活配置**: 支持多种部署策略

在合适的场景下（Actor ≈ Critic 大小），可以实现接近 2x 的加速。完整版的灵活部署功能将提供更多优化选项。

## 支持与反馈

如果遇到问题或有改进建议：

1. 查看 [veRL 文档](https://verl.readthedocs.io/)
2. 提交 [GitHub Issue](https://github.com/volcengine/verl/issues)
3. 参考示例代码进行配置

祝训练顺利！
