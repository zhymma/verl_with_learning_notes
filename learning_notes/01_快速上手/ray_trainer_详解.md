# RayPPOTrainer 训练主循环详解

> 深入解析 verl 的核心训练流程，理解每一步的工作原理

---

## 📋 本文内容

1. RayPPOTrainer 类概览
2. 初始化流程（__init__）
3. 训练主循环（fit）
4. 单步训练（_train_step）
5. 数据流转详解
6. 实际例子追踪

---

## 1. RayPPOTrainer 类概览

### 1.1 文件位置

**主文件：** `verl/trainer/ppo/ray_trainer.py` (约2500行)

**关键类：**
```python
class RayPPOTrainer:
    """Ray-based PPO/GRPO trainer

    核心职责：
    1. 管理分布式资源（GPU）
    2. 创建和协调多个 WorkerGroup（Actor, Critic, Rollout）
    3. 执行训练循环
    4. 收集和记录指标
    """
```

### 1.2 类的整体结构

```python
# verl/trainer/ppo/ray_trainer.py

class RayPPOTrainer:
    def __init__(self, config):
        """初始化：创建资源池和 Worker 组"""

    def fit(self):
        """主训练循环：迭代 epoch 和 batch"""

    def _train_step(self, batch):
        """单步训练：rollout → reward → advantage → update"""

    def _compute_reward(self, data):
        """计算 reward"""

    def _validate(self):
        """验证和评估"""
```

---

## 2. 初始化流程详解

### 2.1 初始化代码（简化版）

```python
# verl/trainer/ppo/ray_trainer.py: 第 100-400 行（简化）

def __init__(self, config):
    self.config = config

    # ========== 步骤 1: 创建资源池 ==========
    print("[Step 1] 创建 GPU 资源池...")
    self.resource_pool_manager = ResourcePoolManager(
        process_on_nodes=ray.cluster_resources(),
        use_gpu=True
    )
    self.resource_pool_manager.create_resource_pool()

    # ========== 步骤 2: 创建 Actor+Rollout+Ref WorkerGroup ==========
    print("[Step 2] 创建 Actor+Rollout+Ref WorkerGroup...")
    self.actor_rollout_wg = self._create_actor_rollout_worker_group()
    # 内部包含：
    # - Actor 模型（训练中）
    # - Rollout 引擎（vLLM/SGLang，用于生成）
    # - Reference 模型（固定，用于 KL 计算）

    # ========== 步骤 3: 创建 Critic WorkerGroup（如果启用）==========
    if config.critic.enable:
        print("[Step 3] 创建 Critic WorkerGroup...")
        self.critic_wg = self._create_critic_worker_group()

    # ========== 步骤 4: 创建 RewardManager ==========
    print("[Step 4] 创建 RewardManager...")
    self.reward_manager = RewardManager(...)

    # ========== 步骤 5: 创建 DataLoader ==========
    print("[Step 5] 创建 DataLoader...")
    self.train_dataloader = self._create_dataloader()

    print("初始化完成！")
```

### 2.2 资源池创建（ResourcePoolManager）

**代码位置：** `verl/single_controller/ray/base.py`

```python
# 资源池的作用：管理所有 GPU，分配给不同的 WorkerGroup

class ResourcePoolManager:
    def create_resource_pool(self):
        """
        假设有 8 张 GPU：

        资源池分配示例（Colocate 模式）：
        ┌─────────────────────────────────────┐
        │ GPU 0-3: Actor + Rollout + Ref      │  ← 共享 GPU
        │ GPU 4-7: Critic                     │
        └─────────────────────────────────────┘

        非 Colocate 模式：
        ┌─────────────────────────────────────┐
        │ GPU 0-3: Actor                      │
        │ GPU 4-5: Rollout                    │
        │ GPU 6-7: Critic                     │
        └─────────────────────────────────────┘
        """
        # 实际代码会根据配置智能分配
```

### 2.3 Actor+Rollout WorkerGroup 创建

```python
# verl/trainer/ppo/ray_trainer.py: 第 450-550 行（简化）

def _create_actor_rollout_worker_group(self):
    """创建 Actor+Rollout+Ref WorkerGroup

    这是 verl 的核心创新：将训练（Actor）和推理（Rollout）
    共享同一组 GPU，避免权重拷贝
    """

    # 1. 从配置创建 Worker 类
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    # 2. 创建 WorkerGroup（分布式）
    worker_group = RayWorkerGroup(
        resource_pool=self.actor_rollout_pool,  # GPU 资源
        ray_cls_with_init=ActorRolloutRefWorker,
        num_workers=4,  # 例如 4 个 worker，每个管理 1-2 张 GPU
    )

    # 3. 初始化 worker（加载模型）
    worker_group.init_model(
        model_path=self.config.actor_rollout_ref.model.path,
        enable_gradient_checkpointing=True,
    )

    return worker_group
```

**关键点：Actor、Rollout、Ref 在同一个 Worker 中！**

```
ActorRolloutRefWorker 内部结构：
┌─────────────────────────────────────────┐
│ ActorRolloutRefWorker (单个 Worker)     │
│                                         │
│  ┌─────────────┐                       │
│  │ Actor Model │ ← FSDP 包装，可训练   │
│  └─────────────┘                       │
│         ↕ 权重同步                      │
│  ┌─────────────┐                       │
│  │Rollout(vLLM)│ ← 推理引擎            │
│  └─────────────┘                       │
│         ↕                               │
│  ┌─────────────┐                       │
│  │ Ref Model   │ ← 固定，用于KL计算    │
│  └─────────────┘                       │
└─────────────────────────────────────────┘
```

---

## 3. 训练主循环（fit）

### 3.1 完整的 fit 方法

```python
# verl/trainer/ppo/ray_trainer.py: 第 1000-1100 行（简化）

def fit(self):
    """主训练循环"""

    for epoch in range(self.config.trainer.total_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}")
        print('='*60)

        # 遍历 DataLoader
        for batch_idx, batch in enumerate(self.train_dataloader):
            # 核心：单步训练
            metrics = self._train_step(batch)

            # 记录指标
            self._log_metrics(metrics, self.global_step)

            # 更新步数
            self.global_step += 1

            # 定期保存 checkpoint
            if self.global_step % self.config.trainer.save_freq == 0:
                self._save_checkpoint()

        # 每个 epoch 结束后验证
        if epoch % self.config.trainer.test_freq == 0:
            self._validate()
```

### 3.2 DataLoader 的数据格式

```python
# DataLoader 产生的 batch 格式：

batch = {
    'input_ids': tensor([batch_size, max_prompt_length]),  # Prompt tokens
    'attention_mask': tensor([batch_size, max_prompt_length]),
    'data_source': ['gsm8k', 'gsm8k', ...],  # 数据来源
    'ground_truth': ['42', '100', ...],  # 标准答案
}
```

---

## 4. 单步训练详解（_train_step）

这是 **最核心** 的函数！每次调用完成一轮完整的 RL 更新。

### 4.1 _train_step 完整流程

```python
# verl/trainer/ppo/ray_trainer.py: 第 1200-1500 行（带详细注释）

def _train_step(self, batch: Dict) -> Dict[str, Any]:
    """单步训练的完整流程

    输入：
        batch: 从 DataLoader 来的一批 prompts

    输出：
        metrics: 训练指标字典
    """

    # ==========================================
    # 阶段 1: Rollout（生成响应）
    # ==========================================
    print("[Phase 1] Rollout: 生成响应...")

    # 调用 Actor+Rollout WorkerGroup 的 generate_sequences 方法
    # 这会在所有 worker 上并行生成
    rollout_output = self.actor_rollout_wg.generate_sequences(
        prompts=batch,  # 输入 prompts
        temperature=self.config.actor_rollout_ref.rollout.temperature,
        top_p=self.config.actor_rollout_ref.rollout.top_p,
        max_new_tokens=self.config.data.max_response_length,
    )

    # rollout_output 包含：
    # - responses: 生成的 token IDs
    # - response_mask: 哪些 token 是有效的
    # - old_log_probs: 当前策略的 log prob（用于 PPO ratio 计算）

    # ==========================================
    # 阶段 2: Reward Computation（计算奖励）
    # ==========================================
    print("[Phase 2] Reward: 计算奖励...")

    rollout_output = self._compute_reward(rollout_output)

    # 现在 rollout_output 新增了：
    # - rewards: 每个 token 的 reward 分数

    # ==========================================
    # 阶段 3: Reference Log Prob（可选，用于 KL 惩罚）
    # ==========================================
    if self.config.algorithm.use_kl_in_reward:
        print("[Phase 3] Ref: 计算参考模型 log prob...")

        rollout_output = self.actor_rollout_wg.compute_ref_log_prob(
            rollout_output
        )

        # 新增：
        # - ref_log_probs: 参考模型的 log prob

    # ==========================================
    # 阶段 4: Value Computation（PPO 需要，GRPO 不需要）
    # ==========================================
    if self.config.critic.enable:
        print("[Phase 4] Critic: 计算 value...")

        rollout_output = self.critic_wg.compute_values(rollout_output)

        # 新增：
        # - values: Critic 预测的 value

    # ==========================================
    # 阶段 5: Advantage Computation（优势估计）
    # ==========================================
    print("[Phase 5] Advantage: 计算优势值...")

    rollout_output = self._compute_advantage(rollout_output)

    # 新增：
    # - advantages: 优势值（核心！）
    # - returns: 回报值（用于 critic 训练）

    # ==========================================
    # 阶段 6: Actor Update（更新策略）
    # ==========================================
    print("[Phase 6] Actor Update: 更新策略模型...")

    actor_metrics = self.actor_rollout_wg.update_actor(
        data=rollout_output,
        ppo_epochs=self.config.actor_rollout_ref.actor.ppo_epochs,
        ppo_mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
    )

    # actor_metrics 包含：
    # - actor/loss: Actor 损失
    # - actor/policy_loss: 策略损失
    # - actor/entropy: 熵（探索程度）
    # - actor/approx_kl: 近似 KL 散度

    # ==========================================
    # 阶段 7: Critic Update（更新价值函数，PPO 需要）
    # ==========================================
    critic_metrics = {}
    if self.config.critic.enable:
        print("[Phase 7] Critic Update: 更新价值函数...")

        critic_metrics = self.critic_wg.update_critic(
            data=rollout_output,
            ppo_epochs=self.config.critic.ppo_epochs,
        )

        # critic_metrics 包含：
        # - critic/loss: Critic 损失
        # - critic/value_loss: 价值损失

    # ==========================================
    # 阶段 8: 收集所有指标
    # ==========================================
    metrics = {
        **actor_metrics,
        **critic_metrics,
        'reward/mean': rollout_output.batch['rewards'].mean().item(),
        'kl/mean': (rollout_output.batch['old_log_probs'] -
                    rollout_output.batch['ref_log_probs']).mean().item(),
    }

    return metrics
```

### 4.2 数据在各阶段的变化

```
阶段 0（输入）:
  DataProto {
    batch: {
      'input_ids': [B, L_prompt]
    }
  }

阶段 1（Rollout后）:
  DataProto {
    batch: {
      'input_ids': [B, L_prompt]
      'responses': [B, L_response]          ← 新增
      'response_mask': [B, L_response]      ← 新增
      'old_log_probs': [B, L_response]      ← 新增
    }
  }

阶段 2（Reward后）:
  DataProto {
    ...（上面的所有字段）
    batch: {
      'rewards': [B, L_response]            ← 新增
    }
  }

阶段 3（Ref后，如果启用）:
  DataProto {
    ...
    batch: {
      'ref_log_probs': [B, L_response]      ← 新增
    }
  }

阶段 4（Value后，PPO）:
  DataProto {
    ...
    batch: {
      'values': [B, L_response]             ← 新增
    }
  }

阶段 5（Advantage后）:
  DataProto {
    ...
    batch: {
      'advantages': [B, L_response]         ← 新增
      'returns': [B, L_response]            ← 新增（PPO）
    }
  }
```

---

## 5. 详细例子：GSM8K 训练一步

让我们用一个具体例子追踪整个流程：

### 5.1 初始状态

```python
# 假设 batch_size = 2
batch = {
    'input_ids': tensor([
        [123, 456, 789, ...],  # "What is 25 * 4?"
        [234, 567, 890, ...],  # "Calculate 100 / 5"
    ]),
    'data_source': ['gsm8k', 'gsm8k'],
    'ground_truth': ['100', '20'],
}
```

### 5.2 阶段 1：Rollout

```python
# 调用 vLLM 生成
rollout_output = self.actor_rollout_wg.generate_sequences(batch)

# 生成的响应（示例）
rollout_output.batch = {
    'input_ids': [...],  # 原始 prompt
    'responses': tensor([
        [345, 678, 901, ...],  # "Let me think... 25 * 4 = 100"
        [456, 789, 012, ...],  # "100 / 5 = 20"
    ]),
    'response_mask': tensor([
        [1, 1, 1, ..., 1],
        [1, 1, 1, ..., 1],
    ]),
    'old_log_probs': tensor([
        [-0.5, -0.3, -0.4, ...],  # 每个 token 的 log prob
        [-0.6, -0.4, -0.5, ...],
    ]),
}
```

### 5.3 阶段 2：Reward

```python
# 计算 reward
rollout_output = self._compute_reward(rollout_output)

# RewardManager 内部：
# 1. Decode responses: "25 * 4 = 100", "100 / 5 = 20"
# 2. 调用 GSM8K reward 函数
# 3. 提取答案: "100", "20"
# 4. 对比 ground_truth: "100" vs "100" ✓, "20" vs "20" ✓
# 5. 生成 reward: 1.0, 1.0

rollout_output.batch['rewards'] = tensor([
    [0, 0, 0, ..., 1.0],  # 最后一个 token 给 reward
    [0, 0, 0, ..., 1.0],
])
```

### 5.4 阶段 5：Advantage（GRPO）

```python
# GRPO 计算：相对于组平均的优势
# 假设 group_size=4，每个 prompt 生成 4 个响应

# 对于第一个 prompt 的 4 个响应：
group_rewards = [1.0, 0.0, 1.0, 0.0]  # 2 个对，2 个错
group_mean = 0.5

advantages = [
    1.0 - 0.5 = 0.5,   # 好于平均
    0.0 - 0.5 = -0.5,  # 差于平均
    1.0 - 0.5 = 0.5,   # 好于平均
    0.0 - 0.5 = -0.5,  # 差于平均
]

# 这告诉模型：
# - 强化前两个和第三个响应（正优势）
# - 抑制第二个和第四个响应（负优势）
```

### 5.6 阶段 6：Actor Update

```python
# 计算 PPO loss
for mini_batch in split(rollout_output, mini_batch_size):
    # 1. 前向传播，获取新的 log_probs
    new_log_probs = actor_model(mini_batch)

    # 2. 计算 ratio
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 3. PPO clip
    clipped_ratio = torch.clamp(ratio, 1-clip, 1+clip)

    # 4. Policy loss
    loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    # 5. 反向传播
    loss.backward()
    optimizer.step()
```

---

## 6. 关键代码位置速查

| 功能 | 文件 | 行号范围 |
|------|------|---------|
| RayPPOTrainer 类定义 | `verl/trainer/ppo/ray_trainer.py` | 50-100 |
| __init__ 方法 | `verl/trainer/ppo/ray_trainer.py` | 100-400 |
| fit 方法 | `verl/trainer/ppo/ray_trainer.py` | 1000-1100 |
| _train_step 方法 | `verl/trainer/ppo/ray_trainer.py` | 1200-1500 |
| _compute_reward | `verl/trainer/ppo/ray_trainer.py` | 1600-1700 |
| _compute_advantage | `verl/trainer/ppo/ray_trainer.py` | 1800-1900 |
| ActorRolloutRefWorker | `verl/workers/fsdp_workers.py` | 100-800 |
| generate_sequences | `verl/workers/fsdp_workers.py` | 300-400 |
| update_actor | `verl/workers/fsdp_workers.py` | 500-600 |
| PPO loss 计算 | `verl/trainer/ppo/core_algos.py` | 200-300 |
| GRPO advantage 计算 | `verl/trainer/ppo/core_algos.py` | 400-500 |

---

## 7. 调试技巧

### 7.1 打印数据流

```python
# 在 _train_step 的各个阶段添加 print

def _train_step(self, batch):
    print(f"[Debug] Input batch shape: {batch['input_ids'].shape}")

    rollout_output = self.actor_rollout_wg.generate_sequences(batch)
    print(f"[Debug] After rollout: responses shape = {rollout_output.batch['responses'].shape}")

    rollout_output = self._compute_reward(rollout_output)
    print(f"[Debug] Reward mean: {rollout_output.batch['rewards'].mean()}")

    # ... 更多调试输出
```

### 7.2 使用 Ray Dashboard

```bash
# Ray 会自动启动 Dashboard
# 访问 http://localhost:8265

# 可以看到：
# - 每个 Worker 的 GPU 使用率
# - 每个函数的执行时间
# - 资源分配情况
```

### 7.3 保存中间结果

```python
# 在 _train_step 中保存中间数据
def _train_step(self, batch):
    rollout_output = self.actor_rollout_wg.generate_sequences(batch)

    # 保存到文件
    torch.save({
        'responses': rollout_output.batch['responses'],
        'old_log_probs': rollout_output.batch['old_log_probs'],
    }, f'debug_step_{self.global_step}.pt')

    # 后续可以加载分析
```

---

## 8. 常见问题

### Q1: Actor 和 Rollout 如何共享 GPU？

通过 **HybridEngine** 和 **ShardingManager**：
- 训练时：权重在 FSDP 格式
- 推理时：自动 reshard 到 vLLM 格式
- 无需手动拷贝权重

详见：`verl/workers/sharding_manager/`

### Q2: WorkerGroup 是如何并行的？

Ray 自动管理：
```python
# 当调用 worker_group.generate_sequences() 时
# Ray 会并行调用所有 worker 的方法
# 每个 worker 处理一部分数据
```

### Q3: 训练为什么这么慢？

主要瓶颈：
1. **Rollout 阶段**：vLLM 生成响应（最慢）
2. **Weight Resharding**：训练↔推理权重转换
3. **数据传输**：Worker 之间的数据通信

优化方法：
- 增加 GPU 数量
- 减小 response 长度
- 使用更快的推理引擎（SGLang）

---

## 9. 总结

RayPPOTrainer 的核心流程：

```
初始化 → 创建资源池 → 创建 WorkerGroup
    ↓
主循环 → 遍历 DataLoader
    ↓
单步训练 → Rollout → Reward → Advantage → Update
    ↓
记录指标 → 保存 Checkpoint → 验证
```

理解这个流程后，你就掌握了 verl 训练的核心！

---

## 📚 延伸阅读

- [Single Controller 架构](../配置系统详解.md)
- [Reward 系统详解](../../02_数据准备/reward_系统详解.md)
- [官方文档：Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
