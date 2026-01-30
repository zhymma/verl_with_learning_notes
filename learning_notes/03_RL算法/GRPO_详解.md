# GRPO 算法详解

> Group Relative Policy Optimization - 无需 Critic 的强化学习算法

---

## 📖 目录

1. [GRPO 核心思想](#1-grpo-核心思想)
2. [源码深度解析](#2-源码深度解析)
3. [完整训练流程](#3-完整训练流程)
4. [配置参数详解](#4-配置参数详解)
5. [DrGRPO 变体](#5-drgrpo-变体)
6. [调试技巧](#6-调试技巧)
7. [常见问题](#7-常见问题)

---

## 1. GRPO 核心思想

### 1.1 什么是 GRPO？

**GRPO (Group Relative Policy Optimization)** 是一种简化的强化学习算法，由 DeepSeekMath 论文提出。

**核心特点：**
- ✅ **无需 Critic 模型**：不需要训练价值函数网络
- ✅ **基于组相对奖励**：使用同组样本的均值作为 baseline
- ✅ **训练速度快**：省去 Critic 训练时间和显存
- ✅ **适合结果导向任务**：数学推理、代码生成等

### 1.2 GRPO vs PPO

| 特性 | GRPO | PPO |
|------|------|-----|
| **Critic 模型** | ❌ 不需要 | ✅ 需要 |
| **Baseline** | 组内样本均值 | Critic 的价值函数 |
| **优势估计** | 相对于组均值 | GAE（时序差分） |
| **GPU 显存** | 更少（只训练 Actor） | 更多（Actor + Critic） |
| **训练速度** | 更快 | 较慢 |
| **适用场景** | 结果导向任务 | 过程导向任务 |

### 1.3 GRPO 工作流程

```
1. 对于每个 prompt，采样 n 个响应（形成一个"组"）
   ↓
2. 计算每个响应的奖励（通过 RewardManager）
   ↓
3. 在组内计算均值和标准差
   ↓
4. 归一化优势值：(reward - mean) / std
   ↓
5. 使用归一化的优势值更新策略
```

**关键配置：**
- `actor_rollout_ref.rollout.n >= 2`（每个 prompt 采样多个响应）
- `algorithm.adv_estimator=grpo`
- `actor_rollout_ref.actor.use_kl_loss=true`（使用 KL loss 而非 KL reward）

---

## 2. 源码深度解析

### 2.1 函数签名

```python
# 位置: verl/trainer/ppo/core_algos.py:266-330

@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    index: np.ndarray,                  # (bs,) - 分组索引
    epsilon: float = 1e-6,              # 数值稳定性
    norm_adv_by_std_in_grpo: bool = True,  # 是否除以标准差
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算 GRPO 的优势值（仅用于结果奖励）
    """
```

**参数说明：**
- `token_level_rewards`: 每个 token 的奖励（通常只有最后一个 token 有奖励）
- `response_mask`: 标记哪些 token 是响应部分（1）还是 prompt 部分（0）
- `index`: 分组索引，相同索引的响应属于同一个 prompt
- `norm_adv_by_std_in_grpo`: True=原始 GRPO，False=DrGRPO

### 2.2 第 1 步：计算总奖励

```python
# 代码位置: verl/trainer/ppo/core_algos.py:303
scores = token_level_rewards.sum(dim=-1)
```

**作用：**
将 token 级别的奖励求和，得到每个响应的总分。

**示例：**
```python
# 假设有 2 个 prompt，每个采样 2 个响应
token_level_rewards = torch.tensor([
    [0, 0, 0, 1.0],  # prompt 0, response 0 → score=1.0
    [0, 0, 0, 0.0],  # prompt 0, response 1 → score=0.0
    [0, 0, 0, 0.5],  # prompt 1, response 0 → score=0.5
    [0, 0, 0, 1.0],  # prompt 1, response 1 → score=1.0
])

scores = torch.tensor([1.0, 0.0, 0.5, 1.0])
```

### 2.3 第 2 步：按组分组

```python
# 代码位置: verl/trainer/ppo/core_algos.py:305-312
id2score = defaultdict(list)
id2mean = {}
id2std = {}

with torch.no_grad():
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
```

**作用：**
将属于同一个 prompt 的多个响应分到同一组。

**示例：**
```python
index = np.array([0, 0, 1, 1])  # 前 2 个属于 prompt 0，后 2 个属于 prompt 1

# 分组后:
id2score = {
    0: [1.0, 0.0],  # prompt 0 的 2 个响应
    1: [0.5, 1.0],  # prompt 1 的 2 个响应
}
```

### 2.4 第 3 步：计算组内统计量

```python
# 代码位置: verl/trainer/ppo/core_algos.py:313-322
for idx in id2score:
    if len(id2score[idx]) == 1:
        # 只有 1 个样本：均值=0，标准差=1（保持原值）
        id2mean[idx] = torch.tensor(0.0)
        id2std[idx] = torch.tensor(1.0)
    elif len(id2score[idx]) > 1:
        # 多个样本：计算真实的均值和标准差
        scores_tensor = torch.stack(id2score[idx])
        id2mean[idx] = torch.mean(scores_tensor)
        id2std[idx] = torch.std(scores_tensor)
```

**作用：**
计算每个组的均值和标准差，用作 baseline。

**示例：**
```python
# 继续上面的例子
id2mean = {
    0: (1.0 + 0.0) / 2 = 0.5,
    1: (0.5 + 1.0) / 2 = 0.75,
}

id2std = {
    0: std([1.0, 0.0]) = 0.5,
    1: std([0.5, 1.0]) = 0.25,
}
```

### 2.5 第 4 步：归一化优势值

```python
# 代码位置: verl/trainer/ppo/core_algos.py:323-328
for i in range(bsz):
    if norm_adv_by_std_in_grpo:
        # 原始 GRPO：除以标准差
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    else:
        # DrGRPO：不除以标准差
        scores[i] = scores[i] - id2mean[index[i]]
```

**作用：**
将每个响应的奖励归一化为相对于组内均值的优势值。

**示例（norm_adv_by_std_in_grpo=True）：**
```python
# prompt 0, response 0
advantage[0] = (1.0 - 0.5) / (0.5 + 1e-6) ≈ 1.0

# prompt 0, response 1
advantage[1] = (0.0 - 0.5) / (0.5 + 1e-6) ≈ -1.0

# prompt 1, response 0
advantage[2] = (0.5 - 0.75) / (0.25 + 1e-6) ≈ -1.0

# prompt 1, response 1
advantage[3] = (1.0 - 0.75) / (0.25 + 1e-6) ≈ 1.0
```

**解释：**
- 高于组均值的响应 → 正优势值 → 增强概率
- 低于组均值的响应 → 负优势值 → 降低概率

### 2.6 第 5 步：广播到 token 维度

```python
# 代码位置: verl/trainer/ppo/core_algos.py:328
scores = scores.unsqueeze(-1) * response_mask
```

**作用：**
将标量优势值扩展到每个 token，并只在响应部分生效。

**示例：**
```python
# 假设 response_mask:
response_mask = torch.tensor([
    [0, 0, 1, 1],  # 前 2 个是 prompt，后 2 个是 response
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
])

# 广播后:
advantages = torch.tensor([
    [0,  0,  1.0,  1.0],   # prompt 0, response 0
    [0,  0, -1.0, -1.0],   # prompt 0, response 1
    [0,  0, -1.0, -1.0],   # prompt 1, response 0
    [0,  0,  1.0,  1.0],   # prompt 1, response 1
])
```

### 2.7 完整代码流程图

```
输入:
  token_level_rewards: (4, 4) = [[0,0,0,1], [0,0,0,0], [0,0,0,0.5], [0,0,0,1]]
  response_mask: (4, 4)
  index: [0, 0, 1, 1]

第 1 步: 求和
  scores: (4,) = [1.0, 0.0, 0.5, 1.0]

第 2 步: 分组
  id2score[0] = [1.0, 0.0]
  id2score[1] = [0.5, 1.0]

第 3 步: 统计
  id2mean[0] = 0.5,  id2std[0] = 0.5
  id2mean[1] = 0.75, id2std[1] = 0.25

第 4 步: 归一化
  advantages: (4,) = [1.0, -1.0, -1.0, 1.0]

第 5 步: 广播
  advantages: (4, 4) = [[0,0,1,1], [0,0,-1,-1], [0,0,-1,-1], [0,0,1,1]]

输出:
  advantages: (4, 4)
  returns: (4, 4)  # GRPO 中 returns == advantages
```

---

## 3. 完整训练流程

### 3.1 从数据到优势值

```python
# 1. 加载数据（在 RayPPOTrainer.fit 中）
batch = {
    'prompts': [...],  # 256 个 prompts
    'reward_model': [...],
}

# 2. Rollout 生成（在 _train_step 第 1 阶段）
rollout_output = self.actor_rollout_wg.generate_sequences(batch)
# rollout_output.batch 现在有:
#   'input_ids': (1024, seq_len)  # 256 * 4 = 1024 个响应
#   'responses': (1024, response_len)
#   'response_mask': (1024, response_len)

# 3. 计算 Reward（在 _train_step 第 2 阶段）
rollout_output = self._compute_reward(rollout_output)
# rollout_output.batch 现在有:
#   'token_level_rewards': (1024, response_len)
#   'rewards': (1024,)  # 总奖励

# 4. 计算优势值（在 _train_step 第 5 阶段）
index = np.repeat(np.arange(256), 4)  # [0,0,0,0, 1,1,1,1, ..., 255,255,255,255]

advantages, returns = compute_grpo_outcome_advantage(
    token_level_rewards=rollout_output.batch['token_level_rewards'],
    response_mask=rollout_output.batch['response_mask'],
    index=index,
    norm_adv_by_std_in_grpo=True,
)

rollout_output.batch['advantages'] = advantages
```

### 3.2 GSM8K 训练示例追踪

**Prompt:**
```
"Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning..."
```

**生成 4 个响应：**
```python
responses = [
    "Let's solve step by step... #### 12",    # 正确答案
    "First, calculate... #### 15",           # 错误答案
    "We need to find... #### 12",            # 正确答案
    "The answer is... #### 10",              # 错误答案
]
```

**计算 Reward（使用 GSM8K reward）：**
```python
# ground_truth = "12"
rewards = [1.0, 0.0, 1.0, 0.0]  # 只有第 0 和第 2 个正确
```

**计算 GRPO 优势值：**
```python
# 组内统计
mean = (1.0 + 0.0 + 1.0 + 0.0) / 4 = 0.5
std = std([1.0, 0.0, 1.0, 0.0]) = 0.5

# 归一化优势
advantages = [
    (1.0 - 0.5) / 0.5 = 1.0,   # 正确响应 → 正优势
    (0.0 - 0.5) / 0.5 = -1.0,  # 错误响应 → 负优势
    (1.0 - 0.5) / 0.5 = 1.0,   # 正确响应 → 正优势
    (0.0 - 0.5) / 0.5 = -1.0,  # 错误响应 → 负优势
]
```

**策略更新：**
- 增强正确响应的生成概率
- 降低错误响应的生成概率

---

## 4. 配置参数详解

### 4.1 核心配置

```yaml
# 算法选择
algorithm:
  adv_estimator: grpo  # 使用 GRPO 优势估计器

# Rollout 配置
actor_rollout_ref:
  rollout:
    n: 4  # 每个 prompt 生成 4 个响应（必须 >= 2）

  actor:
    # KL 控制（GRPO 推荐使用 KL loss）
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: "k1"

    # PPO 更新
    ppo_epochs: 2
    ppo_mini_batch_size: 64
    clip_ratio: 0.2

    # Loss 聚合
    loss_agg_mode: "token-mean"  # "token-mean" | "seq-mean-token-mean"

# 数据配置
data:
  train_batch_size: 256  # 256 个 prompts → 256*4=1024 个响应
```

### 4.2 参数详解

#### `rollout.n`（重要！）

**作用：**每个 prompt 采样多少个响应

**推荐值：**
- `n=4`: 平衡计算效率和组内方差估计（默认）
- `n=2`: 最小值，方差估计不够准确
- `n=8`: 更准确，但计算开销大

**影响：**
```
总响应数 = train_batch_size * n
显存占用 ∝ n
训练时间 ∝ n（Rollout 阶段）
```

#### `norm_adv_by_std_in_grpo`

**作用：**是否除以标准差

**True（默认）：** 原始 GRPO
```python
advantage = (reward - mean) / (std + epsilon)
```

**False：** DrGRPO 变体
```python
advantage = reward - mean
```

#### `use_kl_loss`

**GRPO 推荐 `true`**，直接在 loss 中加 KL：
```python
total_loss = policy_loss + kl_loss_coef * kl_divergence
```

**vs PPO 的 KL reward penalty：**
```python
reward = original_reward - kl_coef * kl_divergence
```

#### `loss_agg_mode`

**token-mean（默认）：**
```python
loss = mean(losses * response_mask)
```

**seq-mean-token-mean：**
```python
loss = mean([mean(losses[i]) for i in range(bs)])
```

**seq-mean-token-sum-norm（DrGRPO）：**
```python
loss = mean([sum(losses[i]) / norm_factor for i in range(bs)])
```

### 4.3 完整训练命令

```bash
python3 -m verl.trainer.main_ppo \
    # 数据
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    \
    # 模型
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    \
    # Rollout（关键！）
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    \
    # 算法（GRPO 核心）
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    \
    # Actor 训练
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    \
    # 训练
    trainer.total_epochs=3 \
    trainer.logger=tensorboard \
    trainer.n_gpus_per_node=8
```

---

## 5. DrGRPO 变体

### 5.1 DrGRPO 是什么？

**论文：** [Understanding R1-Zero-Like Training](https://arxiv.org/pdf/2503.20783)

**核心发现：**
GRPO 原始实现有"长度偏差"：
- 错误答案往往更长（模型"胡编乱造"）
- 除以标准差会放大这种偏差

**DrGRPO 改进：**
1. 不除以标准差（`norm_adv_by_std_in_grpo=false`）
2. 使用全局归一化（`loss_agg_mode="seq-mean-token-sum-norm"`）

### 5.2 DrGRPO 配置

```yaml
actor_rollout_ref:
  actor:
    # 核心变化
    loss_agg_mode: "seq-mean-token-sum-norm"  # 全局归一化
    loss_scale_factor: 512  # 可选：固定归一化因子
    use_kl_loss: false  # DrGRPO 不用 KL loss

algorithm:
  norm_adv_by_std_in_grpo: false  # 不除以标准差
```

### 5.3 GRPO vs DrGRPO

| 特性 | GRPO | DrGRPO |
|------|------|---------|
| **标准差归一化** | ✅ 使用 | ❌ 不使用 |
| **优势公式** | `(r - μ) / σ` | `r - μ` |
| **Loss 聚合** | token-mean | seq-mean-token-sum-norm |
| **KL 控制** | KL loss | KL reward penalty |
| **长度偏差** | 可能存在 | 减轻 |
| **适用场景** | 一般任务 | 长 CoT 任务 |

---

## 6. 调试技巧

### 6.1 添加优势计算日志

```python
# 在 verl/trainer/ppo/core_algos.py:303 之后添加

scores = token_level_rewards.sum(dim=-1)

# 添加调试输出
print(f"\n[GRPO Debug] Batch info:")
print(f"  Batch size: {scores.shape[0]}")
print(f"  Unique prompts: {len(np.unique(index))}")
print(f"  Samples per prompt: {len(index) // len(np.unique(index))}")
print(f"  Scores: mean={scores.mean():.4f}, std={scores.std():.4f}")
print(f"  Scores range: [{scores.min():.4f}, {scores.max():.4f}]")
```

### 6.2 检查分组正确性

```python
# 在 verl/trainer/ppo/core_algos.py:313 之后添加

for idx in id2score:
    scores_list = [s.item() for s in id2score[idx]]
    print(f"  Group {idx}: scores={scores_list}, mean={id2mean[idx]:.4f}, std={id2std[idx]:.4f}")
```

### 6.3 查看优势分布

```python
# 在 verl/trainer/ppo/core_algos.py:328 之后添加

print(f"\n[GRPO Debug] Advantages:")
print(f"  Mean: {scores.mean():.4f}")
print(f"  Std: {scores.std():.4f}")
print(f"  Min: {scores.min():.4f}, Max: {scores.max():.4f}")
print(f"  Positive ratio: {(scores > 0).float().mean():.2%}")
```

### 6.4 TensorBoard 监控

关键指标：
```python
# 在 RayPPOTrainer 中记录
metrics = {
    'grpo/mean_reward': rewards.mean(),
    'grpo/std_reward': rewards.std(),
    'grpo/mean_advantage': advantages.mean(),
    'grpo/positive_ratio': (advantages > 0).float().mean(),
    'grpo/group_size': rollout_n,
}
```

---

## 7. 常见问题

### Q1: 为什么 GRPO 需要 `rollout.n >= 2`？

**原因：**
GRPO 需要多个样本来估计组内方差。

**如果 n=1：**
```python
id2mean[idx] = torch.tensor(0.0)
id2std[idx] = torch.tensor(1.0)
# 优势值 = reward / 1.0 = reward（没有归一化效果）
```

**推荐 n >= 4：**
- n=2: 方差估计不稳定
- n=4: 平衡点
- n=8: 更准确，但慢 2 倍

### Q2: GRPO 训练不收敛怎么办？

**可能原因 1：** `rollout.n` 太小
```yaml
# 增大采样数
actor_rollout_ref.rollout.n: 8
```

**可能原因 2：** 学习率太大
```yaml
actor_rollout_ref.actor.optim.lr: 5e-7  # 从 1e-6 降低
```

**可能原因 3：** Clipping 太宽松
```yaml
actor_rollout_ref.actor.clip_ratio: 0.1  # 从 0.2 降低
```

### Q3: 显存不够怎么办？

**方法 1：** 减小 `rollout.n`
```yaml
actor_rollout_ref.rollout.n: 2  # 从 4 降到 2
```

**方法 2：** 减小 batch size
```yaml
data.train_batch_size: 128  # 从 256 降低
```

**方法 3：** 使用 Gradient Checkpointing
```yaml
actor_rollout_ref.model.enable_gradient_checkpointing: true
```

### Q4: GRPO vs PPO 哪个更好？

**GRPO 更适合：**
- ✅ 数学推理（GSM8K, MATH）
- ✅ 代码生成（HumanEval）
- ✅ 快速实验
- ✅ GPU 资源有限

**PPO 更适合：**
- ✅ 长文本生成
- ✅ 对话质量优化
- ✅ 需要细粒度价值估计
- ✅ 追求训练稳定性

---

## 📚 参考资源

### 论文
- [DeepSeekMath (GRPO 原始论文)](https://arxiv.org/pdf/2402.03300)
- [DrGRPO 论文](https://arxiv.org/pdf/2503.20783)

### 代码位置
- GRPO 实现: `verl/trainer/ppo/core_algos.py:266-330`
- 向量化 GRPO: `verl/trainer/ppo/core_algos.py:333-357`
- Pass@k GRPO: `verl/trainer/ppo/core_algos.py:360-430`

### 官方文档
- [GRPO 文档](https://verl.readthedocs.io/en/latest/algo/grpo.html)
- [Baseline Performance](https://verl.readthedocs.io/en/latest/algo/baseline.html)

### 示例脚本
- `examples/grpo_trainer/run_qwen3-8b.sh`
- `examples/grpo_trainer/run_gemma-2-9b.sh`

---

*最后更新: 2026-01-26*
