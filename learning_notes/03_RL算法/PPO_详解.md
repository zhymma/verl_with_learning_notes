# PPO 算法详解

> Proximal Policy Optimization - Actor-Critic 强化学习算法

---

## 📖 目录

1. [PPO 核心思想](#1-ppo-核心思想)
2. [GAE 优势估计源码解析](#2-gae-优势估计源码解析)
3. [PPO Clipped Objective 源码解析](#3-ppo-clipped-objective-源码解析)
   - 3.6 [深度解析：old_log_prob vs new_log_prob 与 Importance Sampling](#36-深度解析old_log_prob-vs-new_log_prob-与-importance-sampling-) ⭐
4. [完整训练流程](#4-完整训练流程)
5. [配置参数详解](#5-配置参数详解)
6. [KL 散度控制](#6-kl-散度控制)
   - 6.4 [深度解析：ref_log_probs 计算与 KL 双重惩罚机制](#64-深度解析ref_log_probs-计算与-kl-双重惩罚机制-) ⭐
7. [调试技巧](#7-调试技巧)
8. [常见问题](#8-常见问题)

---

## 1. PPO 核心思想

### 1.1 什么是 PPO？

**PPO (Proximal Policy Optimization)** 是 OpenAI 在 2017 年提出的策略梯度算法，平衡了简单性、稳定性和性能。

**核心特点：**
- ✅ **Actor-Critic 架构**：同时训练策略网络和价值网络
- ✅ **Clipped Surrogate Objective**：限制策略更新幅度
- ✅ **GAE 优势估计**：平衡 bias 和 variance
- ✅ **训练稳定性高**：避免灾难性的策略崩溃

### 1.2 PPO vs GRPO

| 特性 | PPO | GRPO |
|------|-----|------|
| **Critic 模型** | ✅ 需要（价值函数） | ❌ 不需要 |
| **Baseline** | Critic 的 V(s) | 组内样本均值 |
| **优势估计** | GAE（时序差分） | 相对于组均值 |
| **GPU 显存** | 更多（Actor + Critic） | 更少（只有 Actor） |
| **训练稳定性** | 更稳定 | 依赖 rollout.n |
| **适用场景** | 过程导向任务、长序列 | 结果导向任务 |

### 1.3 PPO 三大组件

```
1. Critic 模型
   └─ 估计状态价值 V(s)
   └─ 提供 baseline 减少方差

2. GAE 优势估计
   └─ 计算 A(s,a) = Q(s,a) - V(s)
   └─ 使用 λ 平衡 bias 和 variance

3. Clipped Objective
   └─ 限制策略比率在 [1-ε, 1+ε]
   └─ 防止过大的策略更新
```

**关键配置：**
- `algorithm.adv_estimator=gae`
- `critic.model.path="Qwen/Qwen2.5-7B-Instruct"`
- `algorithm.gamma=0.99`（折扣因子）
- `algorithm.lam=0.95`（GAE lambda）

---

## 2. GAE 优势估计源码解析

### 2.1 函数签名

```python
# 位置: verl/trainer/ppo/core_algos.py:214-262

@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    values: torch.Tensor,               # (bs, response_length) - Critic 输出
    response_mask: torch.Tensor,        # (bs, response_length)
    gamma: torch.Tensor,                # 折扣因子，如 0.99
    lam: torch.Tensor,                  # GAE lambda，如 0.95
):
```

**参数说明：**
- `token_level_rewards`: 每个 token 的奖励
- `values`: Critic 模型预测的价值函数 V(s)
- `gamma`: 折扣因子（未来奖励的权重）
- `lam`: GAE 的 λ 参数（bias-variance tradeoff）

### 2.2 GAE 核心公式

**TD-error（时序差分误差）：**
```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
```

**GAE 优势值：**
```
A_t = δ_t + γλ * A_{t+1}
    = δ_t + γλ * δ_{t+1} + (γλ)² * δ_{t+2} + ...
```

**参数解释：**
- `γ=1, λ=1`: Monte Carlo（无偏，高方差）
- `γ>0, λ=0`: 1-step TD（低方差，有偏）
- `γ=0.99, λ=0.95`: 折中选择（默认）

### 2.3 第 1 步：初始化

```python
# 代码位置: verl/trainer/ppo/core_algos.py:243-247
with torch.no_grad():
    nextvalues = 0           # V(s_{t+1})
    lastgaelam = 0           # A_{t+1}
    advantages_reversed = [] # 存储倒序的优势值
    gen_len = token_level_rewards.shape[-1]
```

**作用：**
从序列末尾开始，反向计算优势值。

### 2.4 第 2 步：逆序循环计算

```python
# 代码位置: verl/trainer/ppo/core_algos.py:249-257
for t in reversed(range(gen_len)):
    # TD-error: δ_t = r_t + γ * V_{t+1} - V_t
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]

    # GAE: A_t = δ_t + γλ * A_{t+1}
    lastgaelam_ = delta + gamma * lam * lastgaelam

    # 只在响应 token 上更新（跳过 prompt）
    nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
    lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

    advantages_reversed.append(lastgaelam)
```

**关键点：**
- `delta`: TD-error，衡量 Critic 的预测误差
- `lastgaelam_`: 新的优势值
- `response_mask`: 确保只在响应部分计算优势

### 2.5 完整计算示例

**假设：**
```python
# 序列长度 T=4（2 个 prompt token + 2 个 response token）
token_level_rewards = [0, 0, 0, 1.0]  # 只有最后一个 token 有奖励
values = [0.2, 0.3, 0.4, 0.5]        # Critic 预测
response_mask = [0, 0, 1, 1]         # 后 2 个是响应
gamma = 0.99
lam = 0.95
```

**逆序计算：**

**t=3（最后一个 token）：**
```python
delta_3 = 1.0 + 0.99 * 0 - 0.5 = 0.5
A_3 = 0.5 + 0.99 * 0.95 * 0 = 0.5
```

**t=2：**
```python
delta_2 = 0 + 0.99 * 0.5 - 0.4 = 0.095
A_2 = 0.095 + 0.99 * 0.95 * 0.5 = 0.565
```

**t=1（prompt token，跳过）：**
```python
# 由于 response_mask[1]=0，A_1 不更新
A_1 = 0
```

**t=0（prompt token，跳过）：**
```python
A_0 = 0
```

**最终优势值：**
```python
advantages = [0, 0, 0.565, 0.5]
```

### 2.6 第 3 步：归一化和返回

```python
# 代码位置: verl/trainer/ppo/core_algos.py:258-262
advantages = torch.stack(advantages_reversed[::-1], dim=1)

# 计算 returns（用于训练 Critic）
returns = advantages + values

# 白化（Whitening）优势值
advantages = verl_F.masked_whiten(advantages, response_mask)

return advantages, returns
```

**作用：**
- `returns`: 用于 Critic 的目标值（MSE loss）
- `advantages`: 归一化后用于 Actor 更新

**masked_whiten 实现：**
```python
def masked_whiten(values, mask):
    mean = masked_mean(values, mask)
    std = masked_std(values, mask)
    return (values - mean) / (std + 1e-8)
```

---

## 3. PPO Clipped Objective 源码解析

### 3.1 函数签名

```python
# 位置: verl/trainer/ppo/core_algos.py:1095-1156

def compute_policy_loss_clip(
    old_log_prob: torch.Tensor,    # 旧策略的 log prob
    log_prob: torch.Tensor,        # 新策略的 log prob
    advantages: torch.Tensor,      # GAE 计算的优势值
    response_mask: torch.Tensor,   # 响应 mask
    cliprange: float,              # ε（clip ratio），如 0.2
    clip_ratio_c: float = 3.0,     # Dual-clip 的下界
    loss_agg_mode: str = "token-mean",
):
```

### 3.2 第 1 步：计算概率比

```python
# 代码位置: verl/trainer/ppo/core_algos.py:1128-1132
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
```

**作用：**
计算新旧策略的概率比 `ratio = π_new / π_old`

**示例：**
```python
old_log_prob = -2.0  # log(π_old(a|s)) = -2.0 → π_old = 0.135
log_prob = -1.5      # log(π_new(a|s)) = -1.5 → π_new = 0.223
ratio = exp(-1.5 - (-2.0)) = exp(0.5) = 1.65
```

**解释：**
- `ratio > 1`: 新策略增加了该动作的概率
- `ratio < 1`: 新策略降低了该动作的概率

### 3.3 第 2 步：计算 Clipped Loss

**标准 PPO Clip：**
```python
# 代码位置: verl/trainer/ppo/core_algos.py:1134-1145
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
```

**公式：**
```
L^CLIP = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

**图示（cliprange=0.2）：**
```
A > 0（好的动作）:
  如果 ratio > 1.2，clip 到 1.2（限制增强幅度）

A < 0（坏的动作）:
  如果 ratio < 0.8，clip 到 0.8（限制惩罚幅度）
```

### 3.4 第 3 步：Dual-clip PPO（可选）

```python
# 代码位置: verl/trainer/ppo/core_algos.py:1147-1153
pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_clipfrac_lower = verl_F.masked_mean(
    torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
)
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

**作用：**
当 `A < 0` 时，进一步限制 ratio 的下界为 `-clip_ratio_c`

**Dual-clip 公式：**
```
当 A < 0 时:
  L = -max(min(ratio * A, clip(ratio, 1-ε, 1+ε) * A), c * A)
```

### 3.5 完整示例

**输入：**
```python
old_log_prob = -2.0
log_prob = -1.5
advantage = 0.5  # 好的动作
cliprange = 0.2
```

**计算：**
```python
ratio = exp(-1.5 - (-2.0)) = 1.65

# Loss 1: 不 clip
loss1 = -0.5 * 1.65 = -0.825

# Loss 2: clip ratio 到 [0.8, 1.2]
clipped_ratio = min(max(1.65, 0.8), 1.2) = 1.2
loss2 = -0.5 * 1.2 = -0.6

# 取 max（loss 越大，梯度越小）
final_loss = max(-0.825, -0.6) = -0.6
```

**解释：**
- 由于 ratio=1.65 > 1.2，被 clip 到 1.2
- 限制了策略更新的幅度，防止过度优化

### 3.6 深度解析：old_log_prob vs new_log_prob 与 Importance Sampling ⭐

#### 3.6.1 核心概念：两个不同的 log_prob

PPO 的训练中涉及 **两个关键的 log 概率**，它们在不同时机计算，服务于不同的目的：

```
时间线：
┌────────────┬──────────────────┬────────────────────────┐
│  Rollout   │ Compute old_log  │ Mini-batch Training    │
│            │                  │ (重复多次)              │
├────────────┼──────────────────┼────────────────────────┤
│ π_rollout  │  π_old (冻结)    │  π_θ (不断更新)        │
│ (vLLM)     │  (FSDP)          │  (FSDP)                │
└────────────┴──────────────────┴────────────────────────┘
                                    ↑
                                new_log_prob
```

#### 3.6.2 old_log_prob（π_old）- 近端锚点

**位置**：`verl/trainer/ppo/ray_trainer.py:1258-1281`

```python
def _compute_old_log_prob(self, batch: DataProto):
    """
    计算 old_log_prob：训练开始前的策略快照
    这是 PPO 的"近端锚点"（proximal anchor）
    """

    # 1. 转换数据格式
    batch_td = batch.to_tensordict()
    batch_td = left_right_2_no_padding(batch_td)

    # 2. 设置元数据
    tu.assign_non_tensor(
        batch_td,
        calculate_entropy=True,
        compute_loss=False
    )

    # 3. 使用当前 Actor 重新计算 log_prob
    output = self.actor_rollout_wg.compute_log_prob(batch_td)

    # 4. 提取结果
    old_log_probs = tu.get(output, "log_probs")
    entropy = tu.get(output, "entropy")

    return old_log_probs, entropy
```

**关键特点**：
- **计算时机**：每个 batch 开始时计算 **一次**
- **策略版本**：当前 Actor 的权重（训练前的快照）
- **作用**：在整个 mini-batch 训练期间 **保持不变**
- **目的**：作为 PPO 的"近端锚点"，限制策略更新幅度

#### 3.6.3 new_log_prob（π_θ）- 优化目标

**位置**：`verl/workers/utils/losses.py:97-174`

```python
def ppo_loss(config, model_output, data):
    """PPO Loss 计算"""

    # 1. 从当前模型的前向传播获取 log_prob
    log_prob = model_output["log_probs"]  # ← new_log_prob (π_θ)

    # 2. 从数据中获取 old_log_prob
    old_log_prob = data["old_log_probs"]  # ← 冻结的参考

    # 3. 计算 importance sampling ratio
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)  # π_θ / π_old

    # 4. PPO Clipping
    advantages = data["advantages"]
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss
```

**关键特点**：
- **计算时机**：每个 mini-batch 的每次前向传播
- **策略版本**：当前正在优化的 Actor 权重（不断更新）
- **作用**：在 mini-batch 训练中 **不断变化**
- **目的**：这是我们要优化的目标策略

#### 3.6.4 Importance Sampling Ratio 计算

**位置**：`verl/trainer/ppo/core_algos.py:1210-1226`

```python
def vanilla_ppo_policy_loss(old_log_prob, log_prob, advantages, ...):
    """标准 PPO 的 importance sampling ratio 计算"""

    # 1. 计算 log ratio（数值稳定）
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20, max=20)

    # 2. 计算 importance sampling ratio
    ratio = torch.exp(negative_approx_kl)  # r = π_θ / π_old

    # 3. PPO Clipping
    cliprange_low = config.clip_ratio       # 0.2
    cliprange_high = config.clip_ratio_high  # 0.2

    pg_losses1 = -advantages * ratio  # 未裁剪
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1-cliprange_low, 1+cliprange_high
    )  # 裁剪到 [0.8, 1.2]

    # 4. 取较大的 loss（更保守）
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    return policy_loss
```

**公式详解**：

**Importance Sampling Ratio**：
$$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)} = \exp(\log \pi_\theta - \log \pi_{\text{old}})$$

**PPO Clipped Objective**：
$$L^{\text{CLIP}}(\theta) = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$$

其中：
- $r_t$ = ratio（重要性采样比率）
- $A_t$ = advantages（优势函数）
- $\epsilon$ = clip_ratio（默认 0.2）

#### 3.6.5 为什么需要两个 log_prob？

##### **原因 1：信任域机制（Trust Region）**

```python
# 如果只有 new_log_prob，没有 old_log_prob：
# ❌ 无法计算 ratio = π_θ / π_old
# ❌ 无法实施 clipping
# ❌ 策略可能剧烈变化，导致崩溃

# 有了 old_log_prob 和 new_log_prob：
ratio = exp(new_log_prob - old_log_prob)
ratio_clipped = clamp(ratio, 0.8, 1.2)  # 限制在 ±20%

# ✅ 策略变化被限制在 ±20%
# ✅ 训练稳定，避免灾难性遗忘
```

##### **原因 2：Mini-batch 训练的稳定性**

```python
# 一个 batch 会进行多次 mini-batch 更新
for epoch in range(ppo_epochs):  # 通常 1-4 次
    for mini_batch in dataloader:
        # new_log_prob 每次都改变
        # 但 old_log_prob 保持不变！

        ratio = π_θ / π_old  # 始终相对于同一个参考点

        # 这确保了：
        # 1. Advantages 不会变得"过时"
        # 2. 优化过程有明确的锚点
        # 3. 每次更新都是相对于同一基准
```

##### **原因 3：防止 Advantage 失效**

```python
# Advantages 基于 rollout 时的 rewards 计算
# 如果策略变化太大，advantages 就不再有效

# 示例：
old_policy: "The answer is 42"  → A = 0.5
# 如果允许策略剧烈变化：
new_policy: "blah blah blah"    → 完全不同的分布！

# PPO 通过 clipping 防止这种情况：
# ratio > 1.2 → 裁剪到 1.2
# ratio < 0.8 → 裁剪到 0.8
# 保证策略变化在合理范围内
```

#### 3.6.6 完整示例：追踪一个 Batch

```python
# 假设：batch_size=4, seq_len=10, ppo_epochs=2

# ==================== Rollout ====================
prompts = ["What is 2+2?", "What is 3+3?", ...]
responses = vllm_generate(prompts)

# ==================== Compute old_log_prob ====================
old_log_prob = actor_model.compute_log_prob(responses)
# shape: [4, 10]
print(f"old_log_prob[0, :5]: {old_log_prob[0, :5]}")
# 输出: [-2.3, -1.8, -0.9, -0.5, -0.2]

# ==================== Advantages ====================
advantages = compute_gae(rewards, values)

# ==================== Mini-batch Training ====================
# Epoch 1
for step in range(100):  # 模拟 100 步优化
    # 前向传播
    new_log_prob = actor_model(responses)  # ← 每次都计算

    # step 1:  [-2.3, -1.8, -0.9, -0.5, -0.2]  (初始接近)
    # step 50: [-2.2, -1.7, -0.8, -0.4, -0.1]  (开始变化)
    # step 100:[-2.1, -1.6, -0.7, -0.3, 0.0]   (继续变化)

    # 计算 ratio（始终相对于 old_log_prob）
    ratio = exp(new_log_prob - old_log_prob)

    # step 1:  [1.0, 1.0, 1.0, 1.0, 1.0]  (几乎没变)
    # step 50: [1.1, 1.1, 1.1, 1.1, 1.2]  (开始偏离)
    # step 100:[1.2, 1.2, 1.2, 1.2, 1.2]  (接近边界)

    # Clipping
    ratio_clipped = clamp(ratio, 0.8, 1.2)
    # step 100:[1.2, 1.2, 1.2, 1.2, 1.2]  (被裁剪)

    # Loss
    loss = -min(ratio * A, ratio_clipped * A).mean()
    loss.backward()
    optimizer.step()

# 关键：在整个训练过程中，old_log_prob 始终不变！
```

#### 3.6.7 三种策略模式

verl 支持三种策略配置：

**模式 1：Decoupled（3 策略）- 默认**
```
π_rollout (vLLM BF16)  → 生成响应
    ↓
π_old (FSDP FP32)      → 重新计算，作为锚点
    ↓
π_θ (FSDP FP32)        → 优化目标

ratio = π_θ / π_old
```
✅ 精确的 ratio 计算
✅ 训练最稳定

**模式 2：Bypass（2 策略）**
```python
# ray_trainer.py:1527-1535
if self.bypass_mode:
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
```
```
π_rollout (vLLM BF16)  → 生成 + 作为 old_log_prob
    ↓
π_θ (FSDP FP32)        → 优化目标

ratio = π_θ / π_rollout
```
✅ 节省计算（不需要重算 old_log_prob）
⚠️ 可能有分布差异（BF16 vs FP32）

**模式 3：Rollout Correction（修正的 2 策略）**
```python
# 计算修正权重
log_ratio = old_log_prob - rollout_log_prob
rollout_is_weights = torch.exp(log_ratio)

# 在 loss 中应用
policy_loss *= rollout_is_weights
```
✅ 修正 vLLM 和 FSDP 的分布差异

#### 3.6.8 对比表

| 维度 | old_log_prob (π_old) | new_log_prob (π_θ) |
|------|---------------------|-------------------|
| **计算时机** | 每个 batch 一次 | 每次前向传播 |
| **策略版本** | 训练前的 Actor 快照 | 当前正在优化的 Actor |
| **在训练中** | 冻结不变 | 不断更新 |
| **用途** | 计算 importance ratio | 优化目标 |
| **代码位置** | `ray_trainer.py:1258` | `losses.py:97` |
| **依赖** | 当前 Actor weights | 当前 Actor weights (evolving) |

**核心要点**：
- old_log_prob 是训练开始时的快照，训练期间保持不变
- new_log_prob 是每次前向传播的输出，不断更新
- ratio = π_θ / π_old 限制策略变化在 ±20%（clip_ratio=0.2）
- 这就是 PPO 稳定训练的核心机制 🎯

---

## 4. 完整训练流程

### 4.1 PPO 训练的 7 个阶段

```python
# 在 RayPPOTrainer._train_step 中

# 阶段 1: Rollout - Actor 生成响应
rollout_output = self.actor_rollout_wg.generate_sequences(batch)

# 阶段 2: Reward - 计算奖励
rollout_output = self._compute_reward(rollout_output)

# 阶段 3: Ref - 计算参考模型 log prob（用于 KL penalty）
rollout_output = self.actor_rollout_wg.compute_ref_log_prob(rollout_output)

# 阶段 4: Value - Critic 预测价值函数（PPO 独有）
rollout_output = self.critic_wg.compute_values(rollout_output)

# 阶段 5: Advantage - 计算 GAE 优势值（PPO 独有）
advantages, returns = compute_gae_advantage_return(
    token_level_rewards=rollout_output.batch['token_level_rewards'],
    values=rollout_output.batch['values'],  # Critic 输出
    response_mask=rollout_output.batch['response_mask'],
    gamma=self.config.algorithm.gamma,
    lam=self.config.algorithm.lam,
)

# 阶段 6: Actor Update - 使用 PPO Clip 更新策略
actor_metrics = self.actor_rollout_wg.update_actor(rollout_output)

# 阶段 7: Critic Update - 使用 MSE loss 更新价值函数（PPO 独有）
critic_metrics = self.critic_wg.update_critic(rollout_output)
```

### 4.2 GSM8K 训练示例追踪

**Prompt:**
```
"Janet's ducks lay 16 eggs per day..."
```

**生成 1 个响应（PPO 通常 n=1）：**
```python
response = "Let's solve step by step... #### 12"
```

**阶段 2: 计算 Reward**
```python
reward = 1.0  # 答案正确
token_level_rewards = [0, 0, ..., 0, 1.0]  # 只有最后一个 token
```

**阶段 4: Critic 预测**
```python
values = critic_model(input_ids)
# values = [0.1, 0.2, 0.3, ..., 0.8]
```

**阶段 5: GAE 优势值**
```python
# 从后向前计算
delta_T = 1.0 + 0 - 0.8 = 0.2
A_T = 0.2

delta_{T-1} = 0 + 0.99 * 0.8 - 0.7 = 0.092
A_{T-1} = 0.092 + 0.99 * 0.95 * 0.2 = 0.280

# ...
advantages = [0.45, 0.42, 0.38, ..., 0.28, 0.2]
```

**阶段 6: Actor 更新**
```python
# 计算 policy loss
loss = compute_policy_loss_clip(
    old_log_prob=old_log_probs,
    log_prob=new_log_probs,
    advantages=advantages,
    cliprange=0.2,
)

# 反向传播更新 Actor
loss.backward()
optimizer.step()
```

**阶段 7: Critic 更新**
```python
# MSE loss
returns = advantages + values
critic_loss = (critic_values - returns) ** 2

# 反向传播更新 Critic
critic_loss.backward()
critic_optimizer.step()
```

---

## 5. 配置参数详解

### 5.1 核心配置

```yaml
# 算法选择
algorithm:
  adv_estimator: gae  # 使用 GAE 优势估计
  gamma: 0.99         # 折扣因子
  lam: 0.95           # GAE lambda

# Critic 模型（PPO 必需！）
critic:
  model:
    path: "Qwen/Qwen2.5-7B-Instruct"
    enable_gradient_checkpointing: true

  # Critic 训练
  ppo_epochs: 2
  ppo_mini_batch_size: 64
  ppo_micro_batch_size: 2
  optim:
    lr: 5e-6

# Actor 配置
actor_rollout_ref:
  rollout:
    n: 1  # PPO 通常 n=1（有 Critic 作为 baseline）

  actor:
    ppo_epochs: 2
    ppo_mini_batch_size: 64
    clip_ratio: 0.2      # PPO clip range ε
    clip_ratio_c: 3.0    # Dual-clip 下界
```

### 5.2 参数详解

#### `gamma`（折扣因子）

**作用：**未来奖励的权重

**推荐值：**
- `gamma=0.99`: 默认值，重视长期奖励
- `gamma=0.95`: 更关注近期奖励
- `gamma=1.0`: 所有奖励等权（不推荐）

**影响：**
```
TD-error: δ_t = r_t + γ * V_{t+1} - V_t
γ 越大，越重视未来价值
```

#### `lam`（GAE lambda）

**作用：**平衡 bias 和 variance

**推荐值：**
- `lam=0.95`: 默认值（折中）
- `lam=1.0`: Monte Carlo（无偏，高方差）
- `lam=0.0`: 1-step TD（低方差，有偏）

**公式：**
```
A_t = δ_t + γλ * δ_{t+1} + (γλ)² * δ_{t+2} + ...
```

#### `clip_ratio`（PPO clip range）

**作用：**限制策略更新幅度

**推荐值：**
- `clip_ratio=0.2`: 默认值
- `clip_ratio=0.1`: 更保守（更稳定）
- `clip_ratio=0.3`: 更激进（可能不稳定）

**含义：**
```
ratio ∈ [1-ε, 1+ε] = [0.8, 1.2]
```

#### `ppo_epochs`

**作用：**每个 batch 重复训练多少次

**推荐值：**
- `ppo_epochs=2`: 平衡效率和稳定性
- `ppo_epochs=1`: 更快，可能欠拟合
- `ppo_epochs=4`: 更充分训练，可能过拟合

### 5.3 完整训练命令

```bash
python3 -m verl.trainer.main_ppo \
    # 数据
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    \
    # Actor 模型
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    \
    # Critic 模型（PPO 必需）
    critic.model.path=Qwen/Qwen2.5-7B-Instruct \
    critic.ppo_epochs=2 \
    critic.ppo_mini_batch_size=64 \
    critic.optim.lr=5e-6 \
    \
    # 算法（PPO 核心）
    algorithm.adv_estimator=gae \
    algorithm.gamma=0.99 \
    algorithm.lam=0.95 \
    \
    # 训练
    trainer.total_epochs=3 \
    trainer.n_gpus_per_node=8
```

---

## 6. KL 散度控制

### 6.1 两种 KL 控制方式

#### 方式 1: KL Reward Penalty（PPO 传统）

```yaml
algorithm:
  use_kl_in_reward: true
  kl_penalty: "kl"  # k1, abs, mse, low_var_kl
  kl_ctrl:
    type: "fixed"  # or "adaptive"
    kl_coef: 0.001
```

**实现：**
```python
reward_with_kl = reward - kl_coef * kl_divergence(new_policy, ref_policy)
```

#### 方式 2: KL Loss（GRPO 推荐，PPO 也可用）

```yaml
actor_rollout_ref:
  actor:
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: "k1"
```

**实现：**
```python
total_loss = policy_loss + kl_loss_coef * kl_divergence
```

### 6.2 KL 散度的 4 种计算方式

```python
# k1 (标准 KL)
kl = old_log_prob - log_prob

# abs
kl = abs(old_log_prob - log_prob)

# mse (k2)
kl = 0.5 * (old_log_prob - log_prob) ** 2

# low_var_kl (k3)
ratio = exp(log_prob - old_log_prob)
kl = (ratio - 1) - log(ratio)
```

**参考：** [KL Approximations Blog](http://joschu.net/blog/kl-approx.html)

### 6.3 Adaptive KL Controller

```yaml
algorithm:
  kl_ctrl:
    type: "adaptive"
    kl_coef: 0.001     # 初始系数
    target_kl: 0.01    # 目标 KL
    horizon: 10000     # 调整窗口
```

**自适应调整：**
```python
if current_kl > target_kl:
    kl_coef *= (1 + proportional_error)  # 增大惩罚
else:
    kl_coef *= (1 - proportional_error)  # 减小惩罚
```

### 6.4 深度解析：ref_log_probs 计算与 KL 双重惩罚机制 ⭐

#### 6.4.1 ref_log_probs 如何计算

**位置**：`verl/trainer/ppo/ray_trainer.py:1231-1256`

```python
def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
    """计算参考策略的 log 概率"""

    batch_td = batch.to_tensordict()
    batch_td = left_right_2_no_padding(batch_td)

    tu.assign_non_tensor(
        batch_td,
        calculate_entropy=False,
        compute_loss=False,
    )

    if self.ref_in_actor:
        # 方式 1：使用 LoRA 训练时
        # 禁用 LoRA adapter，回到 base model
        output = self.actor_rollout_wg.compute_log_prob(
            batch_td,
            no_lora_adapter=True  # 关键：不用 LoRA
        )
    else:
        # 方式 2：使用单独的参考策略 Worker
        output = self.ref_policy_wg.compute_ref_log_prob(batch_td)

    ref_log_prob = tu.get(output, "log_probs")
    return ref_log_prob
```

**两种实现方式对比**：

| 方式 | ref_in_actor=True | ref_in_actor=False |
|------|-------------------|-------------------|
| **模型结构** | Actor = Base + LoRA | Actor 独立 + RefPolicy 独立 |
| **计算方式** | 禁用 LoRA，用 Base | 用单独的 RefPolicy Worker |
| **显存占用** | 更少（共享 base） | 更多（两个完整模型） |
| **灵活性** | LoRA 训练专用 | 参考策略可以是任意模型 |
| **适用场景** | 微调、对齐 | 完全独立的参考策略 |

#### 6.4.2 KL 双重惩罚：use_kl_in_reward vs use_kl_loss

verl 实现了 **两种完全独立** 的 KL 应用方式，可以单独或组合使用：

##### **机制 A：KL in Reward（Reward 阶段）**

**位置**：`verl/trainer/ppo/ray_trainer.py:127-166`

```python
def apply_kl_penalty(data, kl_ctrl, kl_penalty="kl"):
    """在 reward 中减去 KL 散度"""

    # 1. 计算 KL 散度
    kld = core_algos.kl_penalty(
        logprob=data.batch["old_log_probs"],      # π_old
        ref_logprob=data.batch["ref_log_prob"],   # π_ref
        kl_penalty=kl_penalty
    )

    # 2. 从 reward 中减去 KL
    beta = kl_ctrl.value  # 自适应系数
    token_level_rewards = token_level_scores - beta * kld

    # 3. 更新自适应控制器
    current_kl = masked_mean(kld, mask=response_mask)
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)

    return token_level_rewards
```

**配置**：
```yaml
algorithm:
  use_kl_in_reward: true
  kl_penalty: "low_var_kl"  # k3 估计器
  kl_ctrl:
    type: "adaptive"
    kl_coef: 0.1
    target_kl: 0.1
```

##### **机制 B：KL in Loss（Policy 更新阶段）**

**位置**：`verl/workers/utils/losses.py:96-174`

```python
def ppo_loss(config, model_output, data):
    """PPO Loss 计算"""

    # 1. 标准 PPO Loss
    log_prob = model_output["log_probs"]
    old_log_prob = data["old_log_probs"]

    ratio = torch.exp(log_prob - old_log_prob)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    total_loss = policy_loss

    # 2. 如果启用 KL Loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]

        # 计算 KL 散度
        kld = kl_penalty(
            logprob=log_prob,              # π_θ
            ref_logprob=ref_log_prob,      # π_ref
            kl_penalty=config.kl_loss_type
        )

        kl_loss = agg_loss(kld, loss_mask=response_mask)
        total_loss += config.kl_loss_coef * kl_loss

    return total_loss
```

**配置**：
```yaml
actor_rollout_ref:
  actor:
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: "low_var_kl"
```

##### **两者对比表**

| 维度 | KL in Reward | KL in Loss |
|------|-------------|-----------|
| **时机** | Reward 计算后，Advantage 前 | Actor 更新时，Loss 计算中 |
| **影响** | Token-level rewards | Policy gradient |
| **公式** | `r' = r - β * KL(π_old ‖ π_ref)` | `L = L_ppo + λ * KL(π_θ ‖ π_ref)` |
| **策略** | π_old, π_ref | π_θ, π_ref |
| **系数** | β (自适应，AdaptiveKLController) | λ (固定) |
| **配置** | `algorithm.use_kl_in_reward` | `actor.use_kl_loss` |
| **默认** | False | False |
| **估计器** | `"kl"` (k1) | `"low_var_kl"` (k3) |

##### **完整训练流程（双 KL）**

```python
# Step 1: 生成响应
responses = actor_rollout_wg.generate_sequences(prompts)

# Step 2: 计算原始 Reward
raw_rewards = reward_model(responses)

# Step 3: 计算 ref_log_prob
ref_log_prob = _compute_ref_log_prob(batch)  # log π_ref

# Step 4: 计算 old_log_prob
old_log_prob = actor_rollout_wg.compute_log_prob(batch)  # log π_old

# ========== KL in Reward ==========
if use_kl_in_reward:
    kld = old_log_prob - ref_log_prob
    token_level_rewards = raw_rewards - beta * kld
    # beta 自适应调整
else:
    token_level_rewards = raw_rewards

# Step 5: 计算 Advantage (基于调整后的 rewards)
advantages = compute_gae(token_level_rewards, values)

# Step 6: Actor 更新
for epoch in range(ppo_epochs):
    new_log_prob = actor_model(responses)  # log π_θ

    # PPO Loss
    ratio = torch.exp(new_log_prob - old_log_prob)
    ppo_loss = -torch.min(
        ratio * advantages,
        torch.clamp(ratio, 0.8, 1.2) * advantages
    ).mean()

    total_loss = ppo_loss

    # ========== KL in Loss ==========
    if use_kl_loss:
        kld = new_log_prob - ref_log_prob
        kl_loss = kld.mean()
        total_loss += lambda_kl * kl_loss

    total_loss.backward()
```

##### **使用场景建议**

**场景 1：RLHF 对齐（只用 KL in Reward）**
```yaml
algorithm:
  use_kl_in_reward: true
  kl_ctrl:
    type: "adaptive"
    kl_coef: 0.1

actor:
  use_kl_loss: false
```
✅ 适合：对齐任务，防止偏离 base model
✅ 优点：自适应调整，训练稳定

**场景 2：强正则化（只用 KL in Loss）**
```yaml
algorithm:
  use_kl_in_reward: false

actor:
  use_kl_loss: true
  kl_loss_coef: 0.001
```
✅ 适合：需要直接约束 policy
✅ 优点：理论清晰，梯度直接

**场景 3：双重约束（两者都用）**
```yaml
algorithm:
  use_kl_in_reward: true
  kl_ctrl:
    type: "fixed"
    kl_coef: 0.05

actor:
  use_kl_loss: true
  kl_loss_coef: 0.001
```
✅ 适合：安全性要求极高的场景
⚠️ 注意：可能过度保守，policy 更新慢

---

## 7. 调试技巧

### 7.1 添加 GAE 计算日志

```python
# 在 verl/trainer/ppo/core_algos.py:250 添加

for t in reversed(range(gen_len)):
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]

    # 添加调试
    if t == gen_len - 1:  # 最后一个 token
        print(f"[GAE Debug] t={t}")
        print(f"  reward: {token_level_rewards[:, t][:3]}")
        print(f"  value: {values[:, t][:3]}")
        print(f"  delta: {delta[:3]}")
```

### 7.2 检查 Critic 预测质量

```python
# 在 RayPPOTrainer._train_step 阶段 5 后添加

print(f"\n[Critic Debug]")
print(f"  Values mean: {values.mean():.4f}, std: {values.std():.4f}")
print(f"  Rewards mean: {token_level_rewards.sum(-1).mean():.4f}")
print(f"  Critic MSE: {((values - returns) ** 2).mean():.4f}")
```

### 7.3 监控 Clipping 比例

```python
# 在 compute_policy_loss_clip 中已有
pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

# 在 TensorBoard 记录
metrics = {
    'ppo/clipfrac': pg_clipfrac,  # 被 clip 的比例
    'ppo/kl': ppo_kl,              # KL 散度
}
```

**理想值：**
- `clipfrac < 0.2`: 策略更新合理
- `clipfrac > 0.5`: 策略更新过激，考虑降低学习率

### 7.4 TensorBoard 监控

关键指标：
```python
metrics = {
    # 奖励
    'reward/mean': rewards.mean(),

    # Advantage
    'advantage/mean': advantages.mean(),
    'advantage/std': advantages.std(),

    # Critic
    'critic/value_mean': values.mean(),
    'critic/loss': critic_loss,

    # Actor
    'actor/loss': policy_loss,
    'actor/clipfrac': clipfrac,

    # KL
    'kl/mean': kl_divergence.mean(),
}
```

---

## 8. 常见问题

### Q1: PPO 需要多少 GPU 显存？

**计算公式：**
```
总显存 = Actor 显存 + Critic 显存 + Rollout 显存
```

**示例（Qwen2.5-7B）：**
- Actor: ~30GB（FSDP 训练）
- Critic: ~30GB（FSDP 训练）
- Rollout: ~20GB（vLLM 推理）
- **总计：~80GB**（需要 2 张 A100 80GB）

**优化方法：**
- 使用 Gradient Checkpointing
- 减小 micro batch size
- 使用 mixed precision (FP16/BF16)

### Q2: Critic 损失不下降怎么办？

**可能原因 1：** 学习率太小
```yaml
critic.optim.lr: 1e-5  # 从 5e-6 增大
```

**可能原因 2：** Reward 分布变化大
```yaml
# 增加 Critic 训练 epochs
critic.ppo_epochs: 4
```

**可能原因 3：** 值函数初始化不好
```yaml
# 使用预训练模型初始化
critic.model.path: "path/to/pretrained/critic"
```

### Q3: PPO vs GRPO 如何选择？

**选择 PPO：**
- ✅ 长序列生成（如长文本、对话）
- ✅ 需要细粒度价值估计
- ✅ 追求训练稳定性
- ✅ GPU 资源充足

**选择 GRPO：**
- ✅ 结果导向任务（数学、代码）
- ✅ 快速实验
- ✅ GPU 资源有限
- ✅ 训练速度优先

### Q4: GAE 的 lambda 怎么调？

**lambda 影响：**
- `λ=0`: 只用 1-step TD（低方差，高偏差）
  - 适合：Critic 很准确
- `λ=1`: 用完整 MC（高方差，无偏）
  - 适合：Critic 不准确
- `λ=0.95`: 折中（默认推荐）

**调参建议：**
1. 先用 λ=0.95
2. 如果训练不稳定 → 降低 λ（如 0.9）
3. 如果 Critic loss 很低 → 降低 λ（信任 Critic）
4. 如果 Critic loss 很高 → 增大 λ（不信任 Critic）

### Q5: Dual-clip PPO 什么时候用？

**标准场景：**
- 使用标准 PPO（`clip_ratio_c` 很大，实际不生效）

**Dual-clip 场景：**
- 负优势时，策略下降过快
- 设置 `clip_ratio_c=3.0`（默认）

**公式：**
```
当 A < 0 时，ratio 下界 = -clip_ratio_c
防止过度惩罚坏动作
```

---

## 📚 参考资源

### 论文
- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [GAE 论文](https://arxiv.org/abs/1506.02438)
- [Dual-clip PPO](https://arxiv.org/pdf/1912.09729)

### 代码位置
- GAE 实现: `verl/trainer/ppo/core_algos.py:214-262`
- PPO Clip Loss: `verl/trainer/ppo/core_algos.py:1095-1156`
- Critic 更新: `verl/workers/fsdp_workers.py` (CriticWorker)

### 官方文档
- [PPO 文档](https://verl.readthedocs.io/en/latest/algo/ppo.html)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

### 示例脚本
- `examples/ppo_trainer/run_gemma.sh`
- `examples/ppo_trainer/run_qwen2.5-0.5b.sh`

---

*最后更新: 2026-01-26*
