# Reward 系统详解

> 深入理解 verl 的 Reward 计算流程，学会设计和调试 Reward 函数

---

## 📋 本文内容

1. Reward 系统概览
2. RewardManager 工作流程
   - 2.3 Reward Token 放置机制与 Advantage 广播 ⭐
3. GSM8K Reward 详细解析
4. 自定义 Reward 函数
5. Reward 调试技巧
6. 常见问题解决

---

## 1. Reward 系统概览

### 1.1 Reward 在 RL 训练中的作用

```
训练流程：
1. Rollout 生成响应
   ↓
2. Reward 函数评分  ← 我们在这里！
   ↓
3. Advantage 计算
   ↓
4. Policy 更新
```

**Reward 的重要性：**
- ✅ **定义目标**：告诉模型什么是"好"的输出
- ✅ **引导学习**：高 reward → 强化，低 reward → 抑制
- ✅ **影响效果**：Reward 设计直接决定最终模型行为

### 1.2 Reward 类型

verl 支持三种 Reward 计算方式：

| 类型 | 说明 | 适用场景 | 示例 |
|------|------|---------|------|
| **Rule-based** | 基于规则的打分 | 有标准答案的任务 | GSM8K, 代码题 |
| **Model-based** | 使用 Reward Model | RLHF, 主观任务 | 对话质量 |
| **Sandbox** | 执行代码获取结果 | 代码生成 | APPS, HumanEval |

---

## 2. RewardManager 工作流程

### 2.1 RewardManager 架构

**核心文件：**
- `verl/trainer/ppo/reward.py` - 主入口
- `verl/workers/reward_manager/` - RewardManager 实现
- `verl/utils/reward_score/` - 内置 reward 函数

**类图：**
```
AbstractRewardManager (抽象基类)
    ↑
    ├── NaiveRewardManager        # 简单的 rule-based
    ├── RateLimitedRewardManager  # 支持 rate limit（API 调用）
    └── RewardLoopManager         # 异步 reward 计算
```

### 2.2 RewardManager 初始化

**代码位置：** `verl/trainer/ppo/reward.py: 第 99-175 行`

```python
# verl/trainer/ppo/reward.py

def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """加载 RewardManager

    主要步骤：
    1. 加载自定义 reward 函数（如果有）
    2. 选择 RewardManager 类型
    3. 实例化 RewardManager
    """

    # ========== 步骤 1: 获取自定义 reward 函数 ==========
    compute_score = get_custom_reward_fn(config)
    # 如果 config.custom_reward_function.path 存在，则加载

    # ========== 步骤 2: 如果没有自定义函数，使用默认 ==========
    if compute_score is None:
        # 检查是否使用 Sandbox（代码执行）
        sandbox_config = config.reward_model.get("sandbox_fusion")
        if sandbox_config and sandbox_config.get("url"):
            # 使用 Sandbox 执行代码
            compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                ...
            )
        else:
            # 使用默认的 rule-based reward
            compute_score = default_compute_score

    # ========== 步骤 3: 实例化 RewardManager ==========
    reward_manager = NaiveRewardManager(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
    )

    return reward_manager
```

### 2.3 Reward 计算流程

**代码位置：** `verl/workers/reward_manager/naive.py`

```python
# verl/workers/reward_manager/naive.py（简化版）

class NaiveRewardManager:
    def __init__(self, tokenizer, compute_score, ...):
        self.tokenizer = tokenizer
        self.compute_score = compute_score

    def __call__(self, data: DataProto) -> torch.Tensor:
        """计算 reward

        输入：
            data: DataProto，包含：
                - responses: 生成的 token IDs
                - data_source: 数据来源（用于路由）
                - ground_truth: 标准答案

        输出：
            reward_tensor: [batch_size, seq_len] 的 reward
        """

        batch_size = len(data)
        rewards = []

        # ========== 遍历每个样本 ==========
        for i in range(batch_size):
            # 步骤 1: Decode 响应
            response_ids = data.batch['responses'][i]
            response_text = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True
            )

            # 步骤 2: 获取元数据
            data_source = data.non_tensor_batch['data_source'][i]
            ground_truth = data.non_tensor_batch['ground_truth'][i]

            # 步骤 3: 调用 reward 函数
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_text,
                ground_truth=ground_truth,
            )

            # 步骤 4: 将 reward 放到最后一个 token
            seq_len = len(response_ids)
            reward_seq = torch.zeros(seq_len)
            reward_seq[-1] = score  # 只有最后一个 token 有 reward

            rewards.append(reward_seq)

        # ========== 返回 tensor ==========
        reward_tensor = torch.stack(rewards)  # [batch_size, seq_len]
        return reward_tensor
```

**关键点：Reward 通常只在最后一个 token！**
```
Response: "Let me think... 25 * 4 = 100"
Tokens:   [T1, T2, T3, ..., T_n]
Rewards:  [0,  0,  0,  ..., 1.0]  ← 只有最后一个有 reward
```

### 2.3 Reward Token 放置机制与 Advantage 广播 ⭐

> **核心问题**：为什么 Reward 只在最后一个 token？后面会广播到每个 token 吗？

#### 2.3.1 为什么只在最后一个 Token？

**设计理念：Outcome Supervision（结果监督）**

**代码位置：** `verl/trainer/ppo/core_algos.py: 第 265 行`

```python
# verl/trainer/ppo/core_algos.py
def compute_gae_advantage_return(...):
    """
    NOTE(sgm): this implementation only consider outcome supervision,
    where the reward is a scalar.
    """
```

**三个核心原因：**

1. **单一标量 Reward**
   - 每个完整的 response 得到一个评分（如 GSM8K 的 正确/错误）
   - 这是一个 **标量值**（scalar），不是 token 级别的密集信号
   - 例如：`"Let's solve... The answer is 100"` → reward = 1.0（整体正确）

2. **所有 RewardManager 实现都采用此设计**

   **代码位置：**
   - `verl/workers/reward_manager/naive.py: 第 100 行`
   - `verl/workers/reward_manager/dapo.py: 第 127 行`
   - `verl/workers/reward_manager/batch.py: 第 110 行`

   ```python
   # verl/workers/reward_manager/naive.py
   def __call__(self, data: DataProto) -> torch.Tensor:
       for i in range(batch_size):
           # 计算整个 response 的 reward（标量）
           reward = self.compute_score(
               data_source=data_source,
               solution_str=response_text,
               ground_truth=ground_truth,
           )

           # 只在最后一个 token 位置赋值
           valid_response_length = compute_response_length(...)
           reward_tensor[i, valid_response_length - 1] = reward  # ← 只有这里！
           # 其他位置都是 0
   ```

3. **计算效率**
   - 只需计算一次标量 reward
   - 不需要逐 token 分配信用（Credit Assignment 由 Advantage 计算负责）

**Reward Tensor 的形状：**
```python
reward_tensor.shape = (batch_size, sequence_length)

# 实际内容示例（batch_size=2, seq_len=10）：
reward_tensor = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 第一个样本，reward=1.0
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 第二个样本，reward=0.0
]
```

---

#### 2.3.2 Reward 如何影响所有 Token？两种机制

虽然 Reward 只放在最后一个 token，但通过 **Advantage 计算**，这个信号会影响到所有 token。不同算法采用不同策略：

---

##### **方式一：GAE（递归反向传播）**

**代码位置：** `verl/trainer/ppo/core_algos.py: 第 214-262 行`

**核心思想：** Reward 通过 TD-error 和递归反向传播自然影响所有 token。

```python
@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,  # shape: (bs, response_length)
    values: torch.Tensor,               # shape: (bs, response_length)
    response_mask: torch.Tensor,        # shape: (bs, response_length)
    gamma: float,  # 折扣因子，通常 0.99
    lam: float,    # GAE lambda，通常 0.95
):
    """
    Args:
        token_level_rewards: 稀疏 reward，只有最后位置非零
            示例：[0, 0, 0, ..., 0, 1.0]
        values: 每个 token 位置的 value 估计（由 Critic 预测）
        response_mask: EOS 后的 padding 位置为 0
    """

    batch_size = values.size(0)
    gen_len = values.size(1)

    advantages = torch.zeros_like(values)
    lastgaelam = 0
    nextvalues = 0  # 最后一个 token 后面没有 value

    # ========== 从后往前递归计算 ==========
    for t in reversed(range(gen_len)):
        # 步骤 1: 计算 TD-error（δ_t）
        # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        #       ↑ 这里使用了 token_level_rewards
        #       在 t = gen_len-1（最后位置）时，token_level_rewards[:, t] = reward（非零）
        #       在其他位置，token_level_rewards[:, t] = 0

        # 步骤 2: 递归计算 Advantage（GAE 公式）
        # A_t = δ_t + γλ * A_{t+1}
        lastgaelam_ = delta + gamma * lam * lastgaelam

        # 步骤 3: 应用 mask（处理 padding）
        nextvalues = values[:, t] * response_mask[:, t] + \
                     (1 - response_mask[:, t]) * nextvalues
        lastgaelam = lastgaelam_ * response_mask[:, t] + \
                     (1 - response_mask[:, t]) * lastgaelam

        advantages[:, t] = lastgaelam

    returns = advantages + values
    return advantages, returns
```

**关键理解：**

1. **最后一个 token（t=n-1）**：
   ```python
   delta_n = reward + 0 - V(s_n)  # nextvalues=0
   A_n = delta_n                    # lastgaelam=0
   ```

2. **倒数第二个 token（t=n-2）**：
   ```python
   delta_{n-1} = 0 + γ * V(s_n) - V(s_{n-1})
   A_{n-1} = delta_{n-1} + γλ * A_n  # ← A_n 包含了 reward 的信息！
   ```

3. **继续往前（t=n-3, n-4, ...）**：
   ```python
   A_t = δ_t + γλ * A_{t+1}
   ```
   每个位置的 Advantage 都依赖后面位置的 Advantage，从而形成反向传播链。

**示例：追踪一个序列**

```
序列：["Let", "me", "think", "...", "100"]  (5 个 tokens)
Rewards:    [0,     0,     0,      0,    1.0]
Values:     [0.1,  0.2,   0.3,    0.5,   0.8]  (Critic 预测)

γ = 0.99, λ = 0.95

从后往前计算：
t=4 (最后):  δ_4 = 1.0 + 0.99*0 - 0.8 = 0.2
             A_4 = 0.2

t=3:         δ_3 = 0 + 0.99*0.8 - 0.5 = 0.292
             A_3 = 0.292 + 0.99*0.95*0.2 = 0.480

t=2:         δ_2 = 0 + 0.99*0.5 - 0.3 = 0.195
             A_2 = 0.195 + 0.99*0.95*0.480 = 0.646

t=1:         δ_1 = 0 + 0.99*0.3 - 0.2 = 0.097
             A_1 = 0.097 + 0.99*0.95*0.646 = 0.703

t=0:         δ_0 = 0 + 0.99*0.2 - 0.1 = 0.098
             A_0 = 0.098 + 0.99*0.95*0.703 = 0.756

最终 Advantages: [0.756, 0.703, 0.646, 0.480, 0.200]
                  ↑ 所有 token 都得到了 advantage 值！
                  ↑ 越靠前的 token，advantage 衰减越多（由 γλ 控制）
```

**结论：** GAE 不需要显式广播，Reward 通过递归反向传播自然影响所有 token。

---

##### **方式二：GRPO（显式广播）**

**代码位置：** `verl/trainer/ppo/core_algos.py: 第 267-330 行`

**核心思想：** 提取标量 reward → 归一化 → **显式复制到所有 token**。

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # shape: (bs, response_length)
    response_mask: torch.Tensor,        # shape: (bs, response_length)
    index: np.ndarray,                  # 每个样本的 prompt_id（用于分组）
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    GRPO 算法步骤：
    1. 提取标量 reward（sum across tokens）
    2. 按 prompt 分组，计算 group mean/std
    3. 归一化：(reward - mean) / std
    4. **广播：复制到所有 token 位置**
    """

    batch_size, response_length = token_level_rewards.shape

    # ========== 步骤 1: 提取标量 reward ==========
    # 由于只有最后一个 token 有 reward，sum 操作实际上提取了这个值
    scores = token_level_rewards.sum(dim=-1)  # shape: (batch_size,)
    # 示例：[[0, 0, 1.0], [0, 0, 0.5]] → [1.0, 0.5]

    # ========== 步骤 2: 按 Group 归一化 ==========
    # 将同一个 prompt 的多个 response 分组
    id2score = defaultdict(list)
    for i in range(batch_size):
        id2score[index[i]].append(scores[i])

    # 计算每个 group 的 mean 和 std
    id2mean = {}
    id2std = {}
    for idx in id2score:
        if len(id2score[idx]) > 1:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor)

    # 归一化：(score - group_mean) / group_std
    for i in range(batch_size):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]

    # 此时 scores.shape = (batch_size,)
    # 示例：[0.5, -0.3, 1.2, -0.8]

    # ========== 步骤 3: **广播到所有 token** ==========
    # 这是关键的广播操作！
    scores = scores.unsqueeze(-1) * response_mask
    #        ↑ shape: (bs,) → (bs, 1)
    #                          × (bs, response_length) → (bs, response_length)

    # 示例：
    # scores = [0.5, -0.3]  (batch_size=2)
    # response_mask = [[1, 1, 1, 0, 0],  # 前 3 个 token 有效，后 2 个是 padding
    #                   [1, 1, 1, 1, 1]]  # 所有 5 个 token 有效
    #
    # scores.unsqueeze(-1) = [[0.5], [-0.3]]
    #
    # 广播结果：
    # scores = [[0.5, 0.5, 0.5, 0.0, 0.0],   # 有效 token 都是 0.5，padding 是 0
    #           [-0.3, -0.3, -0.3, -0.3, -0.3]]  # 所有 token 都是 -0.3

    return scores, scores  # (advantages, returns)
```

**关键理解：**

1. **`unsqueeze(-1)` 的作用**：
   ```python
   # Before: (batch_size,)
   scores = torch.tensor([0.5, -0.3])

   # After: (batch_size, 1)
   scores = torch.tensor([[0.5], [-0.3]])
   ```

2. **Broadcasting 机制**：
   ```python
   # PyTorch 自动将 (batch_size, 1) 广播到 (batch_size, response_length)
   (bs, 1) × (bs, response_length) → (bs, response_length)

   # 每个标量值复制到整行
   [[0.5]] × [[1, 1, 1, 0, 0]] = [[0.5, 0.5, 0.5, 0.0, 0.0]]
   ```

3. **response_mask 的作用**：
   - 确保 padding 位置的 advantage 为 0
   - 只有有效 token 位置有 advantage 值

**示例：完整流程**

```
输入：
prompt = "What is 25*4?"
responses (同一个 prompt 的 4 个 response):
  - response_0: "The answer is 100"  → reward = 1.0
  - response_1: "Let me think... 100" → reward = 1.0
  - response_2: "It's 99"             → reward = 0.0
  - response_3: "I don't know"        → reward = 0.0

步骤 1: 提取标量 reward
scores = [1.0, 1.0, 0.0, 0.0]

步骤 2: Group 归一化
group_mean = 0.5
group_std = 0.5
normalized_scores = [(1.0-0.5)/0.5, (1.0-0.5)/0.5, (0.0-0.5)/0.5, (0.0-0.5)/0.5]
                  = [1.0, 1.0, -1.0, -1.0]

步骤 3: 广播
假设 response_0 有 5 个 tokens（无 padding）：
advantages[0] = [1.0, 1.0, 1.0, 1.0, 1.0]
                 ↑ 每个 token 都得到相同的 advantage！

假设 response_2 有 3 个有效 tokens + 2 个 padding：
response_mask[2] = [1, 1, 1, 0, 0]
advantages[2] = [-1.0, -1.0, -1.0, 0.0, 0.0]
                 ↑ 有效 token 都是 -1.0，padding 是 0
```

**结论：** GRPO 通过显式广播，将标量 advantage 复制到所有有效 token 位置。

---

#### 2.3.3 其他算法也使用广播

**所有基于 outcome supervision 的算法都采用类似的广播机制：**

**代码位置：** `verl/trainer/ppo/core_algos.py`

| 算法 | 函数名 | 广播代码行 | 广播方式 |
|------|--------|-----------|---------|
| **GRPO** | `compute_grpo_outcome_advantage` | 328 | `scores.unsqueeze(-1) * response_mask` |
| **GRPO_VECTORIZED** | `compute_grpo_vectorized_outcome_advantage` | 356 | `scalars.unsqueeze(-1) * response_mask` |
| **REINFORCE++** | `compute_reinforce_plus_plus` | 418 | `scores.unsqueeze(-1) * response_mask` |
| **RLOO** | `compute_rloo_outcome_advantage` | 470 | `scores.unsqueeze(-1) * response_mask` |
| **OPO** | `compute_opo_outcome_advantage` | 523 | `scores.unsqueeze(-1) * response_mask` |
| **ReMax** | `compute_remax_outcome_advantage` | 577 | `scalars.unsqueeze(-1) * response_mask` |

**唯一的例外是 GAE**，它通过递归反向传播自然处理。

---

#### 2.3.4 response_mask 的关键作用

**定义：** `response_mask` 标识哪些 token 是有效的（1），哪些是 padding（0）。

**代码位置：** `verl/trainer/ppo/ray_trainer.py`

```python
def compute_response_mask(data: DataProto):
    """
    计算 response 部分的 attention mask

    Returns:
        torch.Tensor: shape (batch_size, response_length)
            - 1.0: 有效 token（包括 EOS）
            - 0.0: padding token
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]

    # 提取 response 部分的 mask
    return attention_mask[:, -response_length:]
```

**为什么需要 mask？**

1. **变长序列**：不同 response 长度不同，需要 padding 对齐
2. **防止污染**：padding 位置不应参与梯度计算
3. **广播过滤**：确保只有有效 token 得到 advantage

**示例：**
```python
# Batch 中的两个 response（长度不同）
response_0 = "The answer is 100"        # 5 个 tokens
response_1 = "100"                      # 1 个 token

# Padding 后（max_length=5）
padded_responses = [
    [token_1, token_2, token_3, token_4, token_5],  # response_0（无 padding）
    [token_1, <pad>,  <pad>,  <pad>,  <pad>],       # response_1（4 个 padding）
]

response_mask = [
    [1, 1, 1, 1, 1],  # 所有位置有效
    [1, 0, 0, 0, 0],  # 只有第一个位置有效
]

# 广播 advantage（假设 normalized_score = 0.5）
advantages = [
    [0.5, 0.5, 0.5, 0.5, 0.5],  # response_0：所有 token 都有 advantage
    [0.5, 0.0, 0.0, 0.0, 0.0],  # response_1：只有第一个 token 有 advantage
]
```

---

#### 2.3.5 完整数据流图示

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. REWARD COMPUTATION (RewardManager)                           │
└─────────────────────────────────────────────────────────────────┘
   Input: Response sequences
   ↓
   responses = ["Let me think... 25 * 4 = 100", "The answer is 99"]
   ↓
   token_level_rewards = [
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  ← reward 只在最后
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   ]
   shape: (batch_size, response_length)

┌─────────────────────────────────────────────────────────────────┐
│ 2A. GAE ADVANTAGE CALCULATION (递归反向传播)                    │
└─────────────────────────────────────────────────────────────────┘
   Input: token_level_rewards, values, response_mask
   ↓
   for t in reversed(range(response_length)):
       delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
       advantages[:, t] = delta + gamma * lam * advantages[:, t+1]
   ↓
   Output: advantages = [
       [0.85, 0.78, 0.69, 0.58, 0.45, 0.30, 0.20],  ← 自然衰减
       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
   ]

┌─────────────────────────────────────────────────────────────────┐
│ 2B. GRPO ADVANTAGE CALCULATION (显式广播)                       │
└─────────────────────────────────────────────────────────────────┘
   Input: token_level_rewards, response_mask, index
   ↓
   步骤 1: 提取标量
   scores = token_level_rewards.sum(dim=-1)  # [1.0, 0.0]
   ↓
   步骤 2: Group 归一化
   normalized_scores = [1.0, -1.0]  # 假设 group_mean=0.5, group_std=0.5
   ↓
   步骤 3: 广播
   advantages = normalized_scores.unsqueeze(-1) * response_mask
   ↓
   Output: advantages = [
       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  ← 所有 token 相同
       [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
   ]

┌─────────────────────────────────────────────────────────────────┐
│ 3. LOSS COMPUTATION (Actor Training)                            │
└─────────────────────────────────────────────────────────────────┘
   Input: advantages, old_log_probs, new_log_probs
   ↓
   ratio = exp(new_log_probs - old_log_probs)  # importance sampling
   clipped_ratio = clip(ratio, 1-ε, 1+ε)
   ↓
   loss = -min(ratio * advantages, clipped_ratio * advantages)
   ↓
   # 每个 token 位置都有 loss，求 mean
   final_loss = loss.sum() / response_mask.sum()
   ↓
   Backpropagation → Update Actor Model
```

---

#### 2.3.6 总结对比

| 维度 | GAE | GRPO |
|------|-----|------|
| **Reward 放置** | 最后一个 token | 最后一个 token |
| **提取方式** | 直接使用 `token_level_rewards[:, t]` | `token_level_rewards.sum(dim=-1)` 提取标量 |
| **广播方式** | 递归反向传播（隐式） | `unsqueeze(-1) * response_mask`（显式）|
| **Advantage 分布** | 非均匀（越靠前越小） | 均匀（所有 token 相同）|
| **依赖 Value** | 是（需要 Critic） | 否（只需要 Group 统计） |
| **计算复杂度** | O(response_length) 递归 | O(batch_size) 归一化 + O(1) 广播 |

**核心要点：**
1. ✅ **Reward 只在最后一个 token**：这是 outcome supervision 的设计
2. ✅ **GAE：递归反向传播**：通过 TD-error 链自然影响所有 token
3. ✅ **GRPO：显式广播**：将标量 advantage 复制到所有 token
4. ✅ **response_mask：过滤 padding**：确保只有有效 token 参与计算

---

## 3. GSM8K Reward 详细解析

### 3.1 GSM8K Reward 函数

**文件位置：** `verl/utils/reward_score/gsm8k.py`

```python
# verl/utils/reward_score/gsm8k.py

def extract_solution(solution_str, method="strict"):
    """从响应中提取答案

    method="strict": 要求格式 "#### 答案"
    method="flexible": 提取最后一个数字

    示例：
    输入: "Let's solve step by step...\\n#### 100"
    输出: "100"
    """

    # 优化：只检查最后 300 字符
    if len(solution_str) > 300:
        solution_str = solution_str[-300:]

    if method == "strict":
        # 匹配 "#### 数字" 格式
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            return None
        else:
            # 取最后一个匹配
            final_answer = solutions[-1].replace(",", "").replace("$", "")
            return final_answer

    elif method == "flexible":
        # 提取所有数字，取最后一个
        numbers = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        if len(numbers) == 0:
            return None
        # 从后往前找第一个有效数字
        for num in reversed(numbers):
            if num not in ["", "."]:
                return num


def compute_score(solution_str, ground_truth,
                 method="strict",
                 format_score=0.0,
                 score=1.0):
    """GSM8K 打分函数

    Args:
        solution_str: 模型生成的响应
        ground_truth: 标准答案
        method: "strict" 或 "flexible"
        format_score: 格式正确但答案错误的分数（默认 0）
        score: 答案正确的分数（默认 1.0）

    Returns:
        float: reward 分数
    """

    # 步骤 1: 提取答案
    answer = extract_solution(solution_str, method=method)

    # 步骤 2: 打分
    if answer is None:
        # 没有答案 → 0 分
        return 0
    elif answer == ground_truth:
        # 答案正确 → 满分
        return score
    else:
        # 格式正确但答案错误 → format_score（通常是 0）
        return format_score
```

### 3.2 GSM8K Reward 示例

**例子 1：完美响应**
```python
# 输入
solution_str = """
Let me solve this step by step:
1. We need to calculate 25 * 4
2. 25 * 4 = 100

#### 100
"""
ground_truth = "100"

# 处理过程
answer = extract_solution(solution_str, "strict")
# → answer = "100"

score = compute_score(solution_str, ground_truth)
# → score = 1.0 ✓
```

**例子 2：答案错误**
```python
# 输入
solution_str = """
Let me calculate:
25 * 4 = 90

#### 90
"""
ground_truth = "100"

# 处理过程
answer = extract_solution(solution_str, "strict")
# → answer = "90"

score = compute_score(solution_str, ground_truth)
# → score = 0.0 ✗（答案错误）
```

**例子 3：格式错误**
```python
# 输入
solution_str = """
The answer is 100
"""
ground_truth = "100"

# 处理过程
answer = extract_solution(solution_str, "strict")
# → answer = None（没有 "####"）

score = compute_score(solution_str, ground_truth)
# → score = 0.0 ✗（格式错误）
```

**例子 4：flexible 模式**
```python
# 输入
solution_str = """
The final answer is 100
"""
ground_truth = "100"

# 处理过程（flexible 模式）
answer = extract_solution(solution_str, "flexible")
# → answer = "100"（提取最后一个数字）

score = compute_score(solution_str, ground_truth, method="flexible")
# → score = 1.0 ✓
```

---

## 4. 自定义 Reward 函数

### 4.1 Reward 函数签名

```python
def my_reward_function(
    data_source: str,      # 数据来源（如 "gsm8k"）
    solution_str: str,     # 模型生成的响应
    ground_truth: Any,     # 标准答案
    extra_info: dict = None  # 额外信息（可选）
) -> float:
    """自定义 reward 函数

    返回：
        float: reward 分数（通常 0-1 之间）
    """
    pass
```

### 4.2 实例 1：代码生成 Reward

```python
# my_code_reward.py

import subprocess
import tempfile
import os

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """代码生成任务的 Reward

    评估：
    1. 代码是否能运行
    2. 是否通过测试用例
    """

    # 步骤 1: 提取代码
    code = extract_code_block(solution_str)
    if code is None:
        return 0.0

    # 步骤 2: 获取测试用例
    test_cases = extra_info.get('test_cases', [])
    if not test_cases:
        return 0.0

    # 步骤 3: 执行测试
    passed = 0
    for test_case in test_cases:
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(code)
                f.write(f"\\n{test_case}")
                temp_file = f.name

            # 执行代码
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                timeout=5,
                text=True
            )

            # 检查结果
            if result.returncode == 0:
                passed += 1

            # 清理
            os.unlink(temp_file)

        except Exception as e:
            # 运行失败
            pass

    # 步骤 4: 计算分数
    score = passed / len(test_cases)
    return score


def extract_code_block(text):
    """从响应中提取代码块"""
    import re

    # 匹配 ```python ... ``` 或 ```...```
    patterns = [
        r"```python\\n(.+?)```",
        r"```\\n(.+?)```"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)

    return None
```

### 4.3 实例 2：多目标 Reward

```python
# multi_objective_reward.py

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """多目标 Reward：正确性 + 简洁性

    Reward = 0.7 * correctness + 0.3 * conciseness
    """

    # 目标 1: 正确性
    answer = extract_answer(solution_str)
    if answer == ground_truth:
        correctness = 1.0
    else:
        correctness = 0.0

    # 目标 2: 简洁性（惩罚过长的响应）
    response_length = len(solution_str)
    if response_length < 100:
        conciseness = 1.0
    elif response_length < 300:
        conciseness = 0.5
    else:
        conciseness = 0.0

    # 组合
    final_score = 0.7 * correctness + 0.3 * conciseness
    return final_score
```

### 4.4 实例 3：Reward Shaping

```python
# reward_shaping.py

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Reward Shaping：提供中间奖励

    不只是最终答案，中间步骤也给 reward
    """

    # 最终答案 reward
    answer = extract_answer(solution_str)
    if answer == ground_truth:
        final_reward = 1.0
    else:
        final_reward = 0.0

    # 中间步骤 reward
    intermediate_reward = 0.0

    # 检查是否包含关键步骤
    if "step by step" in solution_str.lower():
        intermediate_reward += 0.1

    if "let me think" in solution_str.lower():
        intermediate_reward += 0.05

    # 检查是否列出计算步骤
    if "=" in solution_str:
        intermediate_reward += 0.1

    # 总 reward
    total_reward = final_reward + intermediate_reward
    return min(total_reward, 1.0)  # 限制在 [0, 1]
```

### 4.5 使用自定义 Reward

```bash
# 方法 1：通过配置文件
python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=/path/to/my_code_reward.py \
    custom_reward_function.name=compute_score

# 方法 2：如果函数名就是 compute_score，可以省略 name
python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=/path/to/my_reward.py
```

---

## 5. Reward 调试技巧

### 5.1 打印 Reward 详情

```python
# 在 reward 函数中添加调试输出

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # 提取答案
    answer = extract_answer(solution_str)

    # 计算分数
    if answer == ground_truth:
        score = 1.0
    else:
        score = 0.0

    # 调试输出
    if extra_info and extra_info.get('debug', False):
        print(f"[DEBUG Reward]")
        print(f"  Solution (前100字符): {solution_str[:100]}")
        print(f"  提取答案: {answer}")
        print(f"  标准答案: {ground_truth}")
        print(f"  分数: {score}")

    return score
```

### 5.2 统计 Reward 分布

```python
# 在 RewardManager 中收集统计

class DebugRewardManager(NaiveRewardManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_history = []

    def __call__(self, data):
        rewards = super().__call__(data)

        # 收集统计
        self.reward_history.append(rewards.mean().item())

        # 每 100 个 batch 打印统计
        if len(self.reward_history) % 100 == 0:
            import numpy as np
            print(f"[Reward Stats] 最近 100 个 batch:")
            print(f"  平均: {np.mean(self.reward_history[-100:]):.3f}")
            print(f"  最大: {np.max(self.reward_history[-100:]):.3f}")
            print(f"  最小: {np.min(self.reward_history[-100:]):.3f}")

        return rewards
```

### 5.3 保存失败样本

```python
# 保存 reward=0 的样本用于分析

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    answer = extract_answer(solution_str)
    score = 1.0 if answer == ground_truth else 0.0

    # 保存失败样本
    if score == 0.0:
        import json
        with open('failed_samples.jsonl', 'a') as f:
            sample = {
                'solution': solution_str,
                'extracted_answer': answer,
                'ground_truth': ground_truth,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\\n')

    return score
```

### 5.4 可视化 Reward

```python
# visualization.py
import pandas as pd
import matplotlib.pyplot as plt

# 读取训练日志
df = pd.read_csv('training_log.csv')

# 绘制 reward 曲线
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['reward_mean'])
plt.xlabel('Training Step')
plt.ylabel('Mean Reward')
plt.title('Reward Progression')
plt.savefig('reward_curve.png')
```

---

## 6. 常见问题解决

### Q1: Reward 一直是 0

**可能原因：**

1. **格式不匹配**
```python
# 检查：模型输出格式
print(f"Response: {solution_str}")
# 是否符合 reward 函数的预期格式？
```

2. **ground_truth 字段缺失**
```python
# 检查数据
import pandas as pd
df = pd.read_parquet('train.parquet')
print(df.iloc[0]['reward_model'])
# 应该有 'ground_truth' 字段
```

3. **Reward 函数报错但被忽略**
```python
# 添加 try-except 捕获
def compute_score(...):
    try:
        # 你的逻辑
        ...
    except Exception as e:
        print(f"❌ Reward 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return 0.0
```

### Q2: Reward 不稳定

**现象：** Reward 在相似的响应上给出不同分数

**解决方法：**
```python
# 添加答案标准化
def normalize_answer(answer):
    """标准化答案格式"""
    answer = answer.strip()
    answer = answer.lower()
    answer = answer.replace(',', '')
    answer = answer.replace('$', '')
    return answer

def compute_score(data_source, solution_str, ground_truth, ...):
    answer = extract_answer(solution_str)
    answer = normalize_answer(answer)
    ground_truth = normalize_answer(ground_truth)

    if answer == ground_truth:
        return 1.0
    else:
        return 0.0
```

### Q3: Reward 计算太慢

**解决方法：**

1. **并行化**
```python
# 使用 multiprocessing
from multiprocessing import Pool

class ParallelRewardManager(NaiveRewardManager):
    def __init__(self, *args, num_workers=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = Pool(num_workers)

    def __call__(self, data):
        # 并行计算 reward
        rewards = self.pool.map(self.compute_single, data)
        return torch.tensor(rewards)
```

2. **优化正则表达式**
```python
# 只检查字符串末尾
def extract_solution(solution_str):
    # 只取最后 300 字符
    if len(solution_str) > 300:
        solution_str = solution_str[-300:]

    # 提取答案
    ...
```

### Q4: 如何设计好的 Reward？

**原则：**

1. **Sparse vs Dense**
```python
# Sparse（稀疏）：只有最终答案有 reward
# 优点：简单
# 缺点：学习慢

# Dense（密集）：中间步骤也有 reward
# 优点：学习快
# 缺点：可能引导错误行为（reward hacking）
```

2. **避免 Reward Hacking**
```python
# 错误示例：只根据长度给 reward
def bad_reward(solution_str, ...):
    # ❌ 模型会学会输出很长的无意义文本
    return len(solution_str) / 1000

# 正确示例：结合多个指标
def good_reward(solution_str, ground_truth, ...):
    correctness = check_answer(solution_str, ground_truth)
    length_penalty = max(0, 1 - len(solution_str) / 500)
    return correctness * (1 + 0.1 * length_penalty)
```

3. **归一化 Reward**
```python
# 将 reward 归一化到 [0, 1] 范围
def compute_score(...):
    raw_score = calculate_raw_score(...)

    # 归一化
    normalized_score = min(max(raw_score, 0.0), 1.0)

    return normalized_score
```

---

## 7. 总结

**Reward 系统的核心流程：**
```
1. 加载 RewardManager（训练开始时）
   ↓
2. 生成响应后，调用 reward_manager(data)
   ↓
3. 对每个样本：
   - Decode 响应
   - 调用 compute_score
   - 将分数放到最后一个 token
   ↓
4. 返回 reward tensor
```

**设计 Reward 的关键点：**
- ✅ 明确目标（什么是"好"的输出）
- ✅ 考虑中间步骤（Dense reward）
- ✅ 避免 Reward Hacking
- ✅ 归一化分数
- ✅ 充分测试和调试

---

## 📚 延伸阅读

- [Reward Function 官方文档](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- [数据流详解](./数据流详解.md)
- [RayPPOTrainer 详解](../01_快速上手/ray_trainer_详解.md)
