# 03 - RL 算法

> 第三部分：深入理解 PPO、GRPO 等强化学习算法

---

## 📚 本章内容

### 📖 学习笔记

#### **03_RL算法概览.md** - 算法对比与选择指南
- verl 支持的 RL 算法总览
- GRPO vs PPO vs RLOO 对比
- 算法选择建议
- 配置切换方法

#### **GRPO_详解.md** - GRPO 算法深度解析（新！）
- GRPO 核心思想（Group Relative 优势估计）
- 无需 Critic 模型的优势
- `compute_grpo_outcome_advantage` 源码分析
  - 分组和均值计算
  - 标准差归一化
  - 优势值广播到 token 维度
- DrGRPO 变体详解
- 完整训练流程示例
- 配置参数详解

#### **PPO_详解.md** - PPO 算法深度解析（新！）
- PPO 核心思想（Clipped Surrogate Objective）
- Actor-Critic 架构
- GAE（Generalized Advantage Estimation）源码分析
  - TD-error 计算
  - 优势值递推
  - Baseline 减去值函数
- `compute_policy_loss` 源码分析
  - Ratio 计算
  - Clipping 机制
  - Dual-clip PPO
- KL 散度控制（KL reward vs KL loss）
- 完整训练流程示例
- 配置参数详解

---

## 🚀 快速开始

### 步骤 1：理解算法区别

```bash
# 阅读算法概览
cat 03_RL算法概览.md
```

关键区别：
- **GRPO**: 无 Critic，基于组相对奖励，更快
- **PPO**: 有 Critic，GAE 优势估计，更稳定

### 步骤 2：查看 GRPO 源码

```bash
# 核心算法实现
cat verl/trainer/ppo/core_algos.py:266-330
```

### 步骤 3：切换算法

```bash
# 使用 GRPO（推荐入门）
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.n=4

# 使用 PPO（更稳定）
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    critic.model.path=Qwen/Qwen2.5-7B-Instruct
```

---

## 📖 推荐学习路径

### 第 1 天：算法概览和 GRPO

1. **阅读** `03_RL算法概览.md`（1 小时）
   - 理解不同算法的应用场景
   - 掌握算法选择标准

2. **阅读** `GRPO_详解.md`（2-3 小时）
   - 深入理解 GRPO 原理
   - 阅读源码实现
   - 理解分组机制和优势计算

3. **实践** 运行 GRPO 训练
   ```bash
   # 使用 GSM8K 数据
   python3 -m verl.trainer.main_ppo \
       data.train_files=$HOME/data/gsm8k/train.parquet \
       data.val_files=$HOME/data/gsm8k/test.parquet \
       data.train_batch_size=256 \
       actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
       actor_rollout_ref.rollout.n=4 \
       algorithm.adv_estimator=grpo \
       actor_rollout_ref.actor.use_kl_loss=true
   ```

4. **调试** 添加优势计算日志
   ```python
   # 在 verl/trainer/ppo/core_algos.py:303 添加
   print(f"[GRPO Debug] Scores: {scores}")
   print(f"  Group means: {list(id2mean.values())[:5]}")
   print(f"  Group stds: {list(id2std.values())[:5]}")
   print(f"  Normalized advantages: {scores[:5]}")
   ```

### 第 2 天：PPO 算法

1. **阅读** `PPO_详解.md`（2-3 小时）
   - 理解 Actor-Critic 架构
   - 掌握 GAE 优势估计
   - 理解 Clipping 机制

2. **对比** GAE vs GRPO 源码
   ```bash
   # 查看 GAE 实现
   grep -A 30 "def compute_gae" verl/trainer/ppo/core_algos.py

   # 查看 GRPO 实现
   grep -A 30 "def compute_grpo" verl/trainer/ppo/core_algos.py
   ```

3. **实践** 运行 PPO 训练
   ```bash
   python3 -m verl.trainer.main_ppo \
       data.train_files=$HOME/data/gsm8k/train.parquet \
       algorithm.adv_estimator=gae \
       critic.model.path=Qwen/Qwen2.5-7B-Instruct \
       actor_rollout_ref.actor.ppo_epochs=2 \
       critic.ppo_epochs=2
   ```

### 第 3 天：算法对比实验

1. **实验 1**：GRPO vs PPO 在 GSM8K 上的效果
   - 运行相同配置的 GRPO 和 PPO
   - 对比 reward/mean 曲线
   - 对比训练速度

2. **实验 2**：不同 rollout.n 的影响
   ```bash
   # GRPO with n=2
   python3 -m verl.trainer.main_ppo \
       algorithm.adv_estimator=grpo \
       actor_rollout_ref.rollout.n=2

   # GRPO with n=4
   python3 -m verl.trainer.main_ppo \
       algorithm.adv_estimator=grpo \
       actor_rollout_ref.rollout.n=4
   ```

3. **实验 3**：KL 控制策略对比
   ```bash
   # KL loss (GRPO 推荐)
   python3 -m verl.trainer.main_ppo \
       actor_rollout_ref.actor.use_kl_loss=true \
       actor_rollout_ref.actor.kl_loss_coef=0.001

   # KL reward penalty
   python3 -m verl.trainer.main_ppo \
       algorithm.use_kl_in_reward=true \
       algorithm.kl_ctrl.kl_coef=0.001
   ```

---

## 📋 学习检查清单

### 算法理解 ✓
- [ ] 理解 PPO、GRPO、RLOO 的核心区别
- [ ] 掌握 GRPO 的分组机制
- [ ] 掌握 PPO 的 GAE 优势估计
- [ ] 理解 Clipping 在 PPO 中的作用
- [ ] 知道何时选择 GRPO vs PPO

### 源码阅读 ✓
- [ ] 阅读 `compute_grpo_outcome_advantage` 实现
- [ ] 阅读 `compute_gae` 实现
- [ ] 理解 `compute_policy_loss` 中的 clipping
- [ ] 理解 KL 散度的两种控制方式

### 实践能力 ✓
- [ ] 运行过 GRPO 训练
- [ ] 运行过 PPO 训练
- [ ] 能够切换不同的优势估计器
- [ ] 能够调整 rollout.n 和 batch size
- [ ] 能够配置 KL 控制策略

---

## 🎯 学习目标

完成本章后，你应该能够：

✅ 深入理解 PPO 和 GRPO 算法原理
✅ 阅读和理解 core_algos.py 源码
✅ 根据任务特点选择合适的算法
✅ 熟练配置算法参数
✅ 调试优势计算和策略更新
✅ 进行算法对比实验

---

## 💡 重点内容

### GRPO 优势计算公式

对于每个组 g（同一个 prompt 生成的 n 个响应）：

```python
# 1. 计算每个响应的总奖励
score_i = sum(token_rewards[i])

# 2. 计算组内均值和标准差
mean_g = mean(score_1, score_2, ..., score_n)
std_g = std(score_1, score_2, ..., score_n)

# 3. 归一化优势值
advantage_i = (score_i - mean_g) / (std_g + epsilon)

# 4. 广播到 token 维度
advantages[i, :] = advantage_i * response_mask[i, :]
```

### GAE 优势计算公式

逐步从后向前计算：

```python
for t in reversed(range(T)):
    # TD-error
    delta_t = reward[t] + gamma * V[t+1] - V[t]

    # GAE 递推
    A[t] = delta_t + gamma * lambda * A[t+1]

# 归一化
A = (A - mean(A)) / (std(A) + epsilon)
```

### PPO Clipped Objective

```python
# 计算概率比
ratio = exp(new_log_prob - old_log_prob)

# Clipped surrogate
loss1 = ratio * advantage
loss2 = clip(ratio, 1-epsilon, 1+epsilon) * advantage
loss = -min(loss1, loss2)
```

### 算法选择建议

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| **数学推理（GSM8K）** | GRPO | 结果导向，无需过程监督 |
| **代码生成** | GRPO | 可执行性是二元结果 |
| **长文本生成** | PPO | Critic 提供更细粒度的价值估计 |
| **对话质量** | PPO | 需要细致的价值函数 |
| **快速实验** | GRPO | 训练速度快，无需 Critic |
| **追求稳定性** | PPO | GAE 方差更小 |

---

## ❓ 常见问题

### Q1: GRPO 和 PPO 哪个更好？

**取决于任务**：
- **结果导向任务**（如数学题、代码生成）：GRPO 更简单高效
- **过程导向任务**（如长文本、对话）：PPO 更稳定

**资源考虑**：
- GRPO 不需要 Critic 模型，节省 GPU 显存和训练时间
- PPO 需要同时训练 Actor 和 Critic

### Q2: rollout.n 设置多少合适？

**GRPO 推荐**：n ≥ 4
- 太小（n=1, 2）：组内方差不准确
- 太大（n>8）：计算开销大，收益递减

**PPO**：n=1 即可
- PPO 使用 Critic 提供 baseline，不需要多个样本

### Q3: 为什么 GRPO 要用 use_kl_loss=true？

GRPO 不在 reward 中加 KL penalty，而是：
```python
loss = policy_loss + kl_loss_coef * kl_divergence
```

这样可以：
- 直接在梯度中控制 KL
- 避免 reward shaping 的影响

### Q4: GAE 中的 lambda 怎么设置？

**lambda** 控制 bias-variance tradeoff：
- `lambda=0`: 只用 1-step TD（低方差，高偏差）
- `lambda=1`: 用完整 Monte Carlo（高方差，无偏）
- `lambda=0.95`（默认）：折中选择

### Q5: 训练不稳定怎么办？

**GRPO 不稳定**：
- 增大 `rollout.n`（更多样本）
- 减小 `clip_ratio`（更保守的更新）
- 使用 DrGRPO（`loss_agg_mode="seq-mean-token-sum-norm"`）

**PPO 不稳定**：
- 调整 `gamma` 和 `lam`
- 增加 `ppo_epochs`
- 使用 Dual-clip PPO

---

## 🔗 相关资源

### 本地文件
- 算法概览: `03_RL算法概览.md`
- GRPO 详解: `GRPO_详解.md`
- PPO 详解: `PPO_详解.md`
- 项目概览: `../../CLAUDE.md`
- 完整学习路线: `../../LEARNING_GUIDE.md`

### 官方文档
- [GRPO 文档](https://verl.readthedocs.io/en/latest/algo/grpo.html)
- [PPO 文档](https://verl.readthedocs.io/en/latest/algo/ppo.html)
- [Baseline Performance](https://verl.readthedocs.io/en/latest/algo/baseline.html)

### 代码位置
- 核心算法: `verl/trainer/ppo/core_algos.py`
  - GRPO: 第 266-330 行
  - GAE: 第 210-262 行
  - Policy Loss: 第 450-550 行
- 配置文件: `verl/trainer/config/ppo_trainer.yaml`
- 算法注册: `verl/trainer/ppo/core_algos.py:112-150`

### 论文和参考
- [DeepSeekMath (GRPO)](https://arxiv.org/pdf/2402.03300)
- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [DrGRPO 论文](https://arxiv.org/pdf/2503.20783)

---

## ⏭️ 下一步

完成本章后，继续学习：
- **04 - Reward 设计**: 更多自定义 Reward 实现和调优技巧
- **05 - Agent RL**: 工具调用和多轮对话的 RL 训练

---

*创建时间: 2026-01-26*
*预计完成时间: 3-4 天*
