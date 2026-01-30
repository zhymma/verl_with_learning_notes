# 04 - Reward 设计

> 第四部分：深入理解 Reward 系统和实现自定义 Reward 函数

---

## 📚 本章内容

### 📖 学习笔记

#### **自定义Reward实践指南.md** - 完整实战教程（新！）
- Reward 函数基础概念
- RewardManager 调用流程详解
- 3 种 Reward 类型对比（Rule-based, Model-based, Sandbox）
- 10+ 个实战示例
  - 数学推理 Reward（GSM8K, MATH）
  - 代码生成 Reward（HumanEval, MBPP）
  - 文本质量 Reward（长度、格式、多样性）
  - 多目标 Reward（准确性 + 简洁性）
  - Reward Shaping 技巧
- Reward 调试完整流程
- 最佳实践和常见错误
- 性能优化技巧

### 🛠️ 实战代码

本部分提供**源码级别的示例分析**，所有示例都可以在以下位置找到：
- 内置 Reward: `verl/utils/reward_score/`
- 示例数据: `examples/data_preprocess/`

---

## 🚀 快速开始

### 步骤 1：理解 Reward 类型

```python
# 类型 1: Rule-based Reward（规则匹配）
def gsm8k_reward(solution_str, ground_truth):
    # 提取答案
    answer = extract_solution(solution_str)
    # 比较
    return 1.0 if answer == ground_truth else 0.0

# 类型 2: Model-based Reward（使用 Reward Model）
reward_model = AutoModelForSequenceClassification.from_pretrained("...")
score = reward_model(prompt, response)

# 类型 3: Sandbox Reward（代码执行）
def code_reward(code_str, test_cases):
    # 执行代码
    result = execute_code(code_str, test_cases)
    # 返回通过率
    return result.pass_rate
```

### 步骤 2：实现你的第一个 Reward

创建 `my_reward.py`:
```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    自定义 Reward 函数

    Args:
        data_source (str): 数据来源（如 "my_task"）
        solution_str (str): 模型生成的响应
        ground_truth (str): 正确答案
        extra_info (dict): 额外信息

    Returns:
        float: 奖励分数（通常 0-1）
    """
    # 示例：长度奖励
    target_length = 100
    actual_length = len(solution_str)

    # 奖励在 80-120 字之间的响应
    if 80 <= actual_length <= 120:
        return 1.0
    else:
        # 超出范围，线性惩罚
        penalty = abs(actual_length - target_length) / target_length
        return max(0.0, 1.0 - penalty)
```

### 步骤 3：配置使用自定义 Reward

```bash
python3 -m verl.trainer.main_ppo \
    data.train_files=my_data.parquet \
    custom_reward_function.path=/path/to/my_reward.py \
    custom_reward_function.name=compute_score
```

或者在数据中配置：
```python
# 数据准备时
data = {
    "data_source": "my_task",
    "prompt": "Write a summary...",
    "reward_model": {
        "style": "rule",
        "module": "my_reward",
        "function": "compute_score"
    }
}
```

---

## 📖 推荐学习路径

### 第 1 天：Reward 基础

1. **阅读** `自定义Reward实践指南.md` 第 1-3 节（2 小时）
   - 理解 Reward 函数的作用
   - 掌握 RewardManager 调用流程
   - 了解 3 种 Reward 类型

2. **实践** 查看内置 Reward 实现
   ```bash
   # GSM8K Reward
   cat verl/utils/reward_score/gsm8k.py

   # MATH Reward
   cat verl/utils/reward_score/math_reward.py
   ```

3. **运行** GSM8K 训练，添加 Reward 日志
   ```python
   # 在 verl/trainer/ppo/reward.py 的 RewardManager.__call__ 中
   print(f"[Reward Debug] Batch size: {len(batch)}")
   print(f"  data_source: {batch['data_source'][0]}")
   print(f"  reward_model: {batch['reward_model'][0]}")
   print(f"  scores: {scores[:5]}")
   ```

### 第 2 天：实现自定义 Reward

1. **阅读** `自定义Reward实践指南.md` 第 4-6 节（2 小时）
   - 学习 10+ 个实战示例
   - 理解 Reward Shaping 技巧
   - 掌握调试方法

2. **实践** 实现你的第一个 Reward
   ```python
   # 示例：简洁性奖励
   def brevity_reward(solution_str, ground_truth, target_length=100):
       length = len(solution_str)
       if length <= target_length:
           return 1.0
       else:
           return max(0.0, 1.0 - (length - target_length) / target_length)
   ```

3. **测试** 在小数据集上测试
   ```bash
   # 创建测试数据（10 个样本）
   python prepare_test_data.py

   # 运行训练
   python3 -m verl.trainer.main_ppo \
       data.train_files=test_data.parquet \
       custom_reward_function.path=my_reward.py \
       trainer.total_epochs=1
   ```

### 第 3 天：高级技巧和调优

1. **阅读** `自定义Reward实践指南.md` 第 7-9 节（2 小时）
   - 学习多目标 Reward
   - 掌握性能优化技巧
   - 理解最佳实践

2. **实践** 多目标 Reward
   ```python
   def multi_objective_reward(solution_str, ground_truth):
       # 目标 1：准确性（权重 0.6）
       accuracy = compute_accuracy(solution_str, ground_truth)

       # 目标 2：简洁性（权重 0.2）
       brevity = compute_brevity(solution_str)

       # 目标 3：可读性（权重 0.2）
       readability = compute_readability(solution_str)

       return 0.6 * accuracy + 0.2 * brevity + 0.2 * readability
   ```

3. **对比实验** 测试不同 Reward 设计
   - Sparse Reward vs Dense Reward
   - Binary Reward vs Continuous Reward
   - Single Objective vs Multi-Objective

---

## 📋 学习检查清单

### Reward 基础理解 ✓
- [ ] 理解 Reward 在 RL 训练中的作用
- [ ] 掌握 RewardManager 调用流程
- [ ] 了解 3 种 Reward 类型的区别
- [ ] 理解 reward_model 配置格式
- [ ] 知道如何查看 Reward 计算日志

### 自定义 Reward 实现 ✓
- [ ] 实现过简单的 Rule-based Reward
- [ ] 理解 compute_score 函数签名
- [ ] 能够配置 custom_reward_function
- [ ] 知道如何调试 Reward 计算
- [ ] 理解 Reward Shaping 的作用

### 高级技巧掌握 ✓
- [ ] 实现过多目标 Reward
- [ ] 理解 Sparse vs Dense Reward
- [ ] 掌握 Reward 性能优化
- [ ] 能够分析 Reward 分布
- [ ] 知道常见错误和解决方法

---

## 🎯 学习目标

完成本章后，你应该能够：

✅ 深入理解 Reward 函数的设计原理
✅ 熟练实现各种类型的自定义 Reward
✅ 掌握 Reward Shaping 技巧
✅ 能够调试和优化 Reward 计算
✅ 设计多目标 Reward 函数
✅ 分析 Reward 对训练的影响

---

## 💡 重点内容

### Reward 函数签名

```python
def compute_score(
    data_source: str,        # 数据来源（如 "gsm8k"）
    solution_str: str,       # 模型生成的完整响应
    ground_truth: str,       # 正确答案（从数据中获取）
    extra_info: dict = None  # 额外信息（可选）
) -> float:                  # 返回 0-1 之间的分数
    """
    计算单个响应的奖励分数
    """
    pass
```

### Reward 配置的 3 个 style

```yaml
# Style 1: rule-based
reward_model:
  style: "rule"
  module: "verl.utils.reward_score.gsm8k"
  function: "compute_score"  # 可选

# Style 2: model-based
reward_model:
  style: "model"
  path: "path/to/reward_model"
  model_type: "sequence_classification"

# Style 3: sandbox
reward_model:
  style: "sandbox"
  language: "python"
  test_cases: [...]
```

### Reward 类型对比

| 类型 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Rule-based** | 快速、可解释 | 需要人工设计规则 | 数学推理、格式检查 |
| **Model-based** | 自动学习 | 需要训练 Reward Model | 对话质量、文本生成 |
| **Sandbox** | 准确（可执行） | 慢、需要安全隔离 | 代码生成 |

### Sparse vs Dense Reward

**Sparse Reward（稀疏奖励）：**
```python
# 只有最终结果有奖励
def sparse_reward(solution_str, ground_truth):
    return 1.0 if solution_str == ground_truth else 0.0
```
✅ 优点：简单、明确
❌ 缺点：难以学习、样本效率低

**Dense Reward（密集奖励）：**
```python
# 每一步都有奖励
def dense_reward(solution_str, ground_truth):
    # 正确性
    accuracy = compute_accuracy(solution_str, ground_truth)

    # 中间步骤奖励
    step_rewards = 0.0
    if "Let's solve step by step" in solution_str:
        step_rewards += 0.1
    if "####" in solution_str:
        step_rewards += 0.1

    return accuracy * 0.8 + step_rewards * 0.2
```
✅ 优点：容易学习、样本效率高
❌ 缺点：可能过拟合规则

---

## ❓ 常见问题

### Q1: Reward 分数的范围是什么？

**推荐范围：0-1**
```python
# 好的设计
return 1.0  # 完全正确
return 0.5  # 部分正确
return 0.0  # 完全错误

# 避免
return 100  # 太大，可能导致训练不稳定
return -1.0  # 负数，不推荐（虽然技术上可行）
```

**为什么 0-1？**
- 便于不同 Reward 的对比
- 避免数值不稳定
- 易于理解和调试

### Q2: 如何设计 Reward 的中间值？

**方法 1：分级奖励**
```python
def graded_reward(solution_str, ground_truth):
    answer = extract_answer(solution_str)

    if answer == ground_truth:
        return 1.0  # 完全正确
    elif has_correct_format(solution_str):
        return 0.3  # 格式正确
    else:
        return 0.0  # 完全错误
```

**方法 2：连续奖励**
```python
def continuous_reward(solution_str, ground_truth):
    # 计算相似度
    similarity = compute_similarity(solution_str, ground_truth)
    return similarity  # 0-1 之间的连续值
```

### Q3: Reward 计算慢怎么办？

**优化方法 1：批量计算**
```python
# 不好：逐个计算
for solution in solutions:
    score = compute_score(solution, ground_truth)

# 好：批量计算
scores = batch_compute_score(solutions, ground_truths)
```

**优化方法 2：缓存结果**
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def compute_score_cached(solution_str, ground_truth):
    return compute_score(solution_str, ground_truth)
```

**优化方法 3：并行计算**
```python
from multiprocessing import Pool

def compute_scores_parallel(solutions, ground_truths, n_workers=4):
    with Pool(n_workers) as pool:
        scores = pool.starmap(compute_score, zip(solutions, ground_truths))
    return scores
```

### Q4: Reward 分布不均衡怎么办？

**问题：** 大部分样本都是 0 或 1

**解决方法 1：Reward Shaping**
```python
def shaped_reward(solution_str, ground_truth):
    # 原始 Reward（Binary）
    raw_reward = 1.0 if solution_str == ground_truth else 0.0

    # Shaping: 添加中间奖励
    length_reward = compute_length_reward(solution_str)
    format_reward = compute_format_reward(solution_str)

    # 组合
    return raw_reward * 0.7 + length_reward * 0.15 + format_reward * 0.15
```

**解决方法 2：Reward Normalization**
```python
def normalized_reward(rewards):
    # 标准化到 [-1, 1]
    mean = np.mean(rewards)
    std = np.std(rewards) + 1e-8
    return (rewards - mean) / std
```

### Q5: 如何验证 Reward 函数的正确性？

**步骤 1：单元测试**
```python
def test_reward_function():
    # 测试正确答案
    assert compute_score("#### 42", "42") == 1.0

    # 测试错误答案
    assert compute_score("#### 43", "42") == 0.0

    # 测试格式错误
    assert compute_score("42", "42") < 1.0
```

**步骤 2：手动检查**
```python
# 打印前 10 个样本的 Reward
for i in range(10):
    solution = solutions[i]
    ground_truth = ground_truths[i]
    score = compute_score(solution, ground_truth)
    print(f"Sample {i}:")
    print(f"  Solution: {solution[:50]}...")
    print(f"  Ground Truth: {ground_truth}")
    print(f"  Score: {score}")
```

**步骤 3：统计分析**
```python
import matplotlib.pyplot as plt

scores = [compute_score(s, gt) for s, gt in zip(solutions, ground_truths)]

plt.hist(scores, bins=20)
plt.xlabel("Reward Score")
plt.ylabel("Count")
plt.title("Reward Distribution")
plt.show()

print(f"Mean: {np.mean(scores):.4f}")
print(f"Std: {np.std(scores):.4f}")
print(f"Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")
```

---

## 🔗 相关资源

### 本地文件
- 详细教程: `自定义Reward实践指南.md`
- Reward 系统深度解析: `../02_数据准备/reward_系统详解.md`
- 项目概览: `../../CLAUDE.md`
- 完整学习路线: `../../LEARNING_GUIDE.md`

### 官方文档
- [Reward Function](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- [Prepare Data](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)

### 代码位置
- RewardManager: `verl/trainer/ppo/reward.py`
- 内置 Reward: `verl/utils/reward_score/`
  - GSM8K: `gsm8k.py`
  - MATH: `math_reward.py`
  - Geo3K: `geo3k.py`
- 数据预处理示例: `examples/data_preprocess/`

### 论文参考
- [RLHF with Reward Models](https://arxiv.org/abs/2203.02155)
- [Reward Shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)

---

## ⏭️ 下一步

完成本章后，继续学习：
- **05 - Agent RL**: 工具调用和多轮对话的 RL 训练
- **进阶**: 实现更复杂的 Reward Model

---

*创建时间: 2026-01-26*
*预计完成时间: 2-3 天*
