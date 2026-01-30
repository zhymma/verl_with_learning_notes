# 02 - 数据准备

> 第二部分：深入理解数据格式和 Reward 系统

---

## 📚 本章内容

### 📖 学习笔记

#### **02_数据准备.md** - 数据格式详解（10000+ 字）
- verl 的 Parquet 数据格式
- 4 种 prompt 格式详解
- 单轮对话数据准备
- 多轮对话数据准备（Agent）
- 多模态数据准备（VLM）
- 数据质量要求和最佳实践

#### **reward_系统详解.md** - Reward 系统深度解析（新！）
- RewardManager 架构和调用流程
- reward_model 配置详解
- GSM8K Reward 源码分析
  - strict 和 flexible 两种提取方法
  - 正则表达式匹配逻辑
  - 完整的计算流程追踪
- 自定义 Reward 函数实现
  - Rule-based Reward 示例
  - Model-based Reward 示例
  - 代码生成 Reward
  - 多目标 Reward
- Reward 调试技巧
- 常见问题和解决方案

### 🛠️ 核心脚本

- **data_quality_check.py** - 数据格式和质量检查

---

## 🚀 快速开始

### 步骤 1：理解数据格式

```bash
# 查看 GSM8K 数据示例
python -c "
import pandas as pd
df = pd.read_parquet('~/data/gsm8k/train.parquet')
print(df.head(1))
print(df.columns.tolist())
"
```

### 步骤 2：检查数据质量

```bash
python data_quality_check.py ~/data/gsm8k/train.parquet
```

### 步骤 3：准备自己的数据

参考 `02_数据准备.md` 第 2-4 节的详细示例。

---

## 📖 推荐学习路径

### 第 1 天：数据格式理解

1. **阅读** `02_数据准备.md`（2 小时）
   - 理解 Parquet 格式
   - 掌握 4 种 prompt 格式
   - 学习数据准备流程

2. **实践** 查看 GSM8K 数据
   ```bash
   # 查看数据结构
   python -c "
   import pandas as pd
   import json
   df = pd.read_parquet('~/data/gsm8k/train.parquet')
   sample = df.iloc[0].to_dict()
   print(json.dumps(sample, indent=2, ensure_ascii=False))
   "
   ```

3. **理解** reward_model 字段
   - 查看 GSM8K 的 reward_model 配置
   - 理解不同 Reward 函数的作用

### 第 2 天：Reward 系统深入

1. **阅读** `reward_系统详解.md`（2 小时）
   - 理解 RewardManager 架构
   - 掌握 GSM8K Reward 实现
   - 学习自定义 Reward 方法

2. **实践** 追踪 Reward 计算
   ```python
   # 在 verl/trainer/ppo/reward.py 的 RewardManager.__call__ 中添加
   print(f"[Debug] Computing reward for batch_size={len(batch)}")
   print(f"  reward_model config: {batch['reward_model'][0]}")

   # 在 verl/utils/reward_score/gsm8k.py 的 compute_score 中添加
   print(f"[Debug] solution: {solution_str}")
   print(f"  ground_truth: {ground_truth}")
   print(f"  extracted: {answer}")
   print(f"  score: {result}")
   ```

### 第 3 天：实践和调试

1. **准备** 自己的数据集
   - 选择一个任务（如代码生成、数学问题）
   - 按照 Parquet 格式准备数据
   - 配置 reward_model

2. **实现** 自定义 Reward
   - 参考 `reward_系统详解.md` 中的示例
   - 在 `verl/utils/reward_score/` 下创建新文件
   - 测试 Reward 函数

---

## 📋 学习检查清单

### 数据格式理解 ✓
- [ ] 理解 Parquet 格式和必需字段
- [ ] 掌握 4 种 prompt 格式（String, StringList, Chat, ChatList）
- [ ] 理解 data_source 和 reward_model 的作用
- [ ] 能够检查数据格式是否正确

### Reward 系统掌握 ✓
- [ ] 理解 RewardManager 的调用流程
- [ ] 掌握 reward_model 配置方法
- [ ] 理解 GSM8K Reward 的实现原理
- [ ] 能够阅读和理解 Reward 函数源码
- [ ] 知道如何调试 Reward 计算

### 数据准备实践 ✓
- [ ] 准备过单轮对话数据
- [ ] 理解多轮对话数据格式
- [ ] （可选）准备过多模态数据
- [ ] 能够实现自定义 Reward 函数

---

## 🎯 学习目标

完成本章后，你应该能够：

✅ 深入理解 verl 的 Parquet 数据格式
✅ 掌握 4 种 prompt 格式的使用场景
✅ 理解 RewardManager 的工作原理
✅ 阅读和理解 Reward 函数源码
✅ 准备各种类型的训练数据
✅ 实现自定义 Reward 函数
✅ 调试 Reward 计算问题

---

## 💡 重点内容

### 数据格式的 3 个必需字段

```python
{
    "data_source": "gsm8k",              # 数据来源标识
    "prompt": "What is 2+2?",            # 输入（4 种格式之一）
    "reward_model": {                    # Reward 计算配置
        "style": "rule",
        "module": "verl.utils.reward_score.gsm8k",
        "ground_truth": "4"
    }
}
```

### GSM8K Reward 的两种提取方法

**strict 方法**（推荐）：
```python
# 匹配 "#### number" 格式
solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
# 示例：
# "#### 42" → "42"
# "#### -3.14" → "-3.14"
```

**flexible 方法**：
```python
# 提取最后一个数字
numbers = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
# 示例：
# "The answer is 42." → "42"
# "We get 3.14 meters" → "3.14"
```

### Reward 配置的 3 种 style

```yaml
# 1. rule-based
reward_model:
  style: "rule"
  module: "verl.utils.reward_score.gsm8k"

# 2. model-based
reward_model:
  style: "model"
  path: "path/to/reward_model"

# 3. sandbox（代码执行）
reward_model:
  style: "sandbox"
  language: "python"
```

---

## ❓ 常见问题

### Q1: prompt 应该用什么格式？

**推荐使用 Chat 格式**（列表形式）：
```python
"prompt": [{"role": "user", "content": "解这道题..."}]
```

其他格式见 `02_数据准备.md` 第 1.3 节。

### Q2: reward_model 的 3 个字段都必需吗？

**必需字段**：
- `style`: "rule" | "model" | "sandbox"

**根据 style 不同**：
- rule → 需要 `module` 指定计算函数
- model → 需要 `path` 指定模型路径
- sandbox → 需要 `language` 指定执行语言

### Q3: GSM8K 的 Reward 怎么计算？

查看 `reward_系统详解.md` 第 2 节，核心逻辑：
1. 用正则提取答案（`#### number`）
2. 与 ground_truth 比较
3. 相等返回 1.0，否则返回 0.0

### Q4: 如何调试 Reward 计算错误？

**方法 1：查看 Reward 日志**
```python
# verl/trainer/ppo/reward.py 添加
print(f"[Debug] Reward: {rewards}")
```

**方法 2：单独测试 Reward 函数**
```python
from verl.utils.reward_score.gsm8k import compute_score
score = compute_score("#### 42", "42")
print(score)  # 应该是 1.0
```

详见 `reward_系统详解.md` 第 5 节。

### Q5: 如何准备多轮对话数据？

**使用 ChatList 格式**：
```python
"prompt": [
    [
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回答"},
        {"role": "user", "content": "第二轮问题"}
    ]
]
```

详见 `02_数据准备.md` 第 3 节。

---

## 🔗 相关资源

### 本地文件
- 数据格式详解: `02_数据准备.md`
- Reward 系统详解: `reward_系统详解.md`
- 项目概览: `../../CLAUDE.md`
- 完整学习路线: `../../LEARNING_GUIDE.md`
- 第一部分: `../01_快速上手/`

### 官方文档
- [Prepare Data](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
- [Reward Function](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)

### 代码位置
- 数据预处理示例: `examples/data_preprocess/`
- RewardManager: `verl/trainer/ppo/reward.py`
- Reward 函数库: `verl/utils/reward_score/`
- 数据加载器: `verl/utils/dataset/rl_dataset.py`

---

## ⏭️ 下一步

完成本章后，继续学习：
- **03 - RL 算法**: 深入理解 GRPO、PPO、RLOO 等算法实现
- **04 - Reward 设计**: 更多自定义 Reward 示例和最佳实践
- **05 - Agent RL**: 工具调用和多轮对话训练

---

*创建时间: 2026-01-25*
*预计完成时间: 2-3 天*
