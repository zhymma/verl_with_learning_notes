# 自定义 Reward 实践指南

> 从零开始掌握 Reward 函数的设计、实现和优化

---

## 📖 目录

1. [Reward 函数基础](#1-reward-函数基础)
2. [RewardManager 调用流程](#2-rewardmanager-调用流程)
3. [3 种 Reward 类型详解](#3-3-种-reward-类型详解)
4. [实战示例 1-5：基础 Reward](#4-实战示例-1-5基础-reward)
5. [实战示例 6-10：高级 Reward](#5-实战示例-6-10高级-reward)
6. [Reward Shaping 技巧](#6-reward-shaping-技巧)
7. [调试和验证](#7-调试和验证)
8. [性能优化](#8-性能优化)
9. [最佳实践](#9-最佳实践)

---

## 1. Reward 函数基础

### 1.1 什么是 Reward 函数？

在强化学习中，**Reward 函数**定义了"什么是好的行为"。它接收模型的输出，返回一个分数，指导模型朝着期望的方向学习。

```
训练流程中的 Reward:
┌──────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────┐
│  Prompt  │ -> │  Model   │ -> │  Response   │ -> │  Reward  │
│          │    │ Generate │    │ "#### 42"   │    │  Score   │
└──────────┘    └──────────┘    └─────────────┘    └──────────┘
                                                           ↓
                                                     ┌──────────┐
                                                     │ 1.0 or 0 │
                                                     └──────────┘
```

**核心作用：**
- 定义"正确"的标准
- 引导模型优化方向
- 决定训练效果的上限

### 1.2 Reward 函数的签名

verl 中的 Reward 函数必须遵循固定的签名：

```python
def compute_score(
    data_source: str,        # 数据来源（如 "gsm8k", "my_task"）
    solution_str: str,       # 模型生成的完整响应
    ground_truth: str,       # 正确答案（从数据的 reward_model 中获取）
    extra_info: dict = None  # 额外信息（可选，用于传递其他参数）
) -> float:                  # 返回奖励分数（通常 0-1）
    """
    计算单个响应的奖励分数

    Args:
        data_source: 用于区分不同数据集，可以在一个函数中处理多个数据集
        solution_str: 模型生成的文本，已经过 detokenize
        ground_truth: 正确答案，从数据中的 reward_model 字段提取
        extra_info: 可选的额外信息字典

    Returns:
        float: 奖励分数，推荐范围 0-1
               1.0 = 完全正确
               0.0 = 完全错误
               0-1 之间 = 部分正确
    """
    pass
```

**参数详解：**

**data_source:**
- 用途：区分不同的任务/数据集
- 示例：`"gsm8k"`, `"math"`, `"code_generation"`
- 使用场景：一个 Reward 函数处理多种任务

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "gsm8k":
        return gsm8k_reward(solution_str, ground_truth)
    elif data_source == "math":
        return math_reward(solution_str, ground_truth)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")
```

**solution_str:**
- 类型：`str`（已经 detokenize）
- 内容：模型生成的完整响应文本
- 示例：`"Let's solve step by step. First, ... Therefore, the answer is #### 42"`

**ground_truth:**
- 类型：`str`（从数据中提取）
- 内容：正确答案或参考答案
- 来源：数据的 `reward_model` 字段中的参数
- 示例：`"42"`, `"\\frac{1}{2}"`, `"def solution():\n    return 42"`

**extra_info:**
- 类型：`dict` 或 `None`
- 用途：传递额外参数（如配置、元数据）
- 示例：`{"method": "strict", "format_score": 0.1}`

### 1.3 Reward 的数值设计

#### 推荐范围：0-1

**原因：**
1. **标准化**：便于不同 Reward 的对比和组合
2. **数值稳定**：避免梯度爆炸或消失
3. **可解释性**：0=错误，1=正确，中间值=部分正确

#### Binary Reward（二元奖励）

```python
def binary_reward(solution_str, ground_truth):
    """
    只有 0 或 1 两种可能

    优点：简单、明确
    缺点：难以学习、样本效率低
    """
    answer = extract_answer(solution_str)
    return 1.0 if answer == ground_truth else 0.0
```

**适用场景：**
- 明确的对错判断（如数学题）
- 数据集足够大
- 不需要细粒度反馈

#### Graded Reward（分级奖励）

```python
def graded_reward(solution_str, ground_truth):
    """
    多个离散的奖励等级

    优点：提供中间反馈
    缺点：需要人工定义等级
    """
    answer = extract_answer(solution_str)

    if answer == ground_truth:
        return 1.0  # 完全正确
    elif has_correct_format(solution_str):
        return 0.3  # 格式正确
    elif has_reasoning_steps(solution_str):
        return 0.1  # 有推理过程
    else:
        return 0.0  # 完全错误
```

**适用场景：**
- 有明确的评分标准
- 需要鼓励部分正确
- 数据集较小，需要密集反馈

#### Continuous Reward（连续奖励）

```python
def continuous_reward(solution_str, ground_truth):
    """
    0-1 之间的连续值

    优点：最大化信息利用
    缺点：可能难以设计合理的连续函数
    """
    # 方法 1：相似度
    similarity = compute_similarity(solution_str, ground_truth)
    return similarity  # 0-1 之间

    # 方法 2：归一化指标
    metric = compute_metric(solution_str)
    max_metric = 100
    return min(1.0, metric / max_metric)

    # 方法 3：多指标加权
    accuracy = compute_accuracy(solution_str, ground_truth)
    length_score = compute_length_score(solution_str)
    format_score = compute_format_score(solution_str)

    return 0.6 * accuracy + 0.2 * length_score + 0.2 * format_score
```

**适用场景：**
- 文本生成质量评估
- 多目标优化
- 需要细粒度反馈

---

## 2. RewardManager 调用流程

### 2.1 RewardManager 架构

**位置：** `verl/trainer/ppo/reward.py`

```python
class RewardManager:
    """
    负责计算 batch 中所有响应的奖励分数

    核心方法：
    - __call__(batch): 计算整个 batch 的 reward
    - _call_single(data_item): 计算单个样本的 reward
    """

    def __init__(self, tokenizer, num_examine: int = 0):
        """
        Args:
            tokenizer: 用于 detokenize
            num_examine: 打印前 N 个样本（调试用）
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, batch: DataProto) -> DataProto:
        """
        计算 batch 中所有样本的 reward

        流程：
        1. 提取 responses 和 reward_model 配置
        2. Detokenize responses
        3. 调用 compute_score_fn
        4. 返回 token-level rewards
        """
        pass
```

### 2.2 完整调用流程

```python
# ==================== 在 RayPPOTrainer._train_step 中 ====================

# 阶段 1: Rollout - 生成响应
rollout_output = self.actor_rollout_wg.generate_sequences(batch)
# rollout_output.batch 包含:
#   'responses': (bs, response_len) - token IDs
#   'response_mask': (bs, response_len)

# 阶段 2: Reward - 计算奖励（调用 RewardManager）
rollout_output = self._compute_reward(rollout_output)

# ==================== _compute_reward 实现 ====================

def _compute_reward(self, rollout_output: DataProto):
    # 调用 RewardManager
    rollout_output = self.reward_manager(rollout_output.batch)

    return rollout_output

# ==================== RewardManager.__call__ 实现 ====================

def __call__(self, batch: DataProto) -> DataProto:
    # 1. 提取必要信息
    responses = batch["responses"]           # (bs, response_len)
    reward_models = batch["reward_model"]    # List[dict]
    data_sources = batch["data_source"]      # List[str]

    # 2. Detokenize
    response_strs = self.tokenizer.batch_decode(
        responses,
        skip_special_tokens=True
    )

    # 3. 逐个计算 reward
    scores = []
    for i in range(len(response_strs)):
        reward_config = reward_models[i]
        data_source = data_sources[i]
        solution_str = response_strs[i]

        # 提取 ground_truth
        ground_truth = reward_config.get("ground_truth", "")

        # 获取 compute_score 函数
        compute_fn = self._get_compute_fn(reward_config)

        # 计算分数
        score = compute_fn(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=reward_config.get("extra_info")
        )

        scores.append(score)

    # 4. 转换为 token-level rewards
    token_level_rewards = self._to_token_level(scores, batch)

    # 5. 存回 batch
    batch["token_level_rewards"] = token_level_rewards
    batch["rewards"] = torch.tensor(scores)

    return batch

# ==================== _get_compute_fn 实现 ====================

def _get_compute_fn(self, reward_config):
    style = reward_config["style"]

    if style == "rule":
        # Rule-based: 导入 Python 模块
        module_path = reward_config["module"]
        function_name = reward_config.get("function", "compute_score")

        module = importlib.import_module(module_path)
        compute_fn = getattr(module, function_name)

    elif style == "model":
        # Model-based: 加载 Reward Model
        model_path = reward_config["path"]
        compute_fn = self._load_reward_model(model_path)

    elif style == "sandbox":
        # Sandbox: 代码执行
        compute_fn = self._create_sandbox_fn(reward_config)

    else:
        raise ValueError(f"Unknown style: {style}")

    return compute_fn
```

### 2.3 数据流示意图

```
输入 batch:
{
    "responses": [[101, 2023, 2003, ...], [...]],  # token IDs
    "reward_model": [
        {
            "style": "rule",
            "module": "verl.utils.reward_score.gsm8k",
            "ground_truth": "42"
        },
        ...
    ],
    "data_source": ["gsm8k", ...]
}

    ↓ Detokenize

response_strs: [
    "Let's solve step by step. ... #### 42",
    ...
]

    ↓ For each response

compute_score(
    data_source="gsm8k",
    solution_str="Let's solve step by step. ... #### 42",
    ground_truth="42"
)

    ↓ Extract answer "42"

answer == ground_truth  → score = 1.0

    ↓ Convert to token-level

token_level_rewards: [
    [0, 0, 0, ..., 1.0],  # 只有最后一个 token 有奖励
    ...
]

输出 batch:
{
    ...(原有字段),
    "token_level_rewards": tensor([[0,0,...,1], [...]]),
    "rewards": tensor([1.0, ...])
}
```

---

## 3. 3 种 Reward 类型详解

### 3.1 Rule-based Reward

**定义：** 基于规则和模式匹配的 Reward 函数

**优点：**
- ✅ 实现简单
- ✅ 计算快速
- ✅ 完全可控、可解释
- ✅ 不需要额外模型

**缺点：**
- ❌ 需要人工设计规则
- ❌ 泛化能力有限
- ❌ 规则可能过于严格或宽松

**配置：**
```python
reward_model = {
    "style": "rule",
    "module": "verl.utils.reward_score.gsm8k",  # Python 模块路径
    "function": "compute_score",                 # 函数名（默认 compute_score）
    "ground_truth": "42",                        # 传递给函数的参数
    "method": "strict"                           # extra_info 中的其他参数
}
```

**示例 1：GSM8K Reward**

**位置：** `verl/utils/reward_score/gsm8k.py`

```python
def compute_score(
    data_source,
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.0,
    score=1.0
):
    """
    GSM8K Reward: 提取 #### 后的数字并比较

    Args:
        method: "strict" 或 "flexible"
        format_score: 格式正确但答案错误的分数
        score: 答案正确的分数

    Returns:
        float: 0.0, format_score, 或 score
    """
    # 1. 提取答案
    answer = extract_solution(solution_str, method=method)

    # 2. 判断
    if answer is None:
        # 没有找到答案（格式错误）
        return 0.0
    elif answer == ground_truth:
        # 答案正确
        return score
    else:
        # 格式正确，答案错误
        return format_score


def extract_solution(solution_str, method="strict"):
    """
    从响应中提取答案

    strict: 匹配 "#### number" 格式
    flexible: 提取最后一个数字
    """
    if method == "strict":
        # 严格模式：必须有 "####"
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if solutions:
            # 取最后一个匹配
            return solutions[-1].replace(",", "").replace("$", "")
        else:
            return None

    elif method == "flexible":
        # 宽松模式：提取最后一个数字
        numbers = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        if numbers:
            return numbers[-1].replace(",", "").replace("$", "")
        else:
            return None
```

**使用示例：**

```python
# 测试
solution_1 = "Let's solve step by step. ... Therefore, #### 42"
solution_2 = "The answer is 42."
solution_3 = "I don't know."

print(compute_score("gsm8k", solution_1, "42", method="strict"))
# 输出: 1.0 (格式和答案都正确)

print(compute_score("gsm8k", solution_2, "42", method="strict"))
# 输出: 0.0 (没有 "####", 格式错误)

print(compute_score("gsm8k", solution_2, "42", method="flexible"))
# 输出: 1.0 (flexible 模式提取到 "42")

print(compute_score("gsm8k", solution_3, "42", method="strict"))
# 输出: 0.0 (没有数字)
```

**示例 2：MATH Reward**

**位置：** `verl/utils/reward_score/math_reward.py`

```python
def compute_score(data_source, solution_str, ground_truth):
    """
    MATH Reward: 提取 \\boxed{answer} 并比较

    步骤：
    1. 找到最后一个 \\boxed{...}
    2. 提取其中的内容
    3. 标准化（去除空格、特殊符号等）
    4. 比较
    """
    try:
        # 1. 提取 boxed 内容
        boxed_str = last_boxed_only_string(solution_str)

        if boxed_str is None:
            return 0.0

        # 2. 去除 \boxed{ 和 }
        answer = remove_boxed(boxed_str)

        # 3. 标准化并比较
        if is_equiv(answer, ground_truth):
            return 1.0
        else:
            return 0.0

    except Exception as e:
        print(f"Error in compute_score: {e}")
        return 0.0


def last_boxed_only_string(string):
    """
    提取最后一个 \\boxed{...} 的完整字符串

    示例：
    "The answer is \\boxed{42} and \\boxed{43}"
    → "\\boxed{43}"
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        # 尝试 \\fbox
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    # 找到匹配的右括号
    i = idx
    num_left_braces = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces += 1
        if string[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                return string[idx : i + 1]
        i += 1

    return None


def is_equiv(str1, str2):
    """
    判断两个数学表达式是否等价

    包含大量的标准化操作：
    - 去除空格
    - 标准化分数表示
    - 处理 LaTeX 符号
    - ...
    """
    str1 = strip_string(str1)
    str2 = strip_string(str2)
    return str1 == str2


def strip_string(string):
    """
    标准化字符串

    操作：
    - 去除换行
    - 去除反斜杠
    - 标准化分数（\\frac{a}{b}）
    - 去除单位
    - 去除百分号
    - ...
    """
    # 1. 去除换行
    string = string.replace("\n", "")

    # 2. 去除 \\ 和特殊符号
    string = string.replace("\\\\", "\\")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # 3. 标准化分数
    string = fix_fracs(string)

    # 4. 去除空格
    string = string.replace(" ", "")

    # ... 更多标准化操作

    return string
```

**使用示例：**

```python
solution_1 = "The solution is \\boxed{\\frac{1}{2}}"
solution_2 = "Answer: \\boxed{0.5}"
solution_3 = "I think it's \\boxed{42}"

print(compute_score("math", solution_1, "\\frac{1}{2}"))
# 输出: 1.0

print(compute_score("math", solution_2, "\\frac{1}{2}"))
# 输出: 1.0 (0.5 会被转换为 \\frac{1}{2})

print(compute_score("math", solution_3, "\\frac{1}{2}"))
# 输出: 0.0
```

### 3.2 Model-based Reward

**定义：** 使用训练好的 Reward Model 来评分

**优点：**
- ✅ 自动学习复杂的评分标准
- ✅ 泛化能力强
- ✅ 适合主观评价（如对话质量）

**缺点：**
- ❌ 需要训练 Reward Model
- ❌ 计算慢（需要前向传播）
- ❌ 可能学习到错误的偏好

**配置：**
```python
reward_model = {
    "style": "model",
    "path": "path/to/reward_model",       # Reward Model 路径
    "model_type": "sequence_classification",  # 模型类型
    "device": "cuda:0"                    # 运行设备
}
```

**实现示例：**

```python
class ModelBasedRewardManager:
    def __init__(self, model_path, device="cuda"):
        # 加载 Reward Model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def compute_score(self, data_source, solution_str, ground_truth, extra_info=None):
        """
        使用 Reward Model 评分

        输入：prompt + response
        输出：0-1 之间的分数
        """
        # 1. 构造输入（可能需要 prompt）
        if extra_info and "prompt" in extra_info:
            text = extra_info["prompt"] + "\n" + solution_str
        else:
            text = solution_str

        # 2. Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # 3. 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, 2) for binary classification

        # 4. 计算分数
        # 假设 label 0 = bad, label 1 = good
        probs = torch.softmax(logits, dim=-1)
        score = probs[0, 1].item()  # 取"good"的概率

        return score
```

**使用示例：**

```python
# 创建 Reward Manager
reward_mgr = ModelBasedRewardManager("OpenAssistant/reward-model-deberta-v3-large")

# 计算分数
score = reward_mgr.compute_score(
    data_source="dialog",
    solution_str="Thank you for your question! The answer is 42.",
    ground_truth="",
    extra_info={"prompt": "What is the meaning of life?"}
)

print(f"Score: {score:.4f}")
# 输出: Score: 0.8523 (示例)
```

### 3.3 Sandbox Reward（代码执行）

**定义：** 在沙箱环境中执行代码，根据测试结果评分

**优点：**
- ✅ 准确（实际执行）
- ✅ 适合代码生成任务
- ✅ 可以测试功能正确性

**缺点：**
- ❌ 计算慢
- ❌ 需要安全隔离（防止恶意代码）
- ❌ 可能有超时、错误等问题

**配置：**
```python
reward_model = {
    "style": "sandbox",
    "language": "python",              # 执行语言
    "timeout": 5,                      # 超时时间（秒）
    "test_cases": [                    # 测试用例
        {"input": [1, 2], "expected": 3},
        {"input": [5, 3], "expected": 8}
    ]
}
```

**实现示例：**

```python
import subprocess
import tempfile
import os

def sandbox_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    在沙箱中执行代码并评分

    流程：
    1. 提取代码
    2. 写入临时文件
    3. 在隔离环境中执行
    4. 检查测试用例
    5. 返回通过率
    """
    # 1. 提取代码
    code = extract_code(solution_str)

    # 2. 获取测试用例
    test_cases = extra_info.get("test_cases", [])

    # 3. 执行测试
    passed = 0
    for test in test_cases:
        try:
            result = execute_code_safely(
                code,
                test["input"],
                timeout=extra_info.get("timeout", 5)
            )

            if result == test["expected"]:
                passed += 1

        except Exception as e:
            # 执行失败（超时、错误等）
            continue

    # 4. 计算通过率
    if len(test_cases) == 0:
        return 0.0

    pass_rate = passed / len(test_cases)
    return pass_rate


def extract_code(solution_str):
    """
    从响应中提取代码块

    支持格式：
    - ```python ... ```
    - ```\n...\n```
    """
    # 匹配代码块
    pattern = r"```(?:python)?\n(.*?)\n```"
    matches = re.findall(pattern, solution_str, re.DOTALL)

    if matches:
        return matches[0]
    else:
        # 假设整个响应就是代码
        return solution_str


def execute_code_safely(code, inputs, timeout=5):
    """
    在隔离环境中安全执行代码

    使用 subprocess + timeout 隔离
    """
    # 1. 创建临时文件
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as f:
        # 写入代码
        f.write(code)
        f.write("\n\n")
        # 写入测试
        f.write(f"result = solution({inputs})\n")
        f.write("print(result)\n")
        temp_file = f.name

    try:
        # 2. 执行
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # 3. 解析输出
        output = result.stdout.strip()
        return eval(output)  # 注意：不安全！生产环境需要更安全的解析

    except subprocess.TimeoutExpired:
        raise TimeoutError("Code execution timeout")

    except Exception as e:
        raise RuntimeError(f"Code execution failed: {e}")

    finally:
        # 4. 清理
        os.remove(temp_file)
```

**使用示例：**

```python
solution = """
```python
def solution(a, b):
    return a + b
```
"""

test_cases = [
    {"input": [1, 2], "expected": 3},
    {"input": [5, 3], "expected": 8},
    {"input": [-1, 1], "expected": 0}
]

score = sandbox_compute_score(
    data_source="code",
    solution_str=solution,
    ground_truth="",
    extra_info={"test_cases": test_cases, "timeout": 5}
)

print(f"Pass rate: {score:.2%}")
# 输出: Pass rate: 100.00% (所有测试通过)
```

---

## 4. 实战示例 1-5：基础 Reward

### 示例 1：长度奖励

**目标：** 鼓励生成特定长度的响应

```python
def length_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    长度奖励：鼓励在目标长度附近的响应

    Args:
        extra_info: {"target_length": 100, "tolerance": 20}
    """
    target = extra_info.get("target_length", 100) if extra_info else 100
    tolerance = extra_info.get("tolerance", 20) if extra_info else 20

    actual = len(solution_str)

    # 在 [target - tolerance, target + tolerance] 范围内得满分
    if abs(actual - target) <= tolerance:
        return 1.0
    else:
        # 超出范围，线性惩罚
        penalty = abs(actual - target) / target
        return max(0.0, 1.0 - penalty)
```

**测试：**
```python
assert length_reward("", "x" * 90, "", {"target_length": 100, "tolerance": 20}) == 1.0
assert length_reward("", "x" * 150, "", {"target_length": 100, "tolerance": 20}) < 1.0
```

### 示例 2：格式检查奖励

**目标：** 检查响应是否包含必需的格式元素

```python
def format_check_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    格式检查奖励

    检查项：
    - 是否有标题
    - 是否有推理步骤
    - 是否有最终答案标记
    """
    score = 0.0

    # 检查 1：有"Let's solve"或类似开头
    if any(phrase in solution_str.lower() for phrase in [
        "let's solve",
        "let us solve",
        "to solve this"
    ]):
        score += 0.3

    # 检查 2：有步骤标记（"Step 1", "First", etc.）
    if any(phrase in solution_str for phrase in [
        "Step 1",
        "First,",
        "1.",
        "Firstly,"
    ]):
        score += 0.3

    # 检查 3：有最终答案标记
    if "####" in solution_str or "Therefore" in solution_str:
        score += 0.4

    return score
```

### 示例 3：关键词奖励

**目标：** 鼓励包含特定关键词

```python
def keyword_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    关键词奖励

    Args:
        extra_info: {"keywords": ["important", "key phrase"], "weights": [0.6, 0.4]}
    """
    keywords = extra_info.get("keywords", []) if extra_info else []
    weights = extra_info.get("weights", [1.0 / len(keywords)] * len(keywords)) if extra_info else []

    if len(keywords) == 0:
        return 0.0

    # 确保 weights 长度匹配
    if len(weights) != len(keywords):
        weights = [1.0 / len(keywords)] * len(keywords)

    score = 0.0
    for keyword, weight in zip(keywords, weights):
        if keyword.lower() in solution_str.lower():
            score += weight

    return min(1.0, score)  # Cap at 1.0
```

**测试：**
```python
extra_info = {
    "keywords": ["reasoning", "step by step"],
    "weights": [0.6, 0.4]
}

text = "Let me explain with step by step reasoning."
score = keyword_reward("", text, "", extra_info)
print(score)  # 1.0 (both keywords present)
```

### 示例 4：禁词惩罚

**目标：** 惩罚包含禁用词的响应

```python
def forbidden_word_penalty(data_source, solution_str, ground_truth, extra_info=None):
    """
    禁词惩罚

    Args:
        extra_info: {"forbidden": ["bad word", "inappropriate"], "penalty_per_word": 0.2}
    """
    forbidden = extra_info.get("forbidden", []) if extra_info else []
    penalty_per_word = extra_info.get("penalty_per_word", 0.2) if extra_info else 0.2

    solution_lower = solution_str.lower()

    penalty = 0.0
    for word in forbidden:
        if word.lower() in solution_lower:
            penalty += penalty_per_word

    return max(0.0, 1.0 - penalty)
```

### 示例 5：组合奖励

**目标：** 结合多个 Reward 函数

```python
def combined_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    组合奖励：准确性 + 格式 + 长度

    权重：
    - 准确性：60%
    - 格式：20%
    - 长度：20%
    """
    # 1. 准确性（Binary）
    answer = extract_answer(solution_str)
    accuracy = 1.0 if answer == ground_truth else 0.0

    # 2. 格式
    format_score = format_check_reward(data_source, solution_str, ground_truth)

    # 3. 长度
    length_score = length_reward(
        data_source,
        solution_str,
        ground_truth,
        {"target_length": 200, "tolerance": 50}
    )

    # 4. 加权组合
    total_score = (
        0.6 * accuracy +
        0.2 * format_score +
        0.2 * length_score
    )

    return total_score
```

**测试：**
```python
# 完全正确，格式好，长度合适
solution_good = "Let's solve step by step. ... Therefore, #### 42"
print(combined_reward("gsm8k", solution_good, "42"))  # 接近 1.0

# 答案错误，但格式和长度好
solution_wrong = "Step by step. ... Therefore, #### 43"
print(combined_reward("gsm8k", solution_wrong, "42"))  # 约 0.4

# 答案正确，但格式差
solution_bad_format = "42"
print(combined_reward("gsm8k", solution_bad_format, "42"))  # 约 0.6
```

---

*（由于篇幅限制，继续在下一部分）*
