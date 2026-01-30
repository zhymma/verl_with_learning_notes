# verl 实战学习指南

本文档为 verl 项目的**应用层面**学习路线，专注于数据准备、RL 算法使用和 Agent 训练，跳过底层分布式实现细节。

配合官方文档 https://verl.readthedocs.io/en/latest/index.html 使用。

---

## 目录

- [学习目标](#学习目标)
- [快速上手（1天）](#快速上手1天)
- [数据准备（1-2天）](#数据准备1-2天)
- [RL 算法实战（2-3天）](#rl-算法实战2-3天)
- [Agent RL 训练（3-5天）](#agent-rl-训练3-5天)
- [进阶技巧](#进阶技巧)
- [常见问题](#常见问题)

---

## 学习目标

完成本指南后，你将能够：

✅ 准备和处理 RL 训练数据
✅ 使用不同的 RL 算法（PPO、GRPO、RLOO 等）进行训练
✅ 设计和实现自定义的 Reward 函数
✅ 训练多轮对话和工具调用的 Agent
✅ 调优训练参数获得更好效果

---

## 🎯 学习层次说明

本指南分为两个层次：

### 应用层（第 1-5 节）
- 快速上手、数据准备、算法使用、参数调优
- 面向用户的实战操作
- 不需要了解底层实现

### 原理层（第 6-7 节）⭐ 新增
- 训练流程深度解析（RayPPOTrainer）
- Reward 系统架构深度解析
- 多文件协作原理
- 面向想深入理解代码的开发者

---

## 快速上手（1天）

### 目标
跑通第一个示例，建立直观感觉。

### 1.1 安装环境

```bash
# 安装 verl + vLLM
pip install -e .[test,vllm]

# 或者安装 SGLang（推荐用于 Agent）
pip install -e .[test,sglang]
```

### 1.2 下载模型和数据

```bash
# 下载小模型（7B，适合入门）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/Qwen2.5-7B-Instruct

# 准备 GSM8K 数据
python examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

### 1.3 运行第一个训练

```bash
# GRPO 训练（最简单的 RL 算法）
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['~/data/gsm8k/train.parquet']" \
    data.val_files="['~/data/gsm8k/test.parquet']" \
    actor_rollout_ref.model.path=~/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.rollout.name=vllm \
    trainer.total_epochs=3 \
    trainer.logger='["tensorboard"]'
```

**观察输出指标：**
- `reward/mean` - 平均奖励，应该逐步上升
- `accuracy` - 准确率（对于 GSM8K）
- `response_length/mean` - 平均响应长度

### 1.4 查看训练曲线

```bash
tensorboard --logdir=./outputs
# 浏览器打开 http://localhost:6006
```

---

## 数据准备（1-2天）

### 2.1 理解数据格式

**官方文档：** https://verl.readthedocs.io/en/latest/preparation/prepare_data.html

verl 使用 **Parquet** 格式，必须包含的字段：

```python
{
    "data_source": "gsm8k",           # 数据来源标识
    "prompt": "问题内容...",          # 必需：输入提示
    "ability": "math",                # 可选：能力类型
    "reward_model": {                 # 可选：用于 reward 计算
        "ground_truth": "42",         # 标准答案
        "style": "short",             # 答案风格
    }
}
```

### 2.2 查看内置数据集示例

```bash
# 查看数据预处理脚本
ls examples/data_preprocess/

# 常用数据集
gsm8k.py                  # GSM8K 数学题
math_dataset.py           # MATH 数据集
geo3k.py                  # 几何题
gsm8k_multiturn_w_tool.py # GSM8K + 工具调用
```

### 2.3 准备自己的数据集

#### 单轮数据

```python
# my_data_prep.py
import pandas as pd

data = [
    {
        "data_source": "my_task",
        "prompt": "写一个快速排序算法",
        "reward_model": {
            "ground_truth": "def quick_sort(arr): ...",
            "test_cases": ["test1", "test2"]
        }
    },
    # ... 更多数据
]

df = pd.DataFrame(data)
df.to_parquet("my_data/train.parquet")
print(f"Saved {len(df)} samples")
```

#### 多轮对话数据

```python
# 多轮对话格式
data = [
    {
        "data_source": "multiturn",
        "prompt": [
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "第一轮回答"},
            {"role": "user", "content": "第二轮问题"}
        ],
        "reward_model": {
            "ground_truth": "最终答案"
        }
    }
]
```

### 2.4 多模态数据（VLM）

```python
# 视觉语言模型数据
data = [
    {
        "data_source": "vqa",
        "prompt": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "图片中有什么？"}
        ],
        "reward_model": {
            "ground_truth": "一只猫"
        }
    }
]
```

### 2.5 数据质量检查

```python
# check_data.py
import pandas as pd

df = pd.read_parquet("my_data/train.parquet")

print(f"总样本数: {len(df)}")
print(f"字段: {df.columns.tolist()}")
print(f"\n前3条样本:")
print(df.head(3))

# 检查 prompt 长度分布
df['prompt_len'] = df['prompt'].str.len()
print(f"\nPrompt 长度统计:")
print(df['prompt_len'].describe())
```

---

## RL 算法实战（2-3天）

### 3.1 算法对比

**官方文档：** https://verl.readthedocs.io/en/latest/algo/algo_intro.html

| 算法 | 适用场景 | 特点 | 配置 |
|------|---------|------|------|
| **GRPO** | 入门首选 | 简单稳定，不需要 Critic | `algorithm.adv_estimator=grpo` |
| **PPO** | 通用 | 经典算法，需要 Critic | `algorithm.adv_estimator=gae` |
| **RLOO** | Best-of-N | 在 N 个候选中选最优 | `algorithm.adv_estimator=rloo` |
| **ReMax** | 高质量数据 | 最大化 reward 期望 | `algorithm.adv_estimator=remax` |
| **REINFORCE++** | 简单任务 | 改进版 REINFORCE | `algorithm.adv_estimator=reinforce_plus_plus` |

### 3.2 GRPO 训练（推荐入门）

**官方文档：** https://verl.readthedocs.io/en/latest/algo/grpo.html

```bash
# examples/grpo_trainer/run_qwen2-7b.sh
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.group_size=4 \              # GRPO 组大小
    data.train_files="['~/data/gsm8k/train.parquet']" \
    actor_rollout_ref.model.path=~/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.name=vllm \
    trainer.total_epochs=10
```

**核心参数：**
- `algorithm.group_size`: 每个 prompt 生成几个响应（通常 4-8）
- `algorithm.kl_penalty`: KL 惩罚系数（默认 0.001）

### 3.3 PPO 训练

**官方文档：** https://verl.readthedocs.io/en/latest/algo/ppo.html

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \          # 使用 GAE
    algorithm.gamma=1.0 \                  # 折扣因子
    algorithm.lam=0.95 \                   # GAE lambda
    critic.enable=true \                   # 启用 Critic
    critic.optim.lr=1e-5 \                 # Critic 学习率
    data.train_files="['~/data/gsm8k/train.parquet']" \
    actor_rollout_ref.model.path=~/models/Qwen2.5-7B-Instruct
```

**与 GRPO 的区别：**
- PPO 需要额外的 **Critic 模型**（预测 value）
- GRPO 更简单，只需要 Actor 模型
- PPO 理论上更稳定，但 GRPO 实践中表现也很好

### 3.4 自定义 Reward 函数

**官方文档：** https://verl.readthedocs.io/en/latest/preparation/reward_function.html

#### 方式一：Rule-based Reward

```python
# my_reward.py
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    自定义 reward 函数

    Args:
        data_source: 数据来源（来自 Parquet 的 data_source 字段）
        solution_str: 模型生成的响应
        ground_truth: 标准答案（来自 reward_model.ground_truth）
        extra_info: 额外信息（来自 reward_model 的其他字段）

    Returns:
        float: reward 分数，通常 0-1 之间
    """
    # 示例：精确匹配
    if solution_str.strip().lower() == ground_truth.strip().lower():
        return 1.0

    # 示例：包含关键词
    if ground_truth.lower() in solution_str.lower():
        return 0.5

    return 0.0
```

```bash
# 使用自定义 reward
python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=/path/to/my_reward.py \
    custom_reward_function.name=compute_score \
    ...
```

#### 方式二：使用内置 Reward

```python
# 内置 reward 在 verl/utils/reward_score/
from verl.utils.reward_score import gsm8k, math_reward, geo3k

# GSM8K: 提取数字答案并比较
# MATH: 数学表达式等价性判断
# Geo3K: 几何题答案判断
```

#### 方式三：Reward Model（模型打分）

```python
# 使用训练好的 Reward Model
reward_model:
  enable: true
  model_path: "Qwen/Qwen2-7B-Reward"
  batch_size: 64
```

### 3.5 算法参数调优

**关键参数速查表：**

```yaml
# 学习率（最重要）
actor_rollout_ref.actor.optim.lr: 1e-6       # 太大容易崩，太小收敛慢
critic.optim.lr: 1e-5                         # Critic 学习率通常比 Actor 大

# KL 惩罚（防止偏离原始模型太远）
algorithm.kl_penalty: 0.001                   # GRPO/RLOO 用
algorithm.kl_ctrl.kl_coef: 0.01              # PPO 用

# 训练稳定性
actor_rollout_ref.actor.ppo_epochs: 1        # PPO epoch 数，越大越稳定但越慢
actor_rollout_ref.actor.clip_ratio: 0.2      # PPO clip 范围

# 数据相关
data.train_batch_size: 1024                   # 全局 batch size
algorithm.group_size: 4                       # GRPO 每个 prompt 的候选数
```

**调优建议：**

1. **学习率过大**症状：reward 突然掉到负数，loss 爆炸
   - 解决：降低 10 倍，如 1e-6 → 1e-7

2. **学习率过小**症状：训练很多 epoch 仍无改善
   - 解决：增加 3-5 倍

3. **KL divergence 过大**症状：模型输出变得很奇怪
   - 解决：增加 `kl_penalty`

4. **不收敛**症状：reward 上下波动，不稳定
   - 增加 `train_batch_size`
   - 降低学习率
   - 增加 `ppo_epochs`

### 3.6 实验对比

**创建实验脚本：**

```bash
# experiment.sh
#!/bin/bash

# 实验 1: GRPO baseline
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.project_name=exp_grpo_lr1e6

# 实验 2: GRPO 更大学习率
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    trainer.project_name=exp_grpo_lr5e6

# 实验 3: PPO 对比
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    critic.enable=true \
    trainer.project_name=exp_ppo
```

---

## Agent RL 训练（3-5天）

### 4.1 什么是 Agent RL

**Agent RL** 训练智能体进行**多轮对话**和**工具调用**，与单轮 RL 的区别：

| 维度 | 单轮 RL | Agent RL |
|------|---------|----------|
| 交互 | 一问一答 | 多轮对话 |
| 工具 | 无 | 计算器、搜索、代码执行等 |
| 状态 | 无状态 | 维护对话历史 |
| Reward | 立即反馈 | 最终结果 |

### 4.2 Agent Loop 框架

**官方文档：** https://verl.readthedocs.io/en/latest/advance/agent_loop.html

**核心概念：**

```
用户问题
    ↓
Agent 生成响应
    ↓
解析是否调用工具？
    ├─ 是 → 执行工具 → 将结果加入对话 → 继续生成
    └─ 否 → 返回最终答案 → 计算 Reward
```

### 4.3 准备工具调用数据

```bash
# GSM8K + 计算器工具
python examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_dir ~/data/gsm8k_tool

# Geo3K + 几何工具
python examples/data_preprocess/geo3k_multiturn_w_tool.py \
    --local_dir ~/data/geo3k_tool
```

**数据格式：**

```python
{
    "data_source": "gsm8k",
    "prompt": [
        {"role": "user", "content": "计算 23 * 45 + 67"}
    ],
    "tool_config": {
        "available_tools": ["calculator"],
        "tool_format": "react"  # 或 "function_calling"
    },
    "reward_model": {
        "ground_truth": "1102"
    }
}
```

### 4.4 配置工具

```yaml
# config/tool_config/gsm8k_tool_config.yaml
tools:
  - name: calculator
    description: "执行数学计算"
    parameters:
      expression:
        type: string
        description: "数学表达式，如 '23 * 45 + 67'"

  - name: python
    description: "执行 Python 代码"
    parameters:
      code:
        type: string
        description: "Python 代码"
```

### 4.5 运行 Agent RL 训练

```bash
# 使用 SGLang（推荐用于 Agent）
python3 -m verl.trainer.main_ppo \
    --config-path=examples/sglang_multiturn/config \
    data.train_files="['~/data/gsm8k_tool/train.parquet']" \
    actor_rollout_ref.model.path=~/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.agent_loop=tool_agent_loop \
    tool_config_path=examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml
```

**关键配置：**
- `actor_rollout_ref.rollout.agent_loop`: 指定 agent loop 类型
  - `single_turn_agent_loop`: 单轮
  - `tool_agent_loop`: 工具调用（ReAct）
  - 自定义：`my_custom_agent_loop`

### 4.6 自定义 Agent Loop

**场景：** 实现特殊的工具调用逻辑或多轮交互模式

```python
# my_agent_loop.py
from verl.experimental.agent_loop import AgentLoopBase, AgentLoopOutput

class MyAgentLoop(AgentLoopBase):
    """自定义 Agent Loop"""

    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        """
        执行多轮交互

        Returns:
            AgentLoopOutput: 包含 prompt_ids, response_ids, response_mask
        """
        messages = kwargs.get("messages", [])
        max_turns = kwargs.get("max_turns", 5)

        all_response_ids = []
        all_response_mask = []

        for turn in range(max_turns):
            # 1. 调用 LLM 生成响应
            prompt_text = self._format_messages(messages)
            prompt_ids = self._tokenize(prompt_text)

            response_ids = await self.server_manager.generate(
                request_id=self.request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )

            # 2. 解析响应
            response_text = self._decode(response_ids)
            tool_calls = self._parse_tool_calls(response_text)

            # 3. 执行工具（如果有）
            if tool_calls:
                for tool_call in tool_calls:
                    result = await self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call["id"]
                    })
            else:
                # 没有工具调用，结束
                break

            all_response_ids.append(response_ids)
            all_response_mask.append([1] * len(response_ids))

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=torch.cat(all_response_ids),
            response_mask=torch.cat(all_response_mask),
        )

    def _parse_tool_calls(self, text):
        """解析工具调用"""
        # 你的解析逻辑
        # 例如：匹配 <tool>calculator</tool><args>{"expr": "1+1"}</args>
        pass

    async def _execute_tool(self, tool_call):
        """执行工具"""
        tool_name = tool_call["name"]

        if tool_name == "calculator":
            expr = tool_call["args"]["expression"]
            return str(eval(expr))  # 注意：实际使用需要安全的 eval

        elif tool_name == "python":
            code = tool_call["args"]["code"]
            # 使用沙箱执行
            result = self._safe_exec_python(code)
            return result
```

**注册自定义 Agent Loop：**

```python
# 在训练脚本中
from verl.experimental.agent_loop import register_agent_loop
from my_agent_loop import MyAgentLoop

register_agent_loop("my_agent_loop", MyAgentLoop)
```

### 4.7 Agent 训练技巧

**1. 增加探索**

```yaml
# 使用更高的 temperature
actor_rollout_ref.rollout.temperature: 0.7  # 默认 0.6

# 使用 top_p 采样
actor_rollout_ref.rollout.top_p: 0.9
```

**2. 限制工具调用次数**

```yaml
# 防止无限循环
tool_config:
  max_tool_calls: 5
  timeout: 30  # 秒
```

**3. 分层 Reward**

```python
def compute_agent_reward(data_source, solution_str, ground_truth, extra_info):
    """Agent reward = 任务完成度 + 工具使用效率"""

    # 任务完成度
    task_reward = 1.0 if check_answer(solution_str, ground_truth) else 0.0

    # 工具使用效率（惩罚过多工具调用）
    num_tool_calls = extra_info.get("num_tool_calls", 0)
    efficiency_penalty = -0.1 * max(0, num_tool_calls - 3)

    return task_reward + efficiency_penalty
```

### 4.8 完整 Agent 训练示例

查看官方示例：

```bash
# GSM8K 工具调用
examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh

# 配置文件
examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml
```

---

## 进阶技巧

### 5.1 使用 LoRA 加速训练

**场景：** 模型太大，全量训练显存不够

```yaml
actor_rollout_ref:
  actor:
    lora:
      enable: true
      r: 16                          # LoRA rank
      lora_alpha: 32
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

```bash
# LoRA 训练示例
examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh
```

### 5.2 多数据集混合训练

```yaml
data:
  train_files:
    - "~/data/gsm8k/train.parquet"
    - "~/data/math/train.parquet"
    - "~/data/code/train.parquet"

  # 数据集采样权重
  dataset_weights: [0.5, 0.3, 0.2]
```

### 5.3 Curriculum Learning（课程学习）

**策略：** 从简单任务到困难任务

```python
# prepare_curriculum_data.py
import pandas as pd

# 按难度分级
easy_df = df[df['difficulty'] == 'easy']
medium_df = df[df['difficulty'] == 'medium']
hard_df = df[df['difficulty'] == 'hard']

easy_df.to_parquet("curriculum/stage1_easy.parquet")
medium_df.to_parquet("curriculum/stage2_medium.parquet")
hard_df.to_parquet("curriculum/stage3_hard.parquet")
```

```bash
# 分阶段训练
# Stage 1: 简单任务
python3 -m verl.trainer.main_ppo data.train_files="['curriculum/stage1_easy.parquet']" trainer.total_epochs=5

# Stage 2: 中等任务（从 stage1 checkpoint 继续）
python3 -m verl.trainer.main_ppo data.train_files="['curriculum/stage2_medium.parquet']" actor_rollout_ref.model.path=outputs/stage1/checkpoint-xxx

# Stage 3: 困难任务
python3 -m verl.trainer.main_ppo data.train_files="['curriculum/stage3_hard.parquet']" actor_rollout_ref.model.path=outputs/stage2/checkpoint-xxx
```

### 5.4 在线数据增强

```python
# 在 reward 函数中动态生成新样本
def compute_score_with_augmentation(data_source, solution_str, ground_truth, extra_info):
    score = basic_score(solution_str, ground_truth)

    # 如果答对了，生成相似的难题
    if score > 0.9:
        augmented_prompt = generate_harder_version(extra_info['original_prompt'])
        # 保存到数据集供下一轮训练使用
        save_to_buffer(augmented_prompt)

    return score
```

### 5.5 监控和调试

```bash
# 启用详细日志
export VERL_LOGGING_LEVEL=DEBUG

# 查看生成的样本
trainer.log_generation_samples: true

# 使用 WandB
trainer.logger: '["wandb"]'
trainer.project_name: my_project
trainer.run_name: exp1
```

---

## 常见问题

### Q1: Reward 一直是 0，怎么办？

**可能原因：**
1. Reward 函数写错了
2. 数据格式不对（缺少 `reward_model.ground_truth`）
3. 模型输出格式不符合预期

**调试方法：**
```python
# 在 reward 函数中添加调试输出
def compute_score(data_source, solution_str, ground_truth, extra_info):
    print(f"DEBUG:")
    print(f"  Solution: {solution_str[:100]}")
    print(f"  Ground truth: {ground_truth}")

    score = ...
    print(f"  Score: {score}")
    return score
```

### Q2: 训练过程中 reward 突然下降

**可能原因：**
- 学习率太大，导致策略崩溃
- KL divergence 太大，偏离原始模型太远

**解决方法：**
```yaml
# 降低学习率
actor_rollout_ref.actor.optim.lr: 5e-7  # 从 1e-6 降低

# 增加 KL 惩罚
algorithm.kl_penalty: 0.01  # 从 0.001 增加
```

### Q3: OOM（显存不足）

**解决方法：**
```yaml
# 1. 减小 batch size
data.train_batch_size: 512
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 8

# 2. 减小推理显存占用
actor_rollout_ref.rollout.gpu_memory_utilization: 0.4

# 3. 使用 LoRA
actor_rollout_ref.actor.lora.enable: true

# 4. 开启 gradient checkpointing
actor_rollout_ref.model.enable_gradient_checkpointing: true
```

### Q4: Agent 陷入死循环调用工具

**解决方法：**
```yaml
# 限制工具调用次数
tool_config:
  max_tool_calls: 5

# 在 Agent Loop 中添加超时
sampling_params:
  max_tokens: 2048
  timeout: 30
```

**或者在 Reward 中惩罚：**
```python
def compute_score(...):
    base_score = ...

    # 惩罚过多工具调用
    num_calls = extra_info.get("num_tool_calls", 0)
    if num_calls > 5:
        return base_score - 0.5

    return base_score
```

### Q5: 如何知道训练效果好不好？

**关键指标：**

1. **Reward 趋势**：应该持续上升
2. **Accuracy**：在有标准答案的任务上应该提升
3. **KL divergence**：不应该太大（<10）
4. **Response length**：不应该过短或过长

**对比基线：**
```python
# 评估脚本
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("outputs/checkpoint-xxx")
tokenizer = AutoTokenizer.from_pretrained("outputs/checkpoint-xxx")

# 在测试集上评估
correct = 0
for sample in test_data:
    response = generate(model, tokenizer, sample['prompt'])
    if check_answer(response, sample['ground_truth']):
        correct += 1

accuracy = correct / len(test_data)
print(f"Accuracy: {accuracy:.2%}")
```

---

## 学习检查清单

### 快速上手 ✓
- [ ] 成功安装 verl
- [ ] 跑通 GRPO quickstart
- [ ] 理解训练日志和指标

### 数据准备 ✓
- [ ] 理解 Parquet 数据格式
- [ ] 能够准备单轮对话数据
- [ ] 能够准备多轮对话数据（Agent）
- [ ] 检查数据质量

### RL 算法 ✓
- [ ] 理解 GRPO vs PPO 的区别
- [ ] 训练过至少 3 个不同算法
- [ ] 实现自定义 Reward 函数
- [ ] 能够调优学习率等参数
- [ ] 能够对比不同算法的效果

### Agent RL ✓
- [ ] 理解 Agent Loop 的工作原理
- [ ] 准备工具调用数据
- [ ] 配置工具定义
- [ ] 训练一个工具调用 Agent
- [ ] （可选）实现自定义 Agent Loop

- [ ] 设置监控和调试

---

## 训练流程深度解析（原理层）⭐

> **面向对象**：想深入理解 verl 训练流程的开发者
> **核心文件**：`verl/trainer/ppo/ray_trainer.py` (1741 行)
> **前置知识**：理解 PPO 算法、Ray 分布式框架

### 6.1 RayPPOTrainer 架构概览

**RayPPOTrainer** 是 verl 的核心训练协调器，职责：

```
┌─────────────────────────────────────────────────────────┐
│                   RayPPOTrainer                          │
│                  (单控制器架构)                           │
├─────────────────────────────────────────────────────────┤
│ • 在 Driver 进程上运行训练循环                            │
│ • 通过 Ray RPC 调用分布式 Worker                         │
│ • 执行轻量级计算（Advantage、KL penalty）                 │
│ • 管理 4 类 WorkerGroup：                                │
│   - ActorRollout: 推理生成响应                           │
│   - Critic: 训练价值网络（仅 PPO）                        │
│   - RefPolicy: 计算参考策略 log_prob                     │
│   - RewardModel: 模型打分（可选）                        │
└─────────────────────────────────────────────────────────┘
```

**关键设计原则**：

1. **单控制器（Single Controller）**：所有协调逻辑在 driver 上，worker 只执行计算
2. **混合引擎（Hybrid Engine）**：训练用 FSDP/Megatron，推理用 vLLM/SGLang
3. **异步推理（Async Rollout）**：推理和训练可以并行

---

### 6.2 训练主循环详解

#### fit() 方法流程图

```python
# verl/trainer/ppo/ray_trainer.py: 1349-1741
def fit(self):
    for epoch in range(epochs):
        for batch in train_dataloader:
            # ==================== 第 1 步：生成响应 ====================
            gen_batch = self.actor_rollout_wg.generate_sequences(batch)
            # 返回：responses, log_probs, attention_mask

            # ==================== 第 2 步：计算 Reward ====================
            reward_tensor, extra_info = compute_reward(gen_batch, reward_fn)
            # 返回：token_level_scores [batch_size, seq_len]

            # ==================== 第 3 步：重新计算 Log Prob ====================
            old_log_prob = self._compute_old_log_prob(gen_batch)
            # 为什么重算？需要梯度信息用于 PPO 更新

            # ==================== 第 4 步：参考策略 ====================
            if self.use_reference_policy:
                ref_log_prob = self._compute_ref_log_prob(gen_batch)
                # KL 惩罚需要：KL(π_new || π_ref)

            # ==================== 第 5 步：价值估计（仅 PPO）====================
            if self.use_critic:
                values = self._compute_values(gen_batch)
                # Critic 预测 V(s)，用于 GAE

            # ==================== 第 6 步：Advantage 计算 ====================
            gen_batch = compute_advantage(
                gen_batch,
                adv_estimator="grpo",  # 或 "gae", "rloo" 等
                gamma=1.0,
                lam=0.95,
            )
            # 返回：advantages, returns

            # ==================== 第 7 步：更新 Critic ====================
            if self.use_critic:
                critic_output = self._update_critic(gen_batch)

            # ==================== 第 8 步：更新 Actor ====================
            actor_output = self._update_actor(gen_batch)
            # PPO Loss: clip(ratio * A, ...) - β * KL + α * H
```

---

### 6.3 核心计算方法详解

#### 6.3.1 compute_advantage() - Advantage 估计

**位置**：`verl/trainer/ppo/ray_trainer.py:187-276`

**支持的算法**：

| 算法 | Advantage 公式 | 特点 |
|------|---------------|------|
| **GAE** | `A_t = δ_t + (γλ)δ_{t+1} + ...` <br> `δ_t = r_t + γV(s_{t+1}) - V(s_t)` | 需要 Critic，方差小 |
| **GRPO** | `a_i = (r_i - μ_g) / σ_g` | 组内相对优势，无需 Critic |
| **RLOO** | `a_i = r_i - mean(r_{-i})` | Leave-one-out baseline |
| **REINFORCE++** | `A_t = R_t - b` | 简单折扣回报 |

**示例代码**：

```python
# GAE 实现
if adv_estimator == AdvantageEstimator.GAE:
    # core_algos.compute_gae_advantage_return()
    advantages = []
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        advantages[t] = delta + gamma * lam * advantages[t+1]

    return advantages, returns

# GRPO 实现
elif adv_estimator == AdvantageEstimator.GRPO:
    # 1. 按 uid 分组（同一个 prompt 的多个响应）
    grouped_rewards = group_by_uid(token_level_rewards)

    # 2. 组内归一化
    for group in grouped_rewards:
        mean = group.mean()
        std = group.std()
        advantages = (group - mean) / (std + 1e-8)

    return advantages
```

**关键点**：

- GAE 需要 `values` 输入（来自 Critic），GRPO 不需要
- GRPO 的 `uid` 字段用于分组（`data.non_tensor_batch["uid"]`）
- Advantage 会被 normalize（减均值除方差）

---

#### 6.3.2 apply_kl_penalty() - KL 惩罚

**位置**：`verl/trainer/ppo/ray_trainer.py:127-166`

**目的**：防止新策略偏离参考策略太远

```python
def apply_kl_penalty(data, kl_ctrl, kl_penalty="kl"):
    # 1. 计算 KL 散度
    old_log_prob = data.batch["old_log_probs"]  # π_new
    ref_log_prob = data.batch["ref_log_prob"]   # π_ref

    if kl_penalty == "kl":  # 最常用
        kld = old_log_prob - ref_log_prob
    elif kl_penalty == "mse":
        kld = 0.5 * (old_log_prob - ref_log_prob).square()
    elif kl_penalty == "low_var_kl":  # K3
        ratio = torch.exp(ref_log_prob - old_log_prob)
        kld = ratio - (ref_log_prob - old_log_prob) - 1

    # 2. 应用 KL 惩罚
    beta = kl_ctrl.value  # 动态调整的系数
    token_level_rewards = token_level_scores - beta * kld

    # 3. 更新自适应 KL 控制器
    current_kl = masked_mean(kld, mask=response_mask)
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    # 如果 KL 过大，增加 beta；如果过小，减小 beta

    return token_level_rewards, metrics
```

**KL 控制器（AdaptiveKLController）**：

```python
class AdaptiveKLController:
    def __init__(self, init_kl_coef=0.01, target_kl=6.0, horizon=10000):
        self.value = init_kl_coef  # 初始 β
        self.target = target_kl    # 目标 KL
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        # PID 控制算法
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # 自适应调整 β
```

---

#### 6.3.3 _update_actor() - Actor 更新

**位置**：`verl/trainer/ppo/ray_trainer.py:1283-1317`

```python
def _update_actor(self, batch):
    # 配置 PPO 训练参数
    ppo_mini_batch_size = 256
    ppo_epochs = 1

    # 转换为 TensorDict 格式
    batch_td = batch.to_tensordict()
    batch_td = left_right_2_no_padding(batch_td)  # 转为 no-padding 格式

    # 设置训练元数据
    tu.assign_non_tensor(
        batch_td,
        calculate_entropy=True,           # 计算熵正则
        global_batch_size=ppo_mini_batch_size,
        mini_batch_size=ppo_mini_batch_size,
        epochs=ppo_epochs,
        seed=42,
        dataloader_kwargs={"shuffle": True},
    )

    # RPC 调用 Actor Worker
    actor_output = self.actor_rollout_wg.update_actor(batch_td)
    # Worker 内部会执行 PPO loss 计算和反向传播

    return actor_output
```

**Actor Worker 内部**（在 `verl/workers/fsdp_workers.py`）：

```python
def update_actor(self, batch_td):
    # PPO Loss 计算
    for epoch in range(ppo_epochs):
        for mini_batch in DataLoader(batch_td, batch_size=mini_batch_size):
            # 前向传播
            new_log_probs = model(mini_batch["input_ids"], ...)

            # 计算 ratio
            ratio = torch.exp(new_log_probs - mini_batch["old_log_probs"])

            # PPO Clipped Loss
            advantages = mini_batch["advantages"]
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 熵正则
            entropy = -(new_log_probs * torch.exp(new_log_probs)).sum(-1).mean()
            entropy_loss = -entropy_coeff * entropy

            # 总 Loss
            loss = policy_loss + entropy_loss

            # 反向传播
            loss.backward()
            optimizer.step()

    return {"policy_loss": policy_loss, "entropy": entropy}
```

---

#### 6.3.4 _update_critic() - Critic 更新

**位置**：`verl/trainer/ppo/ray_trainer.py:1319-1347`

```python
def _update_critic(self, batch):
    # Critic 训练参数
    ppo_mini_batch_size = 256
    ppo_epochs = 1

    # 转换格式并 RPC 调用
    batch_td = batch.to_tensordict()
    batch_td = left_right_2_no_padding(batch_td)

    tu.assign_non_tensor(
        batch_td,
        global_batch_size=ppo_mini_batch_size,
        mini_batch_size=ppo_mini_batch_size,
        epochs=ppo_epochs,
    )

    output = self.critic_wg.train_mini_batch(batch_td)
    return output
```

**Critic Worker 内部**：

```python
def train_mini_batch(self, batch_td):
    # Value Function Loss (MSE)
    for epoch in range(ppo_epochs):
        for mini_batch in DataLoader(batch_td, batch_size=mini_batch_size):
            # 前向传播
            predicted_values = critic_model(mini_batch["input_ids"])

            # MSE Loss
            target_returns = mini_batch["returns"]
            value_loss = F.mse_loss(predicted_values, target_returns)

            # 反向传播
            value_loss.backward()
            optimizer.step()

    return {"value_loss": value_loss.item()}
```

---

### 6.4 WorkerGroup 初始化流程

**位置**：`verl/trainer/ppo/ray_trainer.py:788-975`

#### 步骤 1：创建资源池

```python
def init_workers(self):
    # 1. 创建 Ray 资源池
    self.resource_pool_manager.create_resource_pool()
    # 例如：{"global_pool": [8, 8]} → 2 nodes, 8 GPUs each
```

**ResourcePoolManager**：

```python
@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]
    # 例如：{"global_pool": [8, 8], "rollout_pool": [4]}

    def create_resource_pool(self):
        for pool_name, gpus_per_node in self.resource_pool_spec.items():
            pool = RayResourcePool(
                process_on_nodes=gpus_per_node,
                use_gpu=True,
                max_colocate_count=3,  # 最多 3 个 WorkerGroup 共享节点
            )
            self.resource_pool_dict[pool_name] = pool
```

#### 步骤 2：创建 WorkerGroup

```python
# 2.1 ActorRollout WorkerGroup
actor_rollout_cls = RayClassWithInitArgs(
    cls=FSDPActorRolloutRefWorker,  # 或 MegatronWorker
    config=self.config.actor_rollout_ref,
    role="ActorRolloutRef",
)

self.actor_rollout_wg = RayWorkerGroup(
    resource_pool=actor_rollout_resource_pool,
    ray_cls_with_init=actor_rollout_cls,
    num_workers=8,  # 8 个 worker，每个管理 1 张 GPU
)

# 2.2 Critic WorkerGroup（仅 PPO）
if self.use_critic:
    critic_cls = RayClassWithInitArgs(
        cls=FSDPCriticWorker,
        config=self.config.critic,
    )

    self.critic_wg = RayWorkerGroup(
        resource_pool=critic_resource_pool,
        ray_cls_with_init=critic_cls,
        num_workers=8,
    )

# 2.3 RefPolicy WorkerGroup
if self.use_reference_policy:
    ref_policy_cls = RayClassWithInitArgs(
        cls=FSDPRefPolicyWorker,
        config=self.config.actor_rollout_ref,
    )

    self.ref_policy_wg = RayWorkerGroup(...)

# 2.4 RewardModel WorkerGroup（可选）
if self.use_rm:
    rm_cls = RayClassWithInitArgs(
        cls=RewardModelWorker,
        config=self.config.reward_model,
    )

    self.rm_wg = RayWorkerGroup(...)
```

#### 步骤 3：初始化 Workers

```python
# 3. 调用各 WorkerGroup 的初始化方法
self.actor_rollout_wg.init_model()
# 在每个 worker 上加载模型权重

if self.use_critic:
    self.critic_wg.init_model()

if self.use_reference_policy:
    self.ref_policy_wg.init_model()

# 4. 初始化 Rollout Manager
self.async_rollout_manager = AgentLoopManager(
    config=self.config,
    worker_group=self.actor_rollout_wg,
    rollout_resource_pool=actor_rollout_resource_pool,
    rm_resource_pool=rm_resource_pool,
)
```

---

### 6.5 数据流和 DataProto

**DataProto** 是训练数据在 pipeline 中的标准格式：

```python
@dataclass
class DataProto:
    batch: Dict[str, torch.Tensor]      # Tensor 数据
    non_tensor_batch: Dict[str, Any]    # 非 Tensor 数据（元信息）
```

**完整数据流**：

```
初始 Batch（从 DataLoader）
├── batch:
│   ├── prompts: [bs, prompt_len]
│   └── attention_mask: [bs, total_len]
└── non_tensor_batch:
    ├── uid: [bs]              # 分组 ID
    ├── data_source: [bs]      # "gsm8k"
    └── reward_model:
        └── ground_truth: [bs]

↓ 生成响应后

├── batch:
│   ├── prompts: [bs, prompt_len]
│   ├── responses: [bs, response_len]
│   ├── log_probs: [bs, response_len]  # 生成时的 log prob
│   └── attention_mask: [bs, total_len]
└── non_tensor_batch: ...

↓ 计算 Reward 后

├── batch:
│   ├── ...
│   └── token_level_scores: [bs, response_len]  # Reward 在最后一个 token
└── non_tensor_batch: ...

↓ 重新计算 Log Prob 后

├── batch:
│   ├── ...
│   ├── old_log_probs: [bs, response_len]  # 用于 PPO ratio
│   └── entropys: [bs, response_len]
└── ...

↓ 参考策略后

├── batch:
│   ├── ...
│   └── ref_log_prob: [bs, response_len]  # 用于 KL 惩罚
└── ...

↓ 价值估计后

├── batch:
│   ├── ...
│   └── values: [bs, response_len]  # 用于 GAE
└── ...

↓ Advantage 计算后

├── batch:
│   ├── ...
│   ├── advantages: [bs, response_len]
│   └── returns: [bs, response_len]
└── ...

↓ 用于 Actor/Critic 训练
```

---

### 6.6 Checkpoint 管理

#### 保存 Checkpoint

```python
def _save_checkpoint(self):
    checkpoint_dir = f"outputs/global_step_{self.global_steps}"

    # 1. 保存 Actor
    self.actor_rollout_wg.save_checkpoint(
        local_path=f"{checkpoint_dir}/actor",
        global_steps=self.global_steps,
    )

    # 2. 保存 Critic
    if self.use_critic:
        self.critic_wg.save_checkpoint(
            local_path=f"{checkpoint_dir}/critic",
            global_steps=self.global_steps,
        )

    # 3. 保存 DataLoader 状态（用于断点续训）
    dataloader_state = self.train_dataloader.state_dict()
    torch.save(dataloader_state, f"{checkpoint_dir}/dataloader.pt")
```

#### 加载 Checkpoint

```python
def _load_checkpoint(self):
    # 1. 找到最新的 checkpoint
    latest_ckpt = find_latest_ckpt_path("outputs/")
    global_step = int(latest_ckpt.split("global_step_")[-1])

    # 2. 加载 Actor
    self.actor_rollout_wg.load_checkpoint(f"{latest_ckpt}/actor")

    # 3. 加载 Critic
    if self.use_critic:
        self.critic_wg.load_checkpoint(f"{latest_ckpt}/critic")

    # 4. 恢复 DataLoader 状态
    dataloader_state = torch.load(f"{latest_ckpt}/dataloader.pt")
    self.train_dataloader.load_state_dict(dataloader_state)

    return global_step
```

---

### 6.7 实战案例：追踪一个 Batch

假设我们训练 GSM8K，batch_size=4，每个 prompt 生成 2 个响应（group_size=2）。

```python
# 初始数据
batch = {
    "prompts": [[101, 2023, 2003, ...], ...],  # 4 个 prompts
    "uid": [0, 0, 1, 1],  # 前 2 个是 prompt 0 的，后 2 个是 prompt 1 的
    "data_source": ["gsm8k", "gsm8k", "gsm8k", "gsm8k"],
    "reward_model": {
        "ground_truth": ["42", "42", "100", "100"],
    }
}

# Step 1: 生成响应
gen_batch = actor_rollout_wg.generate_sequences(batch)
# gen_batch["responses"] = [[5, 42, 102], [5, 40, 102], [5, 100, 102], [5, 99, 102]]
#                            ↑ 正确      ↑ 错误         ↑ 正确          ↑ 错误

# Step 2: 计算 Reward
reward_tensor = compute_reward(gen_batch)
# reward_tensor = [[0, 0, 1.0], [0, 0, 0.0], [0, 0, 1.0], [0, 0, 0.0]]
#                  ↑ 最后一个 token 有 reward

# Step 3: 重新计算 Log Prob
old_log_prob = actor_rollout_wg.compute_log_prob(gen_batch)
# old_log_prob = [[-2.3, -1.5, -0.8], ...]  # 每个 token 的 log prob

# Step 4: 参考策略
ref_log_prob = ref_policy_wg.compute_ref_log_prob(gen_batch)

# Step 5: KL 惩罚
token_level_rewards = reward_tensor - beta * (old_log_prob - ref_log_prob)

# Step 6: Advantage（GRPO）
# 按 uid 分组
group_0 = [token_level_rewards[0], token_level_rewards[1]]  # uid=0
group_1 = [token_level_rewards[2], token_level_rewards[3]]  # uid=1

# 组内归一化
advantages[0] = (group_0[0] - mean(group_0)) / std(group_0)  # 正值
advantages[1] = (group_0[1] - mean(group_0)) / std(group_0)  # 负值
advantages[2] = (group_1[0] - mean(group_1)) / std(group_1)  # 正值
advantages[3] = (group_1[1] - mean(group_1)) / std(group_1)  # 负值

# Step 7: 更新 Actor
# PPO 会鼓励 sample 0 和 2（正 advantage），抑制 sample 1 和 3（负 advantage）
```

---

### 6.8 小结

**RayPPOTrainer 核心流程**：

1. **生成** → 2. **Reward** → 3. **Old Log Prob** → 4. **Ref Log Prob** → 5. **Values**（PPO）→ 6. **Advantage** → 7. **Critic 更新** → 8. **Actor 更新**

**关键文件**：
- `verl/trainer/ppo/ray_trainer.py` - 主训练循环
- `verl/trainer/ppo/core_algos.py` - Advantage 算法实现
- `verl/workers/fsdp_workers.py` - FSDP Worker 实现
- `verl/single_controller/base.py` - RayWorkerGroup 基类

**进一步学习**：
- 阅读 `core_algos.py` 了解各种 Advantage 算法的数学细节
- 查看 `fsdp_workers.py` 了解 Worker 内部的模型管理
- 研究 `ray_resource_pool.py` 了解 Ray 资源分配策略

---

## Reward 系统架构深度解析（原理层）⭐

> **面向对象**：想自定义 Reward 函数或理解 Reward 计算流程的开发者
> **核心文件**：`verl/workers/reward_manager/`, `verl/utils/reward_score/`
> **前置知识**：理解 RL 中 Reward 的作用

### 7.1 Reward 系统架构概览

verl 的 Reward 系统采用**插件化架构**，支持两大类：

```
┌─────────────────────────────────────────────────────────┐
│                 Reward 计算架构                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐         ┌──────────────────┐    │
│  │  Rule-based      │         │  Model-based     │    │
│  │  (函数打分)       │         │  (模型打分)       │    │
│  └─────────┬────────┘         └────────┬─────────┘    │
│            │                           │              │
│            └───────────┬───────────────┘              │
│                        │                              │
│              ┌─────────▼──────────┐                   │
│              │  RewardManager     │                   │
│              │  (抽象基类)         │                   │
│              └─────────┬──────────┘                   │
│                        │                              │
│        ┌───────────────┼───────────────┐             │
│        │               │               │             │
│  ┌─────▼─────┐  ┌──────▼─────┐  ┌─────▼─────┐       │
│  │  Naive    │  │   Batch    │  │   Prime   │       │
│  │ (逐个打分) │  │ (批量打分)  │  │ (并行打分) │       │
│  └───────────┘  └────────────┘  └───────────┘       │
│                                                       │
└───────────────────────────────────────────────────────┘
```

**4 种 RewardManager**：

| RewardManager | 处理方式 | 适用场景 | 并发度 |
|---------------|---------|---------|-------|
| **Naive** | 逐个样本 | 调试、简单规则 | 1 |
| **Batch** | 批量向量化 | 大 batch、简单规则 | 1 |
| **Prime** | 并行异步 | 代码执行、沙箱 | 64 进程 |
| **DAPO** | 逐个 + 长度惩罚 | DAPO 训练 | 1 |

---

### 7.2 抽象基类：AbstractRewardManager

**位置**：`verl/workers/reward_manager/abstract.py`

```python
class AbstractRewardManager(ABC):
    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,           # 打印前 N 个样本用于调试
        compute_score: RawRewardFn | None,  # 自定义 reward 函数
        reward_fn_key: str = "data_source",  # 用哪个字段路由 reward 函数
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    @abstractmethod
    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """
        计算 Reward

        输入：
            data: DataProto 包含 responses, prompts, 元信息

        输出：
            reward_tensor: [batch_size, seq_len]
            reward 放在最后一个有效 token：reward_tensor[i, valid_len-1]
        """
        pass
```

**关键设计**：

1. **Reward 位置**：只在响应的最后一个 token 设置 reward，其他位置为 0
2. **解码响应**：需要 `tokenizer.decode()` 将 token IDs 转为文本
3. **元信息提取**：从 `data.non_tensor_batch` 获取 `ground_truth`, `data_source`

---

### 7.3 NaiveRewardManager - 逐个打分

**位置**：`verl/workers/reward_manager/naive.py`

```python
@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(
            data.batch["responses"],
            dtype=torch.float32
        )
        reward_extra_info = {}

        # 逐个样本处理
        for i in range(len(data)):
            data_item = data[i]  # 单个样本的 DataProto

            # 1. 提取 prompt 和 response
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]

            prompt_str = self.tokenizer.decode(
                prompt_ids,
                skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True
            )

            # 2. 提取元信息
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            # 3. 调用 compute_score
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            # 4. 将 reward 放在最后一个有效 token
            prompt_length = data_item.batch["attention_mask"][:len(prompt_ids)].sum()
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            reward_tensor[i, valid_response_length - 1] = score

            # 5. 打印前几个样本（调试用）
            if i < self.num_examine:
                print(f"[Reward Debug {i}]")
                print(f"  Prompt: {prompt_str[:100]}...")
                print(f"  Response: {response_str}")
                print(f"  Ground Truth: {ground_truth}")
                print(f"  Score: {score}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
```

**特点**：
- 逐个处理，适合调试
- 打印前 N 个样本的详细信息
- 简单直观

---

### 7.4 BatchRewardManager - 批量打分

**位置**：`verl/workers/reward_manager/batch.py`

```python
@register("batch")
class BatchRewardManager(AbstractRewardManager):
    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(
            data.batch["responses"],
            dtype=torch.float32
        )

        # 1. 批量解码所有响应
        responses_str = []
        ground_truths = []
        data_sources = []
        extra_infos = []
        valid_response_lengths = []

        for i in range(len(data)):
            data_item = data[i]

            # 解码响应
            response_ids = data_item.batch["responses"]
            valid_len = data_item.batch["attention_mask"][prompt_len:].sum()
            response_str = self.tokenizer.decode(
                response_ids[:valid_len],
                skip_special_tokens=True
            )
            responses_str.append(response_str)

            # 提取元信息
            ground_truths.append(
                data_item.non_tensor_batch["reward_model"]["ground_truth"]
            )
            data_sources.append(
                data_item.non_tensor_batch[self.reward_fn_key]
            )
            extra_infos.append(
                data_item.non_tensor_batch.get("extra_info", {})
            )
            valid_response_lengths.append(valid_len)

        # 2. 批量调用 compute_score（向量化）
        scores = self.compute_score(
            data_sources=data_sources,        # 列表
            solution_strs=responses_str,      # 列表
            ground_truths=ground_truths,      # 列表
            extra_infos=extra_infos,          # 列表
            **self.reward_kwargs,
        )
        # 返回：[score1, score2, ...]

        # 3. 将 scores 放入 reward_tensor
        for i in range(len(data)):
            reward_tensor[i, valid_response_lengths[i] - 1] = scores[i]

        return reward_tensor
```

**特点**：
- 批量处理，效率更高
- Reward 函数需要支持列表输入
- 适合大 batch

**对比 Naive**：

| 维度 | Naive | Batch |
|------|-------|-------|
| compute_score 签名 | `(data_source, solution_str, ground_truth, ...)` | `(data_sources, solution_strs, ground_truths, ...)` |
| 处理方式 | 循环调用 | 一次调用 |
| 调试信息 | 打印详细 | 无 |
| 性能 | 慢 | 快 |

---

### 7.5 PrimeRewardManager - 并行异步打分

**位置**：`verl/workers/reward_manager/prime.py`

```python
async def parallel_compute_score_async(
    evaluation_func,
    completions: list[str],
    references: list[str],
    tasks: list[str],
    extra_info: list[dict],
    num_processes=64,
):
    """使用 ProcessPoolExecutor 并行执行"""
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 为每个样本创建异步任务
        tasks_async = [
            single_compute_score(
                evaluation_func, c, r, t, ei, executor, timeout=300.0
            )
            for c, r, t, ei in zip(completions, references, tasks, extra_info)
        ]

        # 等待所有任务完成
        results = await asyncio.gather(*tasks_async, return_exceptions=False)

    # 提取 scores
    scores = [r["score"] if isinstance(r, dict) else r for r in results]
    return scores

async def single_compute_score(
    evaluation_func, completion, reference, task, extra_info, executor, timeout
):
    """单个样本的评分（带超时）"""
    try:
        # 在进程池中执行
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                evaluation_func,
                completion, reference, task, extra_info
            ),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        print(f"[Timeout] Sample took > {timeout}s")
        return 0.0  # 超时返回 0
    except Exception as e:
        print(f"[Error] {e}")
        return 0.0  # 异常返回 0

@register("prime")
class PrimeRewardManager(AbstractRewardManager):
    def verify(self, data):
        """同步封装，内部调用异步函数"""
        scores = run_reward_scoring(
            self.compute_score,
            completions=sequences_str,
            references=ground_truth,
            tasks=data_sources,
            extra_info=extra_info,
            num_processes=64,
        )
        return scores
```

**特点**：
- 64 个进程并行执行（可配置）
- 每个样本超时 300 秒
- 超时或异常返回 0.0
- 适合代码执行、沙箱环境

**使用场景**：

```python
# 代码执行 Reward
def compute_score(solution_str, ground_truth, extra_info):
    # 1. 提取生成的代码
    code = extract_code(solution_str)

    # 2. 在沙箱中执行
    test_cases = extra_info["test_cases"]
    results = []
    for test_input, expected_output in test_cases:
        try:
            actual_output = execute_code(code, test_input, timeout=5)
            results.append(actual_output == expected_output)
        except Exception:
            results.append(False)

    # 3. 计算通过率
    pass_rate = sum(results) / len(results)
    return pass_rate

# Prime 会并行执行 64 个样本，每个样本最多 300 秒
```

---

### 7.6 内置 Reward Score 函数

**目录**：`verl/utils/reward_score/`

#### 7.6.1 GSM8K Reward

**文件**：`verl/utils/reward_score/gsm8k.py`

```python
def extract_solution(solution_str, method="strict"):
    """提取 #### 后的答案"""
    if method == "strict":
        # GSM8K 格式：#### 42
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            return None
        return solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        # 更宽松的匹配
        ...

def compute_score(
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.0,
    score=1.0
):
    """
    GSM8K 评分逻辑

    返回：
        - 答案正确 → 1.0
        - 格式正确但答案错误 → 0.0（或 format_score）
        - 格式错误 → 0.0
    """
    answer = extract_solution(solution_str, method=method)

    if answer is None:
        return 0.0  # 格式错误
    else:
        return score if answer == ground_truth else format_score
```

**示例**：

```python
solution_1 = "Let's think step by step.\n1 + 1 = 2\n#### 2"
solution_2 = "The answer is 2."
ground_truth = "2"

compute_score(solution_1, ground_truth)  # 1.0 (正确)
compute_score(solution_2, ground_truth)  # 0.0 (格式错误，没有 ####)
```

---

#### 7.6.2 MATH Reward（LaTeX）

**文件**：`verl/utils/reward_score/math_reward.py`

```python
def last_boxed_only_string(string):
    """提取 \boxed{} 中的内容"""
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces += 1
        elif string[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    return string[idx:right_brace_idx + 1]

def is_equiv(str1, str2):
    """判断两个数学表达式是否等价"""
    # 1. 规范化（去空格、LaTeX 命令）
    str1 = strip_string(str1)
    str2 = strip_string(str2)

    # 2. 直接字符串比较
    if str1 == str2:
        return True

    # 3. 尝试 sympy 符号计算
    try:
        parsed1 = parse_latex(str1)
        parsed2 = parse_latex(str2)
        return simplify(parsed1 - parsed2) == 0
    except:
        return False

def compute_score(solution_str, ground_truth) -> float:
    """MATH 数据集评分"""
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is None:
            return 0.0

        answer = remove_boxed(string_in_last_boxed)
        if is_equiv(answer, ground_truth):
            return 1.0
    except Exception:
        pass

    return 0.0
```

**示例**：

```python
solution_1 = "The solution is \\boxed{\\frac{1}{2}}"
solution_2 = "The answer is \\boxed{0.5}"
ground_truth = "\\frac{1}{2}"

compute_score(solution_1, ground_truth)  # 1.0
compute_score(solution_2, ground_truth)  # 1.0 (0.5 = 1/2)
```

---

#### 7.6.3 代码执行 Reward（Prime）

**文件**：`verl/utils/reward_score/prime_code.py`

```python
def compute_score(
    solution_str,
    ground_truth,
    extra_info,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    continuous=False,
):
    """
    代码执行评分

    参数：
        sandbox_fusion_url: 云函数 URL
        continuous: True → 返回通过率 [0, 1]，False → 返回 0 或 1
    """
    # 1. 提取代码
    code = extract_code_from_solution(solution_str)

    # 2. 准备测试用例
    test_cases = extra_info.get("test_cases", [])

    # 3. 发送到沙箱执行
    async with concurrent_semaphore:  # 控制并发数
        response = await send_to_sandbox(
            url=sandbox_fusion_url,
            code=code,
            test_cases=test_cases,
            memory_limit_mb=1024,
            timeout_seconds=10,
        )

    # 4. 解析结果
    passed = response["num_passed"]
    total = response["num_total"]

    if continuous:
        return passed / total  # 通过率
    else:
        return 1.0 if passed == total else 0.0  # 全对或全错
```

---

#### 7.6.4 Reward Dispatcher（路由）

**文件**：`verl/utils/reward_score/__init__.py`

```python
def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs
):
    """
    根据 data_source 路由到对应的 reward 函数
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k
        return gsm8k.compute_score(solution_str, ground_truth)

    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math_reward
        return math_reward.compute_score(solution_str, ground_truth)

    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code
        return prime_code.compute_score(
            solution_str, ground_truth, extra_info,
            continuous=True, **kwargs
        )

    elif data_source in ["searchR1_nq", "searchR1_triviaqa", "searchR1_hotpotqa"]:
        from . import search_r1_like_qa_em
        return search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function for {data_source=} not found")
```

**工作流程**：

1. 从 `data.non_tensor_batch["data_source"]` 获取数据来源
2. 根据数据来源选择对应的 reward 函数
3. 调用该函数计算 score

---

### 7.7 配置和加载

#### 7.7.1 Reward Manager 配置

**文件**：`verl/trainer/config/reward_manager.yaml`

```yaml
# Reward Manager 配置
_target_: verl.trainer.config.config.RewardManagerConfig

source: register  # 或 "importlib"（加载外部模块）
name: ${oc.select:reward_model.reward_manager,naive}  # 默认 naive

# 外部模块（当 source=importlib）
module:
  _target_: verl.trainer.config.config.ModuleConfig
  path: /path/to/my_reward_manager.py
  name: MyRewardManager
```

#### 7.7.2 加载 Reward Manager

**文件**：`verl/trainer/ppo/reward.py`

```python
def load_reward_manager(
    config: DictConfig,
    tokenizer: Any,
    num_examine: int,
    **reward_kwargs,
) -> AbstractRewardManager:
    """加载 RewardManager"""

    # 1. 加载自定义 reward 函数（如果有）
    compute_score = get_custom_reward_fn(config)

    # 2. 获取 RewardManager 类
    reward_manager_cfg = config.reward_manager
    if reward_manager_cfg.source == "register":
        # 从注册表加载
        from verl.workers.reward_manager import get_reward_manager_cls
        reward_manager_cls = get_reward_manager_cls(reward_manager_cfg.name)
    elif reward_manager_cfg.source == "importlib":
        # 从外部模块加载
        reward_manager_cls = load_extern_object(
            module_path=reward_manager_cfg.module.path,
            class_name=reward_manager_cfg.module.name,
        )

    # 3. 处理 sandbox fusion（代码执行）
    if compute_score is None and reward_manager_cfg.name == "prime":
        sandbox_config = config.reward_model.get("sandbox_fusion", {})
        if sandbox_config.get("url"):
            from functools import partial
            compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_config["url"],
                concurrent_semaphore=create_semaphore(
                    sandbox_config.get("max_concurrent", 64)
                ),
            )

    # 4. 实例化 RewardManager
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )

def compute_reward(data: DataProto, reward_fn: AbstractRewardManager):
    """计算 batch 的 reward"""
    reward_result = reward_fn(data, return_dict=True)
    reward_tensor = reward_result["reward_tensor"]
    reward_extra_infos = reward_result.get("reward_extra_info", {})
    return reward_tensor, reward_extra_infos
```

---

### 7.8 实战案例

#### 案例 1：GSM8K 规则 Reward

```bash
# 训练命令
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['~/data/gsm8k/train.parquet']" \
    data.reward_fn_key='data_source' \
    actor_rollout_ref.model.path=~/models/Qwen2.5-7B-Instruct \
    reward_model.reward_manager=naive \
    # 不设置 reward_model.enable，使用规则 reward
```

**数据格式**：

```python
{
    "prompts": "Janet's ducks lay 16 eggs per day...",
    "data_source": "openai/gsm8k",  # 路由到 gsm8k.compute_score
    "reward_model": {
        "ground_truth": "18"
    }
}
```

**Reward 流程**：

1. `NaiveRewardManager` 逐个处理样本
2. 解码响应：`"Let's think step by step... #### 18"`
3. `gsm8k.extract_solution()` 提取 `"18"`
4. 与 `ground_truth="18"` 比较 → 返回 1.0
5. 放入 `reward_tensor[i, last_token_idx] = 1.0`

---

#### 案例 2：代码执行 Reward（Prime）

```bash
# 训练命令
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['~/data/code/train.parquet']" \
    reward_model.reward_manager=prime \
    reward_model.sandbox_fusion.url='https://api.sandbox.com/run' \
    reward_model.sandbox_fusion.max_concurrent=128 \
    actor_rollout_ref.model.path=~/models/CodeLlama-7B
```

**数据格式**：

```python
{
    "prompts": "Write a function to check if a number is prime.",
    "data_source": "codecontests",
    "reward_model": {
        "ground_truth": null  # 代码执行不需要 ground_truth
    },
    "extra_info": {
        "test_cases": [
            {"input": "2", "output": "True"},
            {"input": "4", "output": "False"},
            {"input": "17", "output": "True"},
        ]
    }
}
```

**Reward 流程**：

1. `PrimeRewardManager` 并行处理 64 个样本
2. 对每个样本：
   - 提取代码：`extract_code(response)`
   - 发送到沙箱：`sandbox_fusion_url` with test cases
   - 执行测试用例（timeout 10s）
   - 返回通过率：`passed / total`
3. 超时样本返回 0.0
4. 全部完成后返回 reward_tensor

---

#### 案例 3：自定义 Reward 函数

```python
# my_reward.py
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    自定义 Reward：结合正确性和简洁性
    """
    from difflib import SequenceMatcher

    # 1. 正确性（基于相似度）
    similarity = SequenceMatcher(
        None,
        solution_str.lower(),
        ground_truth.lower()
    ).ratio()
    correctness_score = similarity

    # 2. 简洁性（字符数惩罚）
    max_length = extra_info.get("max_length", 500)
    length_penalty = min(len(solution_str) / max_length, 1.0)
    conciseness_score = 1 - length_penalty

    # 3. 关键词奖励
    keywords = extra_info.get("keywords", [])
    keyword_bonus = sum(kw in solution_str for kw in keywords) * 0.1

    # 4. 综合得分
    total_score = (
        correctness_score * 0.7 +
        conciseness_score * 0.2 +
        keyword_bonus * 0.1
    )

    return total_score
```

**使用配置**：

```bash
python3 -m verl.trainer.main_ppo \
    custom_reward_function.path='my_reward.py' \
    custom_reward_function.name='compute_score' \
    reward_model.reward_manager=naive \
    ...
```

---

### 7.9 小结

**Reward 系统核心流程**：

```
Data (responses)
    ↓
RewardManager.__call__()
    ↓
decode responses → extract metadata
    ↓
compute_score_fn()
    ↓
    ├─ default_compute_score (路由到具体函数)
    │   ├─ gsm8k.compute_score
    │   ├─ math_reward.compute_score
    │   ├─ prime_code.compute_score
    │   └─ custom compute_score
    ↓
reward_tensor [batch_size, seq_len]
(reward 在 last_token_idx)
```

**关键文件**：
- `verl/workers/reward_manager/abstract.py` - 抽象基类
- `verl/workers/reward_manager/{naive,batch,prime,dapo}.py` - 具体实现
- `verl/utils/reward_score/__init__.py` - Dispatcher
- `verl/utils/reward_score/{gsm8k,math_reward,prime_code}.py` - 内置函数
- `verl/trainer/ppo/reward.py` - 加载和配置

**进一步学习**：
- 查看 `verl/utils/reward_score/` 了解更多内置 reward 函数
- 阅读 `prime_code.py` 了解沙箱代码执行细节
- 研究 `DAPORewardManager` 了解长度感知的 reward 设计

---

### 进阶技巧 ✓
- [ ] 使用 LoRA 训练大模型
- [ ] 混合多个数据集

---

## 下一步

完成以上内容后，你可以：

1. **深入算法细节**
   - 阅读 HybridFlow 论文：https://arxiv.org/abs/2409.19256
   - 研究 DAPO、PRIME 等高级算法（在 recipe 子模块中）

2. **大规模训练**
   - 学习多机多卡训练
   - 了解 Megatron-LM 后端（超大模型）

3. **生产部署**
   - 模型导出和服务化
   - 推理加速优化

4. **贡献社区**
   - 在 GitHub 上提交 Issue/PR
   - 分享你的训练经验

---

**官方资源：**
- 文档：https://verl.readthedocs.io/en/latest/
- GitHub：https://github.com/volcengine/verl
- Slack：https://join.slack.com/t/verl-project

*最后更新: 2026-01-25*
