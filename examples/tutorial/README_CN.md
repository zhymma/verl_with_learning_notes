# 教程 (Tutorial)

> verl 入门和进阶教程

---

## 📋 概述

本目录包含 verl 的各种教程，从基础入门到高级应用，帮助你快速掌握 verl 的使用。

### 教程列表

| 教程 | 难度 | 时间 | 说明 |
|------|------|------|------|
| **agent_loop_get_started/** | ⭐ 入门 | 30 分钟 | Agent Loop 快速入门 |

---

## 🎓 agent_loop_get_started - Agent Loop 快速入门

### 教程目标

通过这个教程，你将学会：

- ✅ 理解 Agent Loop 的基本概念
- ✅ 实现一个简单的 Agent Loop
- ✅ 使用工具调用
- ✅ 训练一个 Tool-using Agent

### 前置条件

```bash
# 1. 安装 verl
pip install -e .[test,sglang]

# 2. 准备 GSM8K 数据
python examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_save_dir ~/data/gsm8k_tool

# 3. 下载模型
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

### 快速开始

```bash
# 进入教程目录
cd examples/tutorial/agent_loop_get_started

# 运行示例
python simple_agent_loop.py

# 预期输出：
# Agent Loop Example
# ==================
# Prompt: What is 123 + 456?
#
# Turn 1:
#   Assistant: Let me calculate: calculator(123 + 456)
#   Tool: 579
#
# Turn 2:
#   Assistant: The answer is 579.
#
# ✅ Correct!
```

### 教程结构

```
agent_loop_get_started/
├── README.md                   # 教程说明
├── simple_agent_loop.py        # 简单的 Agent Loop 示例
├── custom_agent_loop.py        # 自定义 Agent Loop
├── tool_calling_demo.py        # 工具调用演示
└── train_agent.sh              # 训练脚本
```

### 核心代码讲解

#### 1. 最简单的 Agent Loop

```python
# simple_agent_loop.py
from verl.workers.rollout.sglang_rollout.agent_loop.base import AgentLoopBase

class SimpleAgentLoop(AgentLoopBase):
    """最简单的 Agent Loop 示例"""

    async def generate(self, llm_server, data, **kwargs):
        prompts = data.batch['prompt']
        trajectories = []

        for prompt in prompts:
            history = prompt.copy()

            # 第一次生成
            response = await llm_server.generate([history])
            history.append({
                'role': 'assistant',
                'content': response['text'][0]
            })

            trajectories.append(history)

        data.batch['response'] = trajectories
        return data
```

#### 2. 带工具调用的 Agent Loop

```python
# tool_calling_demo.py
class ToolCallingAgentLoop(AgentLoopBase):
    """支持工具调用的 Agent Loop"""

    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.tools = {
            'calculator': self._calculator
        }

    async def generate(self, llm_server, data, **kwargs):
        prompts = data.batch['prompt']
        trajectories = []

        for prompt in prompts:
            history = prompt.copy()

            for turn in range(self.max_turns):
                # LLM 生成
                response = await llm_server.generate([history])
                assistant_msg = response['text'][0]

                history.append({
                    'role': 'assistant',
                    'content': assistant_msg
                })

                # 检查工具调用
                tool_call = self._parse_tool_call(assistant_msg)

                if tool_call is None:
                    # 没有工具调用，结束
                    break

                # 执行工具
                tool_result = self.tools[tool_call['name']](
                    tool_call['arguments']
                )

                history.append({
                    'role': 'tool',
                    'content': tool_result,
                    'name': tool_call['name']
                })

            trajectories.append(history)

        data.batch['response'] = trajectories
        return data

    def _calculator(self, args):
        """计算器工具"""
        expr = args['expression']
        try:
            return str(eval(expr))
        except:
            return "Error"

    def _parse_tool_call(self, text):
        """解析工具调用"""
        import re
        match = re.search(r'calculator\((.*?)\)', text)
        if match:
            return {
                'name': 'calculator',
                'arguments': {'expression': match.group(1)}
            }
        return None
```

#### 3. 训练 Agent

```bash
# train_agent.sh
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k_tool/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.use_agent_loop=True \
    actor_rollout_ref.rollout.agent_loop_class="examples.tutorial.agent_loop_get_started.tool_calling_demo.ToolCallingAgentLoop" \
    actor_rollout_ref.rollout.n=4 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=10
```

### 进阶练习

#### 练习 1：添加新工具

```python
# 在 ToolCallingAgentLoop 中添加
self.tools['search'] = self._search

def _search(self, args):
    """搜索工具（模拟）"""
    query = args['query']
    # 实际应该调用搜索 API
    return f"Search results for: {query}"
```

#### 练习 2：实现 Reward Shaping

```python
# reward_shaper.py
@RewardManager.register('tutorial_task')
def compute_reward(data):
    rewards = []

    for trajectory, ground_truth in zip(data['trajectories'], data['ground_truths']):
        reward = 0

        # 结果奖励
        if is_correct(trajectory, ground_truth):
            reward += 1.0

        # 过程奖励
        for msg in trajectory:
            if msg['role'] == 'tool':
                reward += 0.1  # 成功调用工具
            if msg['role'] == 'assistant' and 'let me' in msg['content'].lower():
                reward += 0.05  # 表明推理过程

        rewards.append(reward)

    return rewards
```

#### 练习 3：实现多步推理

```python
# multi_step_agent.py
class MultiStepAgentLoop(ToolCallingAgentLoop):
    """支持多步推理的 Agent Loop"""

    def __init__(self, max_turns=10, require_reasoning=True):
        super().__init__(max_turns)
        self.require_reasoning = require_reasoning

    async def generate(self, llm_server, data, **kwargs):
        # ... 同上，但添加推理步骤验证 ...

        for turn in range(self.max_turns):
            # 生成推理步骤
            reasoning_prompt = history + [{
                'role': 'user',
                'content': 'Explain your reasoning:'
            }]

            reasoning = await llm_server.generate([reasoning_prompt])

            history.append({
                'role': 'assistant',
                'content': f"Reasoning: {reasoning['text'][0]}"
            })

            # 然后生成答案
            # ...
```

### 常见问题

#### Q1: 如何调试 Agent Loop？

```python
# 添加日志
import logging
logging.basicConfig(level=logging.DEBUG)

class DebugAgentLoop(AgentLoopBase):
    async def generate(self, llm_server, data, **kwargs):
        logging.info(f"Input prompts: {len(data.batch['prompt'])}")

        for idx, prompt in enumerate(data.batch['prompt']):
            logging.info(f"Processing prompt {idx}")
            logging.info(f"Prompt content: {prompt}")

            # ... 生成逻辑 ...

            logging.info(f"Generated {len(history)} turns")
            for turn_idx, msg in enumerate(history):
                logging.info(f"Turn {turn_idx}: {msg['role']} - {msg['content'][:50]}")
```

#### Q2: Agent Loop 卡住怎么办？

```python
# 添加超时机制
import asyncio

class TimeoutAgentLoop(AgentLoopBase):
    async def generate(self, llm_server, data, timeout=60, **kwargs):
        try:
            return await asyncio.wait_for(
                self._generate_impl(llm_server, data, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logging.warning("Agent Loop timeout!")
            # 返回部分结果
            return self._create_fallback_response(data)
```

### 下一步

完成这个教程后，你可以：

1. **学习更多算法**：查看 [03_RL算法](../../learning_notes/03_RL算法/)
2. **深入 Agent Loop**：阅读 [05_Agent_RL](../../learning_notes/05_Agent_RL/)
3. **实战项目**：尝试 `examples/sglang_multiturn/` 中的完整示例
4. **自定义 Reward**：学习 [04_Reward设计](../../learning_notes/04_Reward设计/)

---

## 📚 其他教程（规划中）

### 即将推出

- [ ] **单轮 RL 训练** - 从零开始的 PPO/GRPO 教程
- [ ] **自定义 Reward 函数** - 实现复杂的 reward shaping
- [ ] **多模态训练** - VLM 的 RL 训练
- [ ] **分布式训练** - 多节点训练配置
- [ ] **模型部署** - 训练后模型的部署

### 贡献教程

欢迎贡献新的教程！

```bash
# 1. Fork 项目
git clone https://github.com/your-username/verl.git

# 2. 创建教程目录
mkdir -p examples/tutorial/your_tutorial_name

# 3. 添加教程文件
# - README.md（教程说明）
# - 示例代码
# - 训练脚本

# 4. 提交 PR
git add .
git commit -m "Add tutorial: your_tutorial_name"
git push origin your-branch
```

---

## 🔗 参考资料

### 学习笔记

- [01_快速上手](../../learning_notes/01_快速上手/) - 环境安装和第一次训练
- [02_数据准备](../../learning_notes/02_数据准备/) - 数据格式详解
- [05_Agent_RL](../../learning_notes/05_Agent_RL/) - Agent Loop 深度解析

### 官方文档

- [verl 文档](https://verl.readthedocs.io/)
- [GitHub 仓库](https://github.com/volcengine/verl)

### 相关示例

- `examples/sglang_multiturn/` - 完整的多轮训练示例
- `examples/data_preprocess/` - 数据预处理
- `examples/ppo_trainer/` - PPO 训练
- `examples/grpo_trainer/` - GRPO 训练

---

**创建时间**: 2026-01-28
**适用版本**: verl v0.2+
**维护者**: verl team
