# 05 - Agent RL

> 第五部分：工具调用和多轮对话的强化学习训练

---

## 📚 本章内容

### 📖 学习笔记

#### **Agent_Loop详解.md** - 完整的 Agent RL 教程（新！）
- Agent Loop 核心概念
- 系统架构深度解析
  - Server-Client 分离设计
  - 异步 Rollout 机制
  - 负载均衡和 Sticky Session
- AgentLoopBase 接口详解
- 工具调用实现
  - Tool 定义和配置
  - Tool 响应处理
  - calc_gsm8k_reward 工具示例
- 多轮对话训练
  - Chat History 管理
  - Token vs Text 一致性问题
  - Response Mask 设计
- 完整训练流程追踪
- MLflow Trace 调试技巧
- LangGraph Agent 集成
- 最佳实践和常见问题

### 🛠️ 实战脚本

本部分提供**源码级别的示例分析**：
- AgentLoopBase 实现: `verl/trainer/ppo/rollout/agent_loop/`
- Tool Agent 示例: `recipe/langgraph_agent/`
- 数据准备: `examples/data_preprocess/gsm8k_tool_agent_loop.py`

---

## 🚀 快速开始

### 步骤 1：理解 Agent Loop 架构

```
Agent Loop 分层架构:

┌─────────────────────────────────────────────────────┐
│             PPOTrainer (训练主循环)                  │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│          AgentLoopManager (管理 Workers)             │
│  - 分发 Prompts 到多个 Workers                       │
│  - 收集所有 AgentLoopOutput                         │
└─────────────────────┬───────────────────────────────┘
                      ↓
       ┌──────────────┴──────────────┐
       ↓                              ↓
┌──────────────┐              ┌──────────────┐
│ AgentLoop    │              │ AgentLoop    │
│ Worker 1     │     ...      │ Worker N     │
│              │              │              │
│ 运行多个     │              │ 运行多个     │
│ AgentLoop    │              │ AgentLoop    │
│ 协程         │              │ 协程         │
└──────┬───────┘              └──────┬───────┘
       │                              │
       └──────────────┬───────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│      AsyncLLMServerManager (LLM 代理)               │
│  - 负载均衡（首次请求选择负载最小的 Server）         │
│  - Sticky Session（后续请求发送到同一 Server）       │
└─────────────────────┬───────────────────────────────┘
                      ↓
       ┌──────────────┴──────────────┐
       ↓                              ↓
┌──────────────┐              ┌──────────────┐
│ AsyncServer  │              │ AsyncServer  │
│   (vLLM/     │     ...      │   (vLLM/     │
│   SGLang)    │              │   SGLang)    │
└──────────────┘              └──────────────┘
```

### 步骤 2：准备工具调用数据

创建包含 `agent_name` 字段的数据：

```bash
# 使用官方脚本准备 GSM8K tool agent 数据
python examples/data_preprocess/gsm8k_tool_agent_loop.py \
    --local_save_dir ~/data/gsm8k_tool
```

数据格式示例：
```python
{
    "data_source": "openai/gsm8k",
    "agent_name": "tool_agent",  # 关键字段！标识使用 tool agent loop
    "prompt": [
        {
            "role": "system",
            "content": "You are a math expert. You should use the `calc_gsm8k_reward` tool..."
        },
        {
            "role": "user",
            "content": "Janet's ducks lay 16 eggs per day. ..."
        }
    ],
    "reward_model": {
        "style": "rule",
        "ground_truth": "42"
    },
    "extra_info": {
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "calc_gsm8k_reward": {
                "create_kwargs": {"ground_truth": "42"}
            }
        }
    }
}
```

### 步骤 3：运行第一次 Agent 训练

```bash
# 安装 mlflow 用于查看 trace
pip install mlflow

# 启动训练（启用 tool calls 和 mlflow trace）
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh

# 训练完成后，启动 mlflow UI 查看 trace
mlflow ui -h 0.0.0.0 -p 5000 --backend-store-uri sqlite:////tmp/mlruns.db

# 在浏览器打开: http://<your-ip>:5000
```

关键配置：
```yaml
# 启用 Agent Loop
data.return_raw_chat: true
actor_rollout_ref.rollout.mode: async

# 指定 rollout 引擎
actor_rollout_ref.rollout.name: sglang  # 或 vllm
```

---

## 📖 推荐学习路径

### 第 1 天：Agent Loop 基础

1. **阅读** `Agent_Loop详解.md` 第 1-3 节（2 小时）
   - 理解 Agent Loop 的设计目标
   - 掌握系统架构
   - 理解异步 Rollout 的必要性

2. **实践** 查看数据格式
   ```bash
   python examples/data_preprocess/gsm8k_tool_agent_loop.py

   # 查看生成的数据
   python -c "
   import pandas as pd
   df = pd.read_parquet('~/data/gsm8k_tool/train.parquet')
   import json
   print(json.dumps(df.iloc[0].to_dict(), indent=2, ensure_ascii=False))
   "
   ```

3. **理解** AgentLoopOutput 结构
   ```python
   class AgentLoopOutput:
       prompt_ids: list[int]      # 原始 prompt 的 token IDs
       response_ids: list[int]    # 完整的响应（LLM 生成 + Tool 响应）
       response_mask: list[int]   # 1=LLM 生成，0=Tool 响应

   # 示例
   output = AgentLoopOutput(
       prompt_ids=[101, 2023, ...],
       response_ids=[
           234, 456,      # LLM: "Let me use"
           678,           # LLM: " tool"
           999,           # Tool 响应开始
           1000,          # Tool 响应
           1001,          # Tool 响应结束
           890, 891       # LLM: "So the answer is"
       ],
       response_mask=[
           1, 1,          # LLM 生成 ✓
           1,             # LLM 生成 ✓
           0,             # Tool 响应（不计算 loss）
           0,             # Tool 响应（不计算 loss）
           0,             # Tool 响应（不计算 loss）
           1, 1           # LLM 生成 ✓
       ]
   )
   ```

### 第 2 天：工具调用实现

1. **阅读** `Agent_Loop详解.md` 第 4-5 节（2 小时）
   - 理解 Tool 定义和配置
   - 掌握 Tool 响应处理
   - 学习 calc_gsm8k_reward 工具示例

2. **实践** 实现自定义 Tool
   ```python
   # 创建 my_tool.py
   class MyCalculatorTool:
       def __init__(self, **create_kwargs):
           """
           初始化工具
           create_kwargs 从数据的 tools_kwargs 中获取
           """
           self.precision = create_kwargs.get("precision", 2)

       def execute(self, expression: str, **execute_kwargs):
           """
           执行计算

           Args:
               expression: 要计算的表达式（如 "2 + 3 * 4"）

           Returns:
               dict: {"result": 计算结果, "success": True/False}
           """
           try:
               result = eval(expression)
               return {
                   "result": round(result, self.precision),
                   "success": True
               }
           except Exception as e:
               return {
                   "result": None,
                   "success": False,
                   "error": str(e)
               }
   ```

3. **配置** 在数据中使用自定义 Tool
   ```python
   data = {
       "agent_name": "tool_agent",
       "prompt": [...],
       "extra_info": {
           "need_tools_kwargs": True,
           "tools_kwargs": {
               "my_calculator": {
                   "create_kwargs": {"precision": 4}
               }
           }
       }
   }
   ```

### 第 3 天：多轮对话和调试

1. **阅读** `Agent_Loop详解.md` 第 6-7 节（2 小时）
   - 理解多轮对话的挑战
   - 掌握 Chat History 管理
   - 学习 Token vs Text 一致性问题

2. **实践** 使用 MLflow Trace 调试
   ```bash
   # 启动训练（自动记录 trace）
   bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh

   # 在另一个终端启动 mlflow UI
   mlflow ui -h 0.0.0.0 -p 5000 --backend-store-uri sqlite:////tmp/mlruns.db
   ```

   在 MLflow UI 中查看：
   - 每个 turn 的 LLM 生成
   - Tool 调用和响应
   - 完整的 token IDs
   - Response mask

3. **分析** Token vs Text 一致性
   ```python
   # 问题示例
   llm_output = "Let me use <tool_call>calc(2+3)</tool_call> to solve it"
   # Token IDs: [123, 456, 789, ...]

   # Tool Parser 提取后
   parsed_message = {
       "role": "assistant",
       "content": "Let me use  to solve it",  # 注意：tool_call 被移除
       "tool_calls": [{"name": "calc", "arguments": "2+3"}]
   }

   # Re-encode
   new_token_ids = tokenizer.encode(parsed_message["content"])
   # [123, 456, 999, ...]  # 不一致！

   # 影响：PPO 训练中的 log_prob 计算不准确
   ```

   **解决方案：Token-based API**
   - 使用 `generate(prompt_ids) -> response_ids`
   - 避免 text → tokens 的转换
   - 保持 trajectory 的一致性

---

## 📋 学习检查清单

### Agent Loop 基础 ✓
- [x] 理解 Server-Client 分离设计的原因
- [x] 掌握异步 Rollout 的工作原理
- [x] 理解 AgentLoopOutput 的结构
- [x] 知道 response_mask 的作用
- [x] 理解负载均衡和 Sticky Session

### 工具调用掌握 ✓
- [x] 能够配置 tool agent 数据
- [x] 理解 tools_kwargs 的结构
- [x] 实现过自定义 Tool
- [x] 知道 Tool 响应如何嵌入到 trajectory
- [x] 理解 response_mask 如何过滤 Tool 部分

### 多轮对话训练 ✓
- [x] 理解多轮对话的数据格式
- [x] 掌握 Chat History 管理
- [x] 理解 Token vs Text 一致性问题
- [x] 能够使用 MLflow Trace 调试
- [x] 知道"Failed to decode tool call"的原因

---

## 🎯 学习目标

完成本章后，你应该能够：

✅ 深入理解 Agent Loop 的架构和设计原理
✅ 准备和使用工具调用数据
✅ 实现自定义 Tool 和 AgentLoopBase
✅ 训练多轮对话的 Agent
✅ 使用 MLflow Trace 调试 Agent 行为
✅ 理解并解决 Token-Text 一致性问题
✅ 集成 LangGraph 等 Agent 框架

---

## 💡 重点内容

### AgentLoopBase 接口

```python
from abc import ABC, abstractmethod
from typing import Any

class AgentLoopBase(ABC):
    @abstractmethod
    async def run(
        self,
        sampling_params: dict[str, Any],
        **kwargs
    ) -> AgentLoopOutput:
        """
        实现 Agent 的主循环

        Args:
            sampling_params: LLM 采样参数（temperature, top_p, etc.）
            **kwargs: 数据集字段（prompt, extra_info, etc.）

        Returns:
            AgentLoopOutput: 包含 prompt_ids, response_ids, response_mask
        """
        raise NotImplementedError
```

### AgentLoopOutput 结构

```python
class AgentLoopOutput(BaseModel):
    prompt_ids: list[int]      # Prompt token IDs
    response_ids: list[int]    # Response token IDs（LLM + Tool）
    response_mask: list[int]   # 1=LLM, 0=Tool
```

**关键点：**
- `response_ids` 包含 LLM 生成的 tokens 和 Tool 响应的 tokens
- `response_mask` 用于区分哪些是 LLM 生成（需要计算 loss），哪些是 Tool 响应（不计算 loss）
- 在 PPO 训练中，只有 `response_mask=1` 的 tokens 会被用于计算 policy loss

### 异步 Rollout 的必要性

**问题：** Tool 调用涉及外部 I/O（网络请求、数据库查询、代码执行）

**传统同步方式的问题：**
```python
# 同步方式（GPU 空闲）
for prompt in batch:
    llm_response = llm.generate(prompt)       # GPU 工作
    tool_result = call_tool(llm_response)     # GPU 空闲等待！⏰
    final_response = llm.generate(context)    # GPU 工作
```

**异步方式的优势：**
```python
# 异步方式（GPU 利用率高）
async def agent_loop(prompt):
    llm_response = await llm.generate(prompt)     # GPU 工作
    tool_result = await call_tool(llm_response)   # GPU 处理其他请求✓
    final_response = await llm.generate(context)  # GPU 工作
    return final_response

# 并发执行多个 agent loops
results = await asyncio.gather(*[
    agent_loop(p) for p in batch
])
```

**性能提升：**
- 单个 Agent Loop 时间：可能相同
- Batch 吞吐量：提升 2-5 倍（取决于 Tool I/O 时间）

### Token-based API vs Chat Completion API

| 特性 | Token-based API | Chat Completion API |
|------|----------------|---------------------|
| **输入** | `prompt_ids: list[int]` | `messages: list[dict]` |
| **输出** | `response_ids: list[int]` | `text: str` |
| **一致性** | ✅ 保证 | ❌ 可能不一致 |
| **训练准确性** | ✅ 高 | ❌ 可能有偏差 |
| **调试难度** | 中等 | 简单 |
| **适用场景** | RL 训练（推荐） | Serving, Agent 系统 |

---

## ❓ 常见问题

### Q1: Agent Loop 和普通 Rollout 的区别？

**普通 Rollout（单轮生成）：**
```
Prompt → LLM Generate → Response
```

**Agent Loop（多轮交互）：**
```
Prompt → LLM Generate → Tool Call → Tool Response
       ↑                                   ↓
       └────── LLM Generate ← Context ─────┘
                     ↓
              Final Response
```

**关键区别：**
- Agent Loop 有多轮 LLM 交互
- 包含外部 Tool 调用
- 需要管理 Chat History
- Response 包含 LLM 和 Tool 的混合内容

### Q2: response_mask 为什么重要？

**作用：**区分 response 中哪些 tokens 是 LLM 生成的，哪些是 Tool 响应

**为什么需要？**
- PPO loss 只应该作用于 LLM 生成的 tokens
- Tool 响应是确定性的，不应该优化

**示例：**
```python
response_text = "Let me calculate: <tool>calc(2+3)</tool> The result is 5"
response_ids = [123, 456, ..., 999, 1000, ..., 789]
response_mask = [  1,   1, ...,   0,    0, ...,   1]
                   ↑   LLM 生成      ↑ Tool      ↑ LLM 生成

# 计算 loss 时
policy_loss = -log_prob[response_mask == 1] * advantages[response_mask == 1]
# 只对 LLM 生成的部分计算 loss
```

### Q3: "Failed to decode tool call" 错误怎么办？

**原因：**
模型在训练初期可能生成不正确的 tool call 格式

**正确格式（示例）：**
```xml
<tool_call>
{"name": "calc", "arguments": {"expression": "2+3"}}
</tool_call>
```

**错误格式（模型可能生成）：**
```
Let me use <tool_call>calc(2+3 to solve
```

**处理方法：**
1. **这是正常现象**：训练过程中会逐步改善
2. **继续训练**：RL 训练会惩罚错误格式，奖励正确格式
3. **检查 Reward**：确保正确格式有更高的 reward
4. **调整提示词**：在 system prompt 中明确格式要求

**调试技巧：**
```python
# 在 AgentLoop 中添加日志
print(f"[Debug] LLM output: {llm_text}")
print(f"  Extracted tool calls: {tool_calls}")
print(f"  Parse success: {parse_success}")
```

### Q4: 如何选择 vLLM vs SGLang？

| 特性 | vLLM | SGLang |
|------|------|--------|
| **性能** | 高 | 更高（优化了多轮） |
| **稳定性** | 成熟稳定 | 较新，发展中 |
| **多轮对话** | 支持 | 优化更好 |
| **部署复杂度** | 简单 | 稍复杂 |
| **推荐场景** | 通用训练 | 多轮对话优先 |

**建议：**
- **初次尝试**：使用 vLLM（更稳定）
- **多轮对话优化**：使用 SGLang
- **生产环境**：两者都可以，看具体需求

### Q5: MLflow Trace 看什么？

**关键信息：**

1. **每个 Turn 的 LLM 生成**
   ```
   Turn 1:
     Input: "Solve: 2+3*4"
     Output: "Let me use <tool_call>calc(2+3*4)</tool_call>"
   ```

2. **Tool 调用详情**
   ```
   Tool: calc
   Input: {"expression": "2+3*4"}
   Output: {"result": 14}
   ```

3. **完整的 Token IDs**
   ```
   prompt_ids: [101, 234, 456, ...]
   response_ids: [789, 890, ..., 999, 1000, ..., 1234]
   response_mask: [1, 1, ..., 0, 0, ..., 1]
   ```

4. **Reward 计算**
   ```
   Final answer: "#### 14"
   Ground truth: "14"
   Reward: 1.0
   ```

**调试技巧：**
- 查看 response_mask 是否正确标记 Tool 部分
- 检查 Tool 响应是否正确嵌入
- 验证最终 response 的完整性

---

## 🔗 相关资源

### 本地文件
- Agent Loop 详解: `Agent_Loop详解.md`
- 第一部分（Ray Trainer）: `../01_快速上手/ray_trainer_详解.md`
- 项目概览: `../../CLAUDE.md`

### 官方文档
- [Agentic RL Training](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)
- [Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
- [Rollout Trace](https://verl.readthedocs.io/en/latest/advance/rollout_trace.html)

### 代码位置
- AgentLoopBase: `verl/trainer/ppo/rollout/agent_loop/`
- AsyncLLMServer: `verl/trainer/ppo/rollout/async_server/`
- 数据预处理: `examples/data_preprocess/gsm8k_tool_agent_loop.py`
- LangGraph 示例: `recipe/langgraph_agent/`

### 示例脚本
- `examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh`
- `examples/grpo_trainer/run_qwen2-7b_seq_balance.sh`

---

## ⏭️ 下一步

完成本章后：
- **实战项目**：在实际任务上应用 Agent RL
- **高级主题**：研究 LangGraph、CrewAI 等框架集成
- **论文复现**：尝试复现 Retool 等论文

---

*创建时间: 2026-01-26*
*预计完成时间: 3-5 天*
