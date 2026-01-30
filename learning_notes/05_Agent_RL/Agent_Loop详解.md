# Agent Loop 详解

> 深入理解多轮对话和工具调用的强化学习训练

---

## 📖 目录

1. [Agent Loop 核心概念](#1-agent-loop-核心概念)
2. [系统架构深度解析](#2-系统架构深度解析)
3. [AgentLoopBase 接口详解](#3-agentloopbase-接口详解)
4. [工具调用实现](#4-工具调用实现)
5. [多轮对话训练](#5-多轮对话训练)
6. [完整训练流程追踪](#6-完整训练流程追踪)
7. [调试技巧](#7-调试技巧)
8. [最佳实践](#8-最佳实践)

---

## 1. Agent Loop 核心概念

### 1.1 什么是 Agent Loop？

**Agent Loop** 是 verl 为多轮对话和工具调用设计的通用接口。

**核心特点：**
- ✅ 支持多轮 LLM 交互
- ✅ 支持工具调用（Tool Calls）
- ✅ 异步执行，提高 GPU 利用率
- ✅ 可插拔的用户自定义 Agent
- ✅ 统一的 LLM generate API

**设计目标：**
1. **可插拔**：用户可以自定义 Agent Loop 逻辑
2. **统一 API**：屏蔽不同推理引擎（vLLM/SGLang）的差异
3. **负载均衡**：多个 LLM Server 之间自动负载均衡

**非目标（Not Goals）：**
- Tool 如何定义（由用户决定）
- Tool 如何调用（由用户实现）

### 1.2 Agent Loop vs 单轮 Rollout

**单轮 Rollout：**
```
┌────────┐      ┌─────────────┐      ┌──────────┐
│ Prompt │  →   │ LLM Generate│  →   │ Response │
└────────┘      └─────────────┘      └──────────┘
```

**Agent Loop（多轮 + 工具）：**
```
┌────────┐      ┌──────────────┐
│ Prompt │  →   │ LLM Generate │
└────────┘      └──────┬───────┘
                       ↓
               ┌───────────────┐
               │ Tool Call?    │
               └───┬───────┬───┘
                  Yes     No
                   ↓       ↓
            ┌──────────┐  ┌──────────┐
            │Call Tool │  │  Done    │
            └─────┬────┘  └──────────┘
                  ↓
            ┌──────────────┐
            │ Tool Response│
            └──────┬───────┘
                   ↓
            ┌──────────────┐
            │ LLM Generate │ (with context)
            └──────┬───────┘
                   ↓
                  ...
```

**关键区别：**

| 特性 | 单轮 Rollout | Agent Loop |
|------|-------------|-----------|
| **LLM 调用次数** | 1 次 | 多次 |
| **Tool 调用** | 无 | 有 |
| **外部 I/O** | 无 | 有（Tool 调用）|
| **GPU 利用率** | 100%（生成时） | 需要异步优化 |
| **Response 结构** | 纯 LLM 生成 | LLM + Tool 混合 |

### 1.3 为什么需要异步 Rollout？

**问题：** Tool 调用涉及外部 I/O，可能很慢

**同步执行的问题：**
```python
# 伪代码：同步 Agent Loop
def sync_agent_loop(prompt):
    response1 = llm.generate(prompt)        # GPU 工作 ✓
    tool_result = call_api(response1)       # GPU 空闲 ✗ (等待网络请求)
    response2 = llm.generate(context)       # GPU 工作 ✓
    return response2

# 批量执行
for prompt in batch:
    result = sync_agent_loop(prompt)  # 串行执行，GPU 大量空闲
```

**时间分析：**
- LLM Generate: 100ms × 2 = 200ms
- Tool Call (API): 500ms
- **总时间**: 700ms
- **GPU 利用率**: 200ms / 700ms = 28.6%

**异步执行的优势：**
```python
# 伪代码：异步 Agent Loop
async def async_agent_loop(prompt):
    response1 = await llm.generate(prompt)        # GPU 工作 ✓
    tool_result = await call_api(response1)       # GPU 处理其他请求 ✓
    response2 = await llm.generate(context)       # GPU 工作 ✓
    return response2

# 并发执行
results = await asyncio.gather(*[
    async_agent_loop(p) for p in batch
])
```

**时间分析（batch_size=8）：**
- 总时间：约 1000ms（并发执行）
- GPU 利用率：约 80%（大部分时间都在处理某个请求）

**性能提升：**
- 同步：700ms × 8 = 5600ms
- 异步：约 1000ms
- **提升：5.6 倍**

---

## 2. 系统架构深度解析

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     PPOTrainer                              │
│  - 训练主循环                                                │
│  - 调用 AgentLoopManager.generate_sequences()              │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  AgentLoopManager                           │
│  - wake_up() 所有 LLM Servers (同步权重)                    │
│  - 分发 prompts 到多个 AgentLoopWorkers                     │
│  - 收集所有 AgentLoopOutput                                 │
│  - sleep() 所有 LLM Servers (释放显存)                      │
└──────────────┬────────────────────────┬─────────────────────┘
               ↓                        ↓
┌──────────────────────┐     ┌──────────────────────┐
│  AgentLoopWorker 1   │     │  AgentLoopWorker N   │
│                      │ ... │                      │
│  运行多个并发的       │     │  运行多个并发的       │
│  AgentLoop 协程      │     │  AgentLoop 协程      │
└──────────┬───────────┘     └──────────┬───────────┘
           │                            │
           └────────────┬───────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              AsyncLLMServerManager                          │
│  - 负载均衡（选择负载最小的 Server）                         │
│  - Sticky Session（后续请求发到同一 Server）                │
│  - 提供统一的 generate() 接口                               │
└──────────────┬────────────────────────┬─────────────────────┘
               ↓                        ↓
┌──────────────────────┐     ┌──────────────────────┐
│  AsyncServer 1       │     │  AsyncServer N       │
│  (vLLM/SGLang)       │ ... │  (vLLM/SGLang)       │
│                      │     │                      │
│  - 连接一个 DP group │     │  - 连接一个 DP group │
│  - 同步训练权重      │     │  - 同步训练权重      │
└──────────────────────┘     └──────────────────────┘
```

### 2.2 核心组件详解

#### 2.2.1 AgentLoopManager

**位置：** `verl/trainer/ppo/rollout/agent_loop/agent_loop_manager.py`

**职责：**
1. 管理所有 LLM Server 的生命周期（wake_up/sleep）
2. 分发 prompts 到多个 Worker
3. 收集并整理所有 AgentLoopOutput

**关键方法：**

```python
class AgentLoopManager:
    def wake_up(self):
        """
        唤醒所有 LLM Servers
        - 从 FSDP/Megatron-LM 同步最新的模型权重到 vLLM/SGLang
        - 为新一轮 Rollout 做准备
        """
        for server in self.servers:
            server.wake_up()

    async def generate_sequences(self, batch):
        """
        主入口：生成一个 batch 的 sequences

        流程：
        1. wake_up() 所有 servers
        2. 分发 prompts 到多个 workers
        3. 并发执行所有 agent loops
        4. 收集所有 outputs
        5. sleep() 所有 servers
        """
        # 1. 唤醒 servers
        self.wake_up()

        # 2. 分发到 workers
        chunks = split_batch(batch, num_workers=self.num_workers)

        # 3. 并发执行
        tasks = [
            worker.process_chunk(chunk)
            for worker, chunk in zip(self.workers, chunks)
        ]
        results = await asyncio.gather(*tasks)

        # 4. 整理输出
        outputs = self.merge_outputs(results)

        # 5. 休眠 servers
        self.sleep()

        return outputs

    def sleep(self):
        """
        休眠所有 LLM Servers
        - 释放 KV Cache
        - (可选) Offload weights 到 CPU
        """
        for server in self.servers:
            server.sleep()
```

#### 2.2.2 AgentLoopWorker

**位置：** `verl/trainer/ppo/rollout/agent_loop/agent_loop_worker.py`

**职责：**
1. 接收一个 chunk 的 prompts
2. 为每个 prompt 创建一个 AgentLoop 实例
3. 并发执行所有 AgentLoop 协程

**关键方法：**

```python
class AgentLoopWorker:
    async def process_chunk(self, chunk):
        """
        处理一个 chunk 的 prompts

        Args:
            chunk: 包含多个 prompts 的 batch chunk

        Returns:
            List[AgentLoopOutput]: 每个 prompt 的输出
        """
        tasks = []

        for i, prompt_data in enumerate(chunk):
            # 根据 agent_name 选择 AgentLoop 类
            agent_name = prompt_data.get("agent_name", "single_turn")

            if agent_name == "tool_agent":
                agent_loop = ToolAgentLoop(
                    llm_server=self.llm_server_manager,
                    tokenizer=self.tokenizer,
                    **prompt_data
                )
            else:
                agent_loop = SingleTurnAgentLoop(
                    llm_server=self.llm_server_manager,
                    tokenizer=self.tokenizer,
                    **prompt_data
                )

            # 创建异步任务
            task = agent_loop.run(sampling_params=self.sampling_params)
            tasks.append(task)

        # 并发执行所有 agent loops
        outputs = await asyncio.gather(*tasks)

        return outputs
```

#### 2.2.3 AsyncLLMServerManager

**位置：** `verl/trainer/ppo/rollout/async_server/async_llm_server_manager.py`

**职责：**
1. 管理多个 AsyncServer 实例
2. 负载均衡（首次请求选择负载最小的 Server）
3. Sticky Session（后续请求发到同一 Server）

**关键方法：**

```python
class AsyncLLMServerManager:
    def __init__(self, servers: List[AsyncServerBase]):
        self.servers = servers
        self.request_to_server = {}  # request_id → server_id

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]:
        """
        生成 tokens

        Args:
            request_id: 请求 ID（用于 sticky session）
            prompt_ids: Prompt token IDs
            sampling_params: 采样参数

        Returns:
            List[int]: 生成的 token IDs
        """
        # 1. 选择 Server（负载均衡 + sticky session）
        if request_id in self.request_to_server:
            # Sticky session: 后续请求发到同一 server
            server_id = self.request_to_server[request_id]
        else:
            # 负载均衡: 选择负载最小的 server
            server_id = self._select_server_with_least_load()
            self.request_to_server[request_id] = server_id

        server = self.servers[server_id]

        # 2. 调用 server generate
        response_ids = await server.generate(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id
        )

        return response_ids

    def _select_server_with_least_load(self) -> int:
        """
        选择负载最小的 server

        Returns:
            int: server_id
        """
        loads = [server.get_current_load() for server in self.servers]
        return loads.index(min(loads))
```

**Sticky Session 的必要性：**

多轮对话中，后续 turns 需要访问之前的 KV Cache：

```
Turn 1: Server 0 → 生成 response 1，缓存 KV Cache
Turn 2: Server 0 → 复用 KV Cache，生成 response 2 ✓

如果 Turn 2 发到 Server 1 → 没有 KV Cache，需要重新计算 ✗
```

#### 2.2.4 AsyncServer (vLLM/SGLang)

**位置：** `verl/trainer/ppo/rollout/async_server/`

**职责：**
1. 封装 vLLM/SGLang 的推理引擎
2. 提供统一的 generate() 接口
3. 处理权重同步（wake_up/sleep）

**vLLM vs SGLang 架构差异：**

**vLLM:**
```
┌────────────────┐
│  AsyncServer   │ (运行在独立进程)
└────────┬───────┘
         │ ZeroMQ
         ↓
┌────────────────┐
│ AsyncLLMEngine │ (运行在独立进程)
└────────┬───────┘
         │ ZeroMQ
         ↓
┌────────────────┐
│  ModelRunner   │ (运行在 FSDP Worker 进程)
└────────────────┘
```

**SGLang:**
```
┌────────────────┐
│  AsyncServer   │ (运行在独立进程)
└────────┬───────┘
         │ Ray RPC
         ↓
┌────────────────┐
│ AsyncLLMEngine │ (运行在 FSDP Worker-0 进程)
└────────┬───────┘
         │ ZeroMQ
         ↓
┌────────────────┐
│  ModelRunner   │ (Subprocesses)
└────────────────┘
```

**关键差异：**
- vLLM: AsyncLLMEngine 独立进程，通过 ZeroMQ 通信
- SGLang: AsyncLLMEngine 在 Worker-0，AsyncServer 通过 Ray RPC 调用

### 2.3 完整数据流

```
┌─────────────┐
│  PPOTrainer │
│  Batch:     │
│  256 prompts│
└──────┬──────┘
       │
       ↓ generate_sequences(batch)
┌─────────────────┐
│ AgentLoopManager│
└──────┬──────────┘
       │ wake_up()
       ↓
┌─────────────────┐
│ LLM Servers     │ (同步权重)
└─────────────────┘
       ↓
┌─────────────────┐
│ 分发到 Workers  │
│ Worker 1: 128   │
│ Worker 2: 128   │
└──────┬──────────┘
       │
       ↓ asyncio.gather()
┌──────────────────────────┐
│  AgentLoopWorker 1       │
│  ┌───────────────────┐   │
│  │ AgentLoop 1       │   │
│  │ AgentLoop 2       │   │
│  │ ...               │   │
│  │ AgentLoop 128     │   │
│  └───────────────────┘   │
│  (并发执行 128 个协程)    │
└────────┬─────────────────┘
         │
         ↓ AgentLoop.run()
┌─────────────────────────┐
│  每个 AgentLoop:         │
│  1. LLM Generate        │
│  2. Parse Tool Call     │
│  3. Execute Tool        │
│  4. LLM Generate (ctx)  │
│  5. Return Output       │
└────────┬────────────────┘
         │
         ↓ AgentLoopOutput
┌─────────────────────────┐
│  {                      │
│    prompt_ids: [...]    │
│    response_ids: [...]  │
│    response_mask: [...] │
│  }                      │
└────────┬────────────────┘
         │
         ↓ gather all outputs
┌─────────────────────────┐
│  AgentLoopManager       │
│  收集 256 个 outputs     │
└────────┬────────────────┘
         │ sleep()
         ↓
┌─────────────────────────┐
│  LLM Servers (释放 KV)  │
└─────────────────────────┘
         │
         ↓ return
┌─────────────────────────┐
│  PPOTrainer             │
│  进入 Reward 计算阶段    │
└─────────────────────────┘
```

---

## 3. AgentLoopBase 接口详解

### 3.1 接口定义

**位置：** `verl/trainer/ppo/rollout/agent_loop/agent_loop_base.py`

```python
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

class AgentLoopOutput(BaseModel):
    """Agent Loop 的输出"""

    prompt_ids: list[int]
    """Prompt token IDs"""

    response_ids: list[int]
    """Response token IDs (包含 LLM 生成 + Tool 响应)"""

    response_mask: list[int]
    """Response mask: 1=LLM 生成，0=Tool 响应"""


class AgentLoopBase(ABC):
    """Agent Loop 基类"""

    def __init__(
        self,
        llm_server: AsyncLLMServerManager,
        tokenizer,
        **kwargs
    ):
        """
        初始化 Agent Loop

        Args:
            llm_server: LLM Server 管理器
            tokenizer: Tokenizer
            **kwargs: 数据集字段（prompt, extra_info, etc.）
        """
        self.llm_server = llm_server
        self.tokenizer = tokenizer

        # 从 kwargs 提取数据
        self.prompt = kwargs.get("prompt")  # List[dict]
        self.extra_info = kwargs.get("extra_info", {})
        self.data_source = kwargs.get("data_source")

    @abstractmethod
    async def run(
        self,
        sampling_params: dict[str, Any],
        **kwargs
    ) -> AgentLoopOutput:
        """
        运行 Agent Loop（需要用户实现）

        Args:
            sampling_params: LLM 采样参数
            **kwargs: 额外参数

        Returns:
            AgentLoopOutput: 包含 prompt_ids, response_ids, response_mask
        """
        raise NotImplementedError
```

### 3.2 SingleTurnAgentLoop 示例

**最简单的实现：单轮生成**

```python
class SingleTurnAgentLoop(AgentLoopBase):
    """单轮 Agent Loop（不使用工具）"""

    async def run(
        self,
        sampling_params: dict[str, Any],
        **kwargs
    ) -> AgentLoopOutput:
        """
        单轮生成

        流程：
        1. 将 prompt messages 转换为 token IDs
        2. 调用 LLM generate
        3. 返回 AgentLoopOutput
        """
        # 1. Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            self.prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. Tokenize
        prompt_ids = self.tokenizer.encode(prompt_text)

        # 3. Generate
        response_ids = await self.llm_server.generate(
            request_id=self._get_request_id(),
            prompt_ids=prompt_ids,
            sampling_params=sampling_params
        )

        # 4. Response mask (全部是 LLM 生成)
        response_mask = [1] * len(response_ids)

        # 5. 返回
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask
        )

    def _get_request_id(self) -> str:
        """生成唯一的 request_id"""
        return f"{self.data_source}_{id(self)}"
```

### 3.3 ToolAgentLoop 示例

**带工具调用的实现**

```python
class ToolAgentLoop(AgentLoopBase):
    """支持工具调用的 Agent Loop"""

    def __init__(self, llm_server, tokenizer, **kwargs):
        super().__init__(llm_server, tokenizer, **kwargs)

        # 初始化工具
        self.tools = self._init_tools()

    def _init_tools(self):
        """
        从 extra_info 中初始化工具

        extra_info.tools_kwargs = {
            "tool_name": {
                "create_kwargs": {...},
                "execute_kwargs": {...}
            }
        }
        """
        tools = {}

        tools_kwargs = self.extra_info.get("tools_kwargs", {})

        for tool_name, tool_config in tools_kwargs.items():
            # 动态导入工具类
            tool_class = self._get_tool_class(tool_name)

            # 创建工具实例
            create_kwargs = tool_config.get("create_kwargs", {})
            tools[tool_name] = tool_class(**create_kwargs)

        return tools

    async def run(
        self,
        sampling_params: dict[str, Any],
        **kwargs
    ) -> AgentLoopOutput:
        """
        多轮 Agent Loop with Tools

        流程：
        1. LLM Generate
        2. Parse Tool Call
        3. Execute Tool (if needed)
        4. LLM Generate with context (if needed)
        5. Return AgentLoopOutput
        """
        # 初始化
        chat_history = list(self.prompt)  # 复制 prompt
        all_response_ids = []
        all_response_mask = []

        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text)

        request_id = self._get_request_id()
        max_turns = 10  # 最大轮次

        for turn in range(max_turns):
            # Turn N: LLM Generate
            response_ids = await self.llm_server.generate(
                request_id=request_id,
                prompt_ids=prompt_ids if turn == 0 else [],  # 首次发 prompt，后续为空
                sampling_params=sampling_params
            )

            # Decode
            response_text = self.tokenizer.decode(response_ids)

            # 添加到总响应
            all_response_ids.extend(response_ids)
            all_response_mask.extend([1] * len(response_ids))  # LLM 生成

            # 更新 chat history
            chat_history.append({
                "role": "assistant",
                "content": response_text
            })

            # Parse tool calls
            tool_calls = self._parse_tool_calls(response_text)

            if not tool_calls:
                # 没有 tool call，结束
                break

            # Execute tools
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]

                # 调用工具
                tool_result = self.tools[tool_name].execute(**tool_args)

                # 将 tool result 转换为 tokens
                tool_result_text = json.dumps(tool_result)
                tool_result_ids = self.tokenizer.encode(tool_result_text)

                # 添加到总响应
                all_response_ids.extend(tool_result_ids)
                all_response_mask.extend([0] * len(tool_result_ids))  # Tool 响应

                # 更新 chat history
                chat_history.append({
                    "role": "tool",
                    "content": tool_result_text,
                    "tool_call_id": tool_call.get("id")
                })

            # 下一轮：LLM 继续生成（基于 tool results）
            # prompt_ids 为空，因为 chat history 已经在 server 端

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            response_mask=all_response_mask
        )

    def _parse_tool_calls(self, text: str) -> list[dict]:
        """
        从文本中解析工具调用

        示例格式:
        <tool_call>
        {"name": "calc", "arguments": {"expression": "2+3"}}
        </tool_call>
        """
        tool_calls = []

        # 正则提取 <tool_call>...</tool_call>
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # 解析失败（格式错误）
                print(f"Failed to parse tool call: {match}")
                continue

        return tool_calls

    def _get_request_id(self) -> str:
        return f"{self.data_source}_{id(self)}"

    def _get_tool_class(self, tool_name: str):
        """动态导入工具类"""
        # 示例：从预定义的工具注册表获取
        from my_tools import TOOL_REGISTRY
        return TOOL_REGISTRY[tool_name]
```

---

## 4. 工具调用实现

### 4.1 工具定义和注册

在 Agent RL 中，工具是 Agent 完成特定任务的关键。我们先看如何定义和注册工具。

#### 工具的数据格式

从 `gsm8k_tool_agent_loop.py` 我们可以看到工具是如何在数据集中定义的：

```python
# examples/data_preprocess/gsm8k_tool_agent_loop.py:96-104
"extra_info": {
    "need_tools_kwargs": True,  # 表示需要工具
    "tools_kwargs": {
        "calc_gsm8k_reward": {  # 工具名
            "create_kwargs": {"ground_truth": solution},  # 工具创建参数
            # "execute_kwargs": {},    # 工具执行参数（可选）
            # "calc_reward_kwargs": {},  # Reward 计算参数（可选）
            # "release_kwargs": {},    # 工具释放参数（可选）
        },
    },
}
```

**关键字段：**
- `need_tools_kwargs`: 是否需要工具
- `tools_kwargs`: 工具配置字典
  - 每个工具有 4 个生命周期钩子：
    - `create_kwargs`: 工具初始化参数（如 ground_truth）
    - `execute_kwargs`: 工具执行参数
    - `calc_reward_kwargs`: Reward 计算参数
    - `release_kwargs`: 工具清理参数

#### 工具的生命周期

每个工具经历 4 个阶段：

```python
# 伪代码示例
class ToolBase:
    def create(self, **create_kwargs):
        """工具初始化（每个 sample 一次）"""
        pass

    def execute(self, **execute_kwargs):
        """工具执行（每次调用一次）"""
        pass

    def calc_reward(self, trajectory, **calc_reward_kwargs):
        """计算 Reward（rollout 结束后）"""
        pass

    def release(self, **release_kwargs):
        """清理资源"""
        pass
```

### 4.2 GSM8K 工具实现分析

让我们深入分析 GSM8K 的 `calc_gsm8k_reward` 工具实现。

#### 核心 Reward 计算逻辑

```python
# verl/utils/reward_score/gsm8k.py:52-72
def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """GSM8k 的评分函数

    Args:
        solution_str: 模型生成的解答文本
        ground_truth: 正确答案
        method: 'strict' 或 'flexible'
        format_score: 格式正确但答案错误的分数
        score: 答案正确的分数
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0  # 没有找到答案，0 分
    else:
        if answer == ground_truth:
            return score  # 答案正确，满分
        else:
            return format_score  # 格式正确，部分分
```

#### 答案提取逻辑

```python
# verl/utils/reward_score/gsm8k.py:20-49
def extract_solution(solution_str, method="strict"):
    # 优化：只在最后 300 字符中搜索（避免正则性能问题）
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # 严格模式：必须有 `#### 答案` 格式
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # 取最后一个答案
            final_answer = solutions[-1].replace(",", "").replace("$", "")

    elif method == "flexible":
        # 灵活模式：提取最后一个数字
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            pass
        else:
            # 找到最后一个非空数字
            for final_answer in reversed(answer):
                if final_answer not in ["", "."]:
                    break

    return final_answer
```

**两种模式对比：**

| 模式 | 要求 | 优点 | 缺点 |
|------|------|------|------|
| **strict** | 必须有 `#### <answer>` 格式 | 同时测试答案和格式 | 可能因格式错误丢分 |
| **flexible** | 提取最后一个数字 | 更宽容，关注答案本身 | 可能提取错误的数字 |

### 4.3 工具调用流程完整示例

现在我们追踪一个完整的 GSM8K 问题的工具调用流程。

#### Step 1: 数据准备

```python
# examples/data_preprocess/gsm8k_tool_agent_loop.py:73-88
{
    "prompt": [
        {
            "role": "system",
            "content": (
                "You are a math expert. You are given a question and you need to solve it step by step. "
                "Reasoning step by step before any tool call. "
                "You should use the `calc_gsm8k_reward` tool after step by step solving the question, "
                "before generate final answer at least once and refine your answer if necessary. "
                "Put your final answer in the format of `#### <answer>`."
            ),
        },
        {
            "role": "user",
            "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast... Let's think step by step and output the final answer after `####`.",
        },
    ],
}
```

#### Step 2: LLM 第一轮生成（推理）

```python
# Agent Loop 调用 LLM
response_1 = await server_manager.generate(
    request_id=request_id,
    prompt_ids=prompt_ids,  # system + user 的 token ids
    sampling_params={...}
)

# 模型输出（示例）：
"""
Let me solve this step by step:
1. Janet's ducks lay 16 eggs per day
2. She eats 3 for breakfast every morning
3. She bakes muffins for her friends every day with 4 eggs
4. So she uses 3 + 4 = 7 eggs per day
5. She has 16 - 7 = 9 eggs left
6. She sells them at the farmers' market for $2 per egg
7. So she makes 9 * $2 = $18 per day

<tool_call>
{"name": "calc_gsm8k_reward", "arguments": {"solution": "#### 18"}}
</tool_call>
"""
```

#### Step 3: 解析工具调用

```python
# verl/experimental/agent_loop/agent_loop.py 中的 _parse_tool_calls
tool_calls = self._parse_tool_calls(response_text)

# 结果：
[
    {
        "name": "calc_gsm8k_reward",
        "arguments": {"solution": "#### 18"}
    }
]
```

#### Step 4: 执行工具

```python
# 工具执行
tool_result = calc_gsm8k_reward.execute(solution="#### 18")

# 内部调用 compute_score
answer = extract_solution("#### 18", method="strict")  # "18"
if answer == ground_truth:  # "18" == "18"
    return 1.0  # 正确！
else:
    return 0.0

# tool_result = {"score": 1.0, "extracted_answer": "18", "correct": True}
```

#### Step 5: 工具结果注入 trajectory

```python
# 将 tool result 转为 tokens 并添加到响应
tool_result_text = json.dumps(tool_result)
# '{"score": 1.0, "extracted_answer": "18", "correct": true}'

tool_result_ids = tokenizer.encode(tool_result_text)
# [123, 456, 789, ...]  # token ids

# 添加到 response_ids
all_response_ids.extend(tool_result_ids)
all_response_mask.extend([0] * len(tool_result_ids))  # ！工具响应 mask=0
```

**关键点：工具响应的 response_mask 为 0**
- `response_mask=1`: LLM 生成的 token（需要计算 loss）
- `response_mask=0`: 工具响应 token（不计算 loss，视为环境观察）

#### Step 6: LLM 第二轮生成（基于工具反馈）

```python
# 更新 chat history
chat_history.append({
    "role": "tool",
    "content": '{"score": 1.0, "extracted_answer": "18", "correct": true}',
    "tool_call_id": "..."
})

# LLM 继续生成
response_2 = await server_manager.generate(
    request_id=request_id,  # 同一个 request_id！
    prompt_ids=[],  # 空的，因为 chat history 在 server 端
    sampling_params={...}
)

# 模型输出（示例）：
"""
Great! The calculation is correct. Let me finalize the answer.

#### 18
"""
```

#### Step 7: 最终 trajectory 结构

```python
AgentLoopOutput(
    prompt_ids=[101, 102, 103, ...],  # system + user 的 token ids

    response_ids=[
        # 第一轮 LLM 生成
        104, 105, 106, ..., 200,  # "Let me solve..."
        201, 202, 203,            # "<tool_call>..."

        # 工具响应（mask=0）
        300, 301, 302, ..., 350,  # '{"score": 1.0, ...}'

        # 第二轮 LLM 生成
        400, 401, 402, ..., 450,  # "Great! The calculation..."
        451, 452, 453,            # "#### 18"
    ],

    response_mask=[
        # 第一轮 LLM（mask=1）
        1, 1, 1, ..., 1,

        # 工具响应（mask=0）
        0, 0, 0, ..., 0,

        # 第二轮 LLM（mask=1）
        1, 1, 1, ..., 1,
    ],

    num_turns=3  # user, assistant(含tool call), tool, assistant
)
```

### 4.4 Sticky Session 机制详解

在多轮对话中，**同一个 request_id 的所有请求必须发送到同一个 vLLM server**，以利用 **Prefix Caching**。

#### AsyncLLMServerManager 的实现

```python
# verl/experimental/agent_loop/agent_loop.py:57-92
class AsyncLLMServerManager:
    def __init__(self, config, server_handles, max_cache_size=10000):
        self.server_handles = server_handles

        # Least requests load balancing（最少请求数负载均衡）
        self.weighted_serveres = [[0, idx, server] for idx, server in enumerate(server_handles)]
        heapq.heapify(self.weighted_serveres)  # 最小堆

        # LRU cache: request_id -> server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str):
        # 1. 如果 request_id 已经映射到某个 server，返回该 server
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        # 2. 选择请求数最少的 server
        _, _, server = self.weighted_serveres[0]
        self.weighted_serveres[0][0] += 1  # 请求数 +1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])

        # 3. 缓存映射
        self.request_id_to_server[request_id] = server
        return server
```

**工作流程：**

```
Sample 1 (request_id="gsm8k_001"):
    Turn 1 → 选择 Server A（请求数最少）→ 缓存 "gsm8k_001" → Server A
    Turn 2 → 查找缓存 → Server A（复用 KV Cache！）
    Turn 3 → 查找缓存 → Server A

Sample 2 (request_id="gsm8k_002"):
    Turn 1 → 选择 Server B（请求数最少）→ 缓存 "gsm8k_002" → Server B
    Turn 2 → 查找缓存 → Server B
```

**性能提升：**

| 场景 | 无 Sticky Session | 有 Sticky Session |
|------|-------------------|-------------------|
| **KV Cache 命中率** | 0% | ~90%+ |
| **延迟（第 2+ 轮）** | 500ms | 100ms |
| **吞吐量** | 低 | 高 5 倍+ |

---

## 5. 多轮对话训练

### 5.1 Token-based API vs Chat Completion API

这是 Agent RL 中最关键的设计决策之一。

#### 问题：为什么需要 Token-based API？

在 RL 训练中，我们需要：
1. **完整的 trajectory token ids**（用于计算 log_prob）
2. **精确的 response_mask**（区分 LLM 生成 vs 工具响应）

**Chat Completion API 的问题：**

```python
# 使用 OpenAI Chat Completion API（vLLM 兼容）
response = client.chat.completions.create(
    model="Qwen2.5-7B",
    messages=[
        {"role": "system", "content": "You are a math expert..."},
        {"role": "user", "content": "Solve: 2+3=?"},
        {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        {"role": "tool", "content": '{"result": 5}'},
    ]
)

# 返回：
# {
#   "choices": [{"message": {"role": "assistant", "content": "The answer is 5"}}]
# }
```

**问题：**
1. ❌ 无法获取完整的 token ids（只有最后一轮的文本）
2. ❌ 无法区分哪些 token 是 LLM 生成，哪些是工具响应
3. ❌ 无法计算 old_log_prob（因为缺少 token ids）

#### 解决方案：Token-based API

```python
# verl 的 Agent Loop 使用 Token-based API
# verl/experimental/agent_loop/agent_loop.py:94-122
@rollout_trace_op
async def generate(
    self,
    request_id,
    *,
    prompt_ids: list[int],  # ！输入是 token ids
    sampling_params: dict[str, Any],
    image_data: Optional[list[Any]] = None,
    video_data: Optional[list[Any]] = None,
) -> TokenOutput:
    server = self._choose_server(request_id)
    output = await server.generate.remote(
        request_id=uuid4().hex,  # 每次生成用新 request_id
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        ...
    )
    return output  # 返回 token ids + log_probs
```

**TokenOutput 结构：**

```python
class TokenOutput:
    output_token_ids: list[int]  # 生成的 token ids
    logprobs: list[float]        # 每个 token 的 log_prob
    finish_reason: str           # "stop" / "length"
```

### 5.2 多轮对话的 Trajectory 一致性

#### 核心挑战

在多轮对话中，我们需要保证：

```
训练时的 input_ids == Rollout 时的 input_ids
```

否则，`old_log_prob` 和 `new_log_prob` 会不匹配，导致 PPO ratio 计算错误。

#### 完整示例：追踪 Token Flow

**Rollout 阶段：**

```python
# Turn 1: User 提问
prompt_ids = tokenizer.encode([
    {"role": "system", "content": "You are a math expert."},
    {"role": "user", "content": "Solve: 2+3=?"}
])
# [101, 102, ..., 200]  # 假设 100 个 token

# LLM 生成
response_1 = await server.generate(request_id="req_001", prompt_ids=prompt_ids)
# output_token_ids: [201, 202, ..., 250]  # "Let me calculate... <tool_call>..."

# Turn 2: 工具响应
tool_result_text = '{"result": 5}'
tool_result_ids = tokenizer.encode(tool_result_text)
# [300, 301, 302]  # 3 个 token

# 构造 Turn 3 的 prompt
# 方法 1（错误）：重新 apply_chat_template
prompt_ids_turn3 = tokenizer.apply_chat_template([
    {"role": "system", "content": "You are a math expert."},
    {"role": "user", "content": "Solve: 2+3=?"},
    {"role": "assistant", "content": response_1_text},
    {"role": "tool", "content": tool_result_text}
])
# ❌ 问题：token ids 可能和原始不一致！
# 原因：chat_template 可能插入额外的 token（如空格、换行）

# 方法 2（正确）：追加 token ids
prompt_ids_turn3 = (
    prompt_ids +              # [101, ..., 200]
    response_1.output_token_ids +  # [201, ..., 250]
    tool_result_ids           # [300, 301, 302]
)
# [101, ..., 200, 201, ..., 250, 300, 301, 302]  # 353 个 token

# LLM 继续生成
response_2 = await server.generate(request_id="req_001", prompt_ids=prompt_ids_turn3)
# output_token_ids: [400, 401, ..., 420]  # "The answer is 5"
```

**最终 Trajectory：**

```python
AgentLoopOutput(
    prompt_ids=[101, 102, ..., 200],  # 初始 prompt（100 tokens）

    response_ids=[
        201, 202, ..., 250,  # Turn 1: LLM 生成（50 tokens）
        300, 301, 302,       # Turn 2: 工具响应（3 tokens）
        400, 401, ..., 420,  # Turn 3: LLM 生成（21 tokens）
    ],  # 总共 74 个 response tokens

    response_mask=[
        1, 1, ..., 1,  # Turn 1: LLM（50 个 1）
        0, 0, 0,       # Turn 2: 工具（3 个 0）
        1, 1, ..., 1,  # Turn 3: LLM（21 个 1）
    ],
)
```

**训练阶段：**

```python
# Actor Update 时重新前向传播
# verl/trainer/ppo/ray_trainer.py 中的 update_policy

input_ids = torch.cat([batch.prompt_ids, batch.response_ids], dim=1)
# Shape: [batch_size, 100 + 74] = [batch_size, 174]

# 前向传播
outputs = actor_model(input_ids=input_ids, attention_mask=attention_mask)
logits = outputs.logits  # [batch_size, 174, vocab_size]

# 计算 new_log_prob
new_log_prob = compute_log_prob(logits, input_ids, response_mask)

# ✅ 因为 input_ids 完全一致，new_log_prob 和 old_log_prob 可以正确对齐
```

### 5.3 Chat History 管理

在 Agent Loop 中，chat history 有两种管理方式：

#### 方式 1：Server 端管理（vLLM Prefix Caching）

```python
# Turn 1
await server.generate(
    request_id="req_001",
    prompt_ids=[101, ..., 200],  # system + user
    ...
)
# vLLM 缓存 KV Cache for request_id="req_001"

# Turn 2
await server.generate(
    request_id="req_001",  # 同一个 request_id
    prompt_ids=[101, ..., 200, 201, ..., 250, 300, 301, 302],  # 追加 tool response
    ...
)
# vLLM 复用前 200 个 token 的 KV Cache
```

**优点：**
- 自动复用 KV Cache
- 无需手动管理 history

**缺点：**
- 依赖 server 端实现
- 调试困难

#### 方式 2：Client 端管理

```python
class ToolAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        chat_history = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]

        all_response_ids = []
        all_response_mask = []

        for turn in range(max_turns):
            # Apply chat template
            prompt_ids = await self.apply_chat_template(
                messages=chat_history,
                remove_system_prompt=(turn > 0)  # 第 2+ 轮移除 system prompt
            )

            # Generate
            response = await self.server_manager.generate(
                request_id=self._get_request_id(),
                prompt_ids=prompt_ids,
                ...
            )

            # Update history
            chat_history.append({
                "role": "assistant",
                "content": self.tokenizer.decode(response.output_token_ids)
            })

            # 追加 response
            all_response_ids.extend(response.output_token_ids)
            all_response_mask.extend([1] * len(response.output_token_ids))

            # ... 工具调用逻辑 ...
```

**优点：**
- 灵活，可自定义
- 易于调试

**缺点：**
- 需要手动管理 history
- 可能引入 token 不一致问题

### 5.4 response_mask 的关键作用

`response_mask` 在训练中有 3 个关键作用：

#### 作用 1：Loss 计算

```python
# verl/trainer/ppo/ray_trainer.py 中的 compute_loss

# 只对 LLM 生成的 token 计算 loss
loss = -advantages * log_ratio  # [batch_size, response_length]
loss = (loss * response_mask).sum() / response_mask.sum()
```

**原因：**
- 工具响应是环境给的，不是 LLM 生成的
- 对工具响应计算 loss 没有意义（会引入噪声）

#### 作用 2：Advantage 广播（GRPO）

```python
# verl/trainer/ppo/core_algos.py:266-330

# Reward 只在最后一个 token
# [batch_size, response_length] → 只有最后一个位置非 0

# GRPO 需要广播到所有 LLM token
advantages = advantages.unsqueeze(-1)  # [batch_size, group_size, 1]
advantages = advantages * response_mask  # 只保留 LLM token
```

#### 作用 3：Metrics 计算

```python
# 计算 LLM 生成的平均长度
llm_lengths = response_mask.sum(dim=1)  # [batch_size]
avg_llm_length = llm_lengths.float().mean()

# 计算 tool token 比例
tool_ratio = (1 - response_mask).sum() / response_mask.numel()
```

---

## 6. 完整训练流程追踪

### 6.1 端到端示例：GSM8K Tool Agent

让我们追踪一个完整的 batch 从 Rollout 到 Training 的全流程。

#### Step 1: 数据加载

```python
# 训练开始，从 Parquet 加载数据
dataset = RLHFDataset.load("~/data/gsm8k/train.parquet")

# Batch size = 2
batch = dataset.sample(2)
# batch.non_tensor_batch["prompt"]:
# [
#   [{"role": "system", ...}, {"role": "user", "content": "问题 1"}],
#   [{"role": "system", ...}, {"role": "user", "content": "问题 2"}]
# ]
# batch.non_tensor_batch["agent_name"]: ["tool_agent", "tool_agent"]
# batch.non_tensor_batch["extra_info"]:
# [
#   {"tools_kwargs": {"calc_gsm8k_reward": {...}}},
#   {"tools_kwargs": {"calc_gsm8k_reward": {...}}}
# ]
```

#### Step 2: Rollout（生成 trajectories）

```python
# AgentLoopManager.generate_sequences()

# 分发到 AgentLoopWorker
outputs = await asyncio.gather(
    worker_1.generate_sequences(batch[0]),  # Sample 1
    worker_2.generate_sequences(batch[1]),  # Sample 2
)

# 每个 worker 运行 ToolAgentLoop
# Sample 1 的 trajectory:
{
    "prompt_ids": [101, ..., 200],  # 100 tokens
    "response_ids": [
        201, ..., 250,  # Turn 1: LLM (50 tokens)
        300, 301, 302,  # Tool response (3 tokens)
        400, ..., 420,  # Turn 2: LLM (21 tokens)
    ],  # 74 tokens
    "response_mask": [
        1, ..., 1,  # 50 个 1
        0, 0, 0,    # 3 个 0
        1, ..., 1,  # 21 个 1
    ],
    "rollout_log_probs": [0.1, 0.2, ..., 0.3],  # 74 个值（对应 response_ids）
}
```

#### Step 3: Reward 计算

```python
# RewardManager.compute_reward()

# 对于 GSM8K，Reward 在 Agent Loop 内部已计算
# calc_gsm8k_reward.execute() 返回 {"score": 1.0, "correct": True}

# Reward 放置在最后一个 token
rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
rm_scores[0, response_length[0]] = 1.0  # Sample 1: 正确
rm_scores[1, response_length[1]] = 0.0  # Sample 2: 错误

# Shape: [2, 74]
# Sample 1: [0, 0, ..., 0, 1.0, 0, ..., 0]  # 最后一个 LLM token = 1.0
# Sample 2: [0, 0, ..., 0, 0.0, 0, ..., 0]
```

#### Step 4: Reference Log Prob 计算

```python
# RefModelWorker.forward_step()

# 使用 Ref Model 重新计算 log_prob
ref_log_probs = ref_model(input_ids=input_ids, attention_mask=attention_mask)

# Shape: [2, 74]
# 这些 log_prob 用于 KL 惩罚
```

#### Step 5: Value 估计（仅 PPO）

如果使用 PPO 算法，需要 Critic Model：

```python
# CriticModelWorker.forward_step()

values = critic_model(input_ids=input_ids, attention_mask=attention_mask)

# Shape: [2, 74]
# 每个 token 的价值估计
```

#### Step 6: Advantage 计算

**GRPO 算法：**

```python
# verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage

# Step 1: Group samples
# Batch [Sample 1, Sample 2] → Group [Sample 1, Sample 2]
# (假设 group_size=2)

# Step 2: KL penalty
kl_penalty = (rollout_log_probs - ref_log_probs).sum(dim=-1)  # [2]
# Sample 1: 0.5
# Sample 2: 0.6

kl_rewards = rm_scores.sum(dim=-1) - beta * kl_penalty
# Sample 1: 1.0 - 0.01 * 0.5 = 0.995
# Sample 2: 0.0 - 0.01 * 0.6 = -0.006

# Step 3: Group baseline
group_mean = kl_rewards.mean()  # (0.995 - 0.006) / 2 = 0.4945

# Step 4: Advantage
advantages = kl_rewards - group_mean
# Sample 1: 0.995 - 0.4945 = 0.5005
# Sample 2: -0.006 - 0.4945 = -0.5005

# Step 5: 广播到所有 token
advantages = advantages.unsqueeze(-1) * response_mask
# Shape: [2, 74]
# Sample 1: [0.5005, 0.5005, ..., 0.5005, 0, 0, 0, 0.5005, ...]  # mask=0 的位置为 0
```

**PPO 算法：**

```python
# verl/trainer/ppo/core_algos.py:compute_gae

# GAE 递归计算
advantages = []
gae = 0
for t in reversed(range(response_length)):
    delta = rewards[t] + gamma * values[t+1] - values[t]
    gae = delta + gamma * lam * gae
    advantages.insert(0, gae)

# Shape: [2, 74]
```

#### Step 7: Actor Update

```python
# ActorModelWorker.update_policy()

# 前向传播
outputs = actor_model(input_ids=input_ids, attention_mask=attention_mask)
logits = outputs.logits  # [2, 174, vocab_size]

# 计算 new_log_prob
new_log_probs = compute_log_prob(logits, response_ids, response_mask)
# Shape: [2, 74]

# PPO ratio
ratio = torch.exp(new_log_probs - old_log_probs)
# Shape: [2, 74]

# Clipped objective
loss_1 = ratio * advantages
loss_2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
loss = -torch.min(loss_1, loss_2)

# 只对 LLM token 计算 loss
loss = (loss * response_mask).sum() / response_mask.sum()

# 反向传播
loss.backward()
optimizer.step()
```

#### Step 8: Critic Update（仅 PPO）

```python
# CriticModelWorker.update_critic()

# 计算 TD target
returns = advantages + values  # [2, 74]

# Critic loss
new_values = critic_model(input_ids=input_ids, attention_mask=attention_mask)
critic_loss = F.mse_loss(new_values, returns, reduction='none')

# 只对 LLM token 计算 loss
critic_loss = (critic_loss * response_mask).sum() / response_mask.sum()

# 反向传播
critic_loss.backward()
critic_optimizer.step()
```

### 6.2 Metrics 收集和可视化

#### AgentLoopWorker 收集的 Metrics

```python
# verl/experimental/agent_loop/agent_loop.py:125-154

class AgentLoopMetrics(BaseModel):
    generate_sequences: float = 0.0  # LLM 生成总耗时
    tool_calls: float = 0.0          # 工具调用总耗时
    num_preempted: int = -1          # 被抢占次数（vLLM）
```

**示例输出：**

```python
metrics = [
    {"generate_sequences": 1.2, "tool_calls": 0.3, "num_preempted": 0},  # Sample 1
    {"generate_sequences": 2.5, "tool_calls": 0.5, "num_preempted": 1},  # Sample 2
]

# AgentLoopManager 汇总
timing = {
    "agent_loop/generate_sequences/min": 1.2,
    "agent_loop/generate_sequences/max": 2.5,
    "agent_loop/generate_sequences/mean": 1.85,
    "agent_loop/tool_calls/min": 0.3,
    "agent_loop/tool_calls/max": 0.5,
    "agent_loop/tool_calls/mean": 0.4,
    "agent_loop/num_preempted/mean": 0.5,
    "agent_loop/slowest/generate_sequences": 2.5,  # 最慢样本
    "agent_loop/slowest/tool_calls": 0.5,
    "agent_loop/slowest/prompt_length": 100,
    "agent_loop/slowest/response_length": 74,
}
```

#### RayPPOTrainer 收集的 Metrics

```python
# verl/trainer/ppo/ray_trainer.py 中的 fit()

metrics = {
    # Rollout
    "throughput/rollout": batch_size / rollout_time,
    "time/rollout": rollout_time,

    # Reward
    "reward/mean": rewards.mean().item(),
    "reward/max": rewards.max().item(),
    "reward/min": rewards.min().item(),

    # Advantage
    "advantage/mean": advantages.mean().item(),
    "advantage/std": advantages.std().item(),

    # Actor
    "policy/approx_kl": approx_kl.mean().item(),
    "policy/ratio/mean": ratio.mean().item(),
    "policy/ratio/max": ratio.max().item(),
    "policy/clipfrac": (torch.abs(ratio - 1) > eps).float().mean().item(),
    "loss/actor": actor_loss.item(),

    # Critic（仅 PPO）
    "loss/critic": critic_loss.item(),

    # Agent Loop（从 AgentLoopManager 传递）
    **agent_loop_timing,
}

# TensorBoard 记录
logger.log_metrics(metrics, step=global_step)
```

#### TensorBoard 可视化

```bash
tensorboard --logdir ~/experiments/gsm8k_tool_agent/logs
```

**关键指标：**

1. **Reward Curve**
   - `reward/mean`: 平均 Reward（应该逐渐上升）
   - `reward/max`: 最大 Reward（1.0 表示有样本完全正确）

2. **Policy Metrics**
   - `policy/approx_kl`: 近似 KL 散度（应该 < 0.1）
   - `policy/ratio/mean`: PPO ratio 均值（应该接近 1.0）
   - `policy/clipfrac`: Clipping 比例（10-30% 正常）

3. **Agent Loop Metrics**
   - `agent_loop/generate_sequences/mean`: 平均生成耗时
   - `agent_loop/tool_calls/mean`: 平均工具调用耗时
   - `agent_loop/num_preempted/mean`: 平均抢占次数

---

## 7. 调试技巧

### 7.1 常见问题诊断

#### 问题 1：Reward 始终为 0

**症状：**
```python
reward/mean: 0.0
reward/max: 0.0
reward/min: 0.0
```

**可能原因：**

1. **工具未正确调用**
   ```python
   # 检查 LLM 输出
   print(response_text)
   # 应该看到: <tool_call>{"name": "calc_gsm8k_reward", ...}</tool_call>

   # 如果没有，可能是：
   # - System prompt 没有指示使用工具
   # - 模型未经过工具调用训练
   ```

2. **答案格式错误**
   ```python
   # 检查答案提取
   from verl.utils.reward_score.gsm8k import extract_solution

   solution_str = "The answer is 18"  # ❌ 没有 #### 格式
   answer = extract_solution(solution_str, method="strict")
   print(answer)  # None → Reward = 0

   solution_str = "#### 18"  # ✅ 正确格式
   answer = extract_solution(solution_str, method="strict")
   print(answer)  # "18" → Reward = 1.0（如果正确）
   ```

3. **ground_truth 不匹配**
   ```python
   # 检查数据
   print(batch.non_tensor_batch["extra_info"][0])
   # {"tools_kwargs": {"calc_gsm8k_reward": {"create_kwargs": {"ground_truth": "18"}}}}

   # 检查模型输出
   print(extracted_answer)  # "18.0" ≠ "18" → Reward = 0
   # 解决：统一格式（去除小数点）
   ```

**调试脚本：**

```python
# debug_reward.py
import re
from verl.utils.reward_score.gsm8k import compute_score, extract_solution

# 测试答案提取
test_cases = [
    "The answer is #### 18",  # ✅ 正确
    "#### 18",                # ✅ 正确
    "The answer is 18",       # ❌ 无格式
    "Let's calculate: #### 18.0",  # ⚠️ 18.0 vs 18
]

ground_truth = "18"

for case in test_cases:
    answer = extract_solution(case, method="strict")
    score = compute_score(case, ground_truth, method="strict")
    print(f"Input: {case!r}")
    print(f"  Extracted: {answer}")
    print(f"  Score: {score}\n")
```

#### 问题 2：PPO ratio 爆炸

**症状：**
```python
policy/ratio/mean: 5.2  # ❌ 应该接近 1.0
policy/ratio/max: 20.3
policy/clipfrac: 0.85    # ❌ 应该 < 0.5
```

**可能原因：**

1. **old_log_prob 和 new_log_prob 不对齐**
   ```python
   # 检查 token ids 一致性
   # rollout_input_ids vs training_input_ids

   assert torch.equal(
       batch["input_ids"],
       torch.cat([batch["prompts"], batch["responses"]], dim=1)
   ), "Input IDs mismatch!"
   ```

2. **学习率过大**
   ```python
   # 降低学习率
   # config/actor.yaml
   actor:
     optim:
       lr: 1e-6  # 从 1e-5 降低到 1e-6
   ```

3. **Clipping 阈值过大**
   ```python
   # config/ppo_trainer.yaml
   algorithm:
     clip_ratio: 0.1  # 从 0.2 降低到 0.1
   ```

**调试脚本：**

```python
# debug_ppo_ratio.py

# 检查 log_prob 分布
import torch

old_log_probs = batch["rollout_log_probs"]  # Rollout 时的
new_log_probs = compute_log_prob(logits, response_ids, response_mask)

# Ratio
ratio = torch.exp(new_log_probs - old_log_probs)

print(f"old_log_probs: mean={old_log_probs.mean():.3f}, std={old_log_probs.std():.3f}")
print(f"new_log_probs: mean={new_log_probs.mean():.3f}, std={new_log_probs.std():.3f}")
print(f"ratio: mean={ratio.mean():.3f}, max={ratio.max():.3f}")

# 检查异常值
outliers = (ratio > 2.0) | (ratio < 0.5)
if outliers.any():
    print(f"Found {outliers.sum()} outliers!")
    print(f"Positions: {outliers.nonzero()}")
```

#### 问题 3：Tool 响应 Token 参与了 Loss 计算

**症状：**
训练不稳定，loss 震荡。

**原因：**
`response_mask` 错误，工具响应被当作 LLM 生成计算了 loss。

**检查：**

```python
# debug_response_mask.py

# 检查 response_mask
response_ids = batch["responses"][0]  # 第一个样本
response_mask = batch["response_mask"][0]

print("Response IDs:", response_ids)
print("Response Mask:", response_mask)

# 解码查看
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
for i, (token_id, mask) in enumerate(zip(response_ids, response_mask)):
    if token_id == 0:  # padding
        break
    token_text = tokenizer.decode([token_id])
    print(f"{i:3d}: {token_id:5d} mask={mask} {token_text!r}")

# 预期：工具响应部分 mask=0
# Turn 1 LLM: mask=1
# Tool response: mask=0  # ← 重点检查这部分
# Turn 2 LLM: mask=1
```

**修复：**

```python
# agent_loop.py 中确保正确设置 mask

# LLM 生成
all_response_ids.extend(response_ids)
all_response_mask.extend([1] * len(response_ids))  # ✅ LLM = 1

# 工具响应
tool_result_ids = tokenizer.encode(tool_result_text)
all_response_ids.extend(tool_result_ids)
all_response_mask.extend([0] * len(tool_result_ids))  # ✅ Tool = 0
```

### 7.2 Tracing 和 Logging

#### RolloutTraceConfig

verl 提供了完整的 Trace 系统用于调试。

```python
# config/ppo_trainer.yaml
actor_rollout_ref:
  rollout:
    trace:
      backend: "mlflow"  # 或 "simple"
      token2text: true   # 将 token ids 转换为文本
      max_samples_per_step_per_worker: 5  # 每个 step 只 trace 5 个样本
```

**使用 MLflow Trace：**

```python
# verl/utils/rollout_trace.py

# 在 Agent Loop 中自动记录
with rollout_trace_attr(
    step=global_step,
    sample_index=i,
    rollout_n=0,
    validate=False,
    name="agent_loop",
    trace=True,
):
    # 所有操作都会被记录
    output = await agent_loop.run(sampling_params, **kwargs)
```

**查看 Trace：**

```bash
mlflow ui --backend-store-uri ~/experiments/gsm8k_tool_agent/mlruns
```

在 MLflow UI 中可以看到：
- 每个样本的完整 trajectory
- 每次工具调用的输入输出
- 每轮 LLM 生成的耗时

#### 自定义 Logging

```python
# 在 Agent Loop 中添加 logging

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ToolAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        logger.info(f"Starting Agent Loop for sample {kwargs.get('index')}")

        for turn in range(max_turns):
            logger.debug(f"Turn {turn}: Generating...")
            response = await self.server_manager.generate(...)
            logger.debug(f"Turn {turn}: Generated {len(response.output_token_ids)} tokens")

            tool_calls = self._parse_tool_calls(response_text)
            if tool_calls:
                logger.info(f"Turn {turn}: Calling {len(tool_calls)} tools")
                for tool_call in tool_calls:
                    logger.debug(f"  Tool: {tool_call['name']}, Args: {tool_call['arguments']}")
                    tool_result = ...
                    logger.debug(f"  Result: {tool_result}")

        logger.info(f"Finished Agent Loop: {len(all_response_ids)} total response tokens")
        return AgentLoopOutput(...)
```

### 7.3 单元测试

**测试 Tool 逻辑：**

```python
# tests/test_gsm8k_tool.py
import pytest
from verl.utils.reward_score.gsm8k import compute_score, extract_solution

def test_extract_solution_strict():
    assert extract_solution("#### 18", method="strict") == "18"
    assert extract_solution("The answer is #### 18", method="strict") == "18"
    assert extract_solution("No answer here", method="strict") is None

def test_compute_score():
    assert compute_score("#### 18", ground_truth="18", method="strict") == 1.0
    assert compute_score("#### 20", ground_truth="18", method="strict") == 0.0
    assert compute_score("No answer", ground_truth="18", method="strict") == 0.0
```

**测试 Agent Loop：**

```python
# tests/test_agent_loop.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_tool_agent_loop():
    # Mock server
    mock_server = AsyncMock()
    mock_server.generate.return_value = TokenOutput(
        output_token_ids=[201, 202, 203],
        logprobs=[0.1, 0.2, 0.3],
        finish_reason="stop"
    )

    # Mock tools
    mock_tool = MagicMock()
    mock_tool.execute.return_value = {"score": 1.0, "correct": True}

    # Create Agent Loop
    agent_loop = ToolAgentLoop(
        server_manager=mock_server,
        tools={"calc_gsm8k_reward": mock_tool},
        ...
    )

    # Run
    output = await agent_loop.run(
        sampling_params={},
        prompt=[...],
        ...
    )

    # Assertions
    assert len(output.response_ids) > 0
    assert len(output.response_mask) == len(output.response_ids)
    assert 0 in output.response_mask  # Tool response exists
    assert 1 in output.response_mask  # LLM generation exists
```

---

## 8. 最佳实践

### 8.1 System Prompt 设计

**好的 System Prompt：**

```python
# examples/data_preprocess/gsm8k_tool_agent_loop.py:76-82
{
    "role": "system",
    "content": (
        "You are a math expert. You are given a question and you need to solve it step by step. "
        # ✅ 明确指示：先推理
        "Reasoning step by step before any tool call. "
        # ✅ 明确指示：何时使用工具
        "You should use the `calc_gsm8k_reward` tool after step by step solving the question, "
        # ✅ 明确指示：可以多次调用
        "before generate final answer at least once and refine your answer if necessary. "
        # ✅ 明确指示：输出格式
        "Put your final answer in the format of `#### <answer>`."
    ),
}
```

**关键要素：**
1. **角色定义**："You are a math expert"
2. **任务描述**："solve it step by step"
3. **工具使用时机**："after step by step solving"
4. **输出格式**："`#### <answer>`"
5. **迭代优化**："refine your answer if necessary"

**避免的错误：**

```python
# ❌ 错误示例 1：过于模糊
"You are a helpful assistant. Answer the question."

# ❌ 错误示例 2：没有指示工具使用
"You are a math expert. Solve the problem step by step."

# ❌ 错误示例 3：没有输出格式要求
"Use the tool to check your answer."
```

### 8.2 工具设计原则

#### 原则 1：工具应该是确定性的

```python
# ✅ 好的工具
def calc_gsm8k_reward(solution: str, ground_truth: str) -> dict:
    answer = extract_solution(solution)
    return {
        "score": 1.0 if answer == ground_truth else 0.0,
        "extracted_answer": answer,
        "correct": answer == ground_truth
    }

# ❌ 不好的工具（非确定性）
def calc_reward_with_llm(solution: str) -> dict:
    # 使用另一个 LLM 评分（可能每次不同）
    score = llm_judge(solution)  # ← 非确定性
    return {"score": score}
```

#### 原则 2：工具应该提供详细反馈

```python
# ✅ 好的工具
{
    "score": 0.0,
    "extracted_answer": "20",
    "correct": False,
    "ground_truth": "18",
    "error_type": "calculation_error"
}

# ❌ 不好的工具
{
    "score": 0.0  # 没有告诉模型哪里错了
}
```

#### 原则 3：工具应该快速执行

```python
# ✅ 好的工具（< 100ms）
def extract_solution(solution_str):
    return re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)

# ❌ 不好的工具（> 1s）
def complex_verification(solution_str):
    # 调用外部 API
    result = requests.post("https://api.example.com/verify", ...)
    return result.json()
```

### 8.3 多轮对话策略

#### 策略 1：限制最大轮数

```python
max_turns = 5  # 防止无限循环

for turn in range(max_turns):
    response = await server.generate(...)
    tool_calls = parse_tool_calls(response)

    if not tool_calls:
        break  # 没有工具调用，结束

    # 执行工具...
```

#### 策略 2：Early Stopping

```python
for turn in range(max_turns):
    response = await server.generate(...)

    # 检查是否已经生成了最终答案
    if "####" in response_text:
        # 已有最终答案，可以提前结束
        break

    tool_calls = parse_tool_calls(response)
    ...
```

#### 策略 3：工具调用预算

```python
max_tool_calls = 3  # 最多调用 3 次工具
tool_call_count = 0

for turn in range(max_turns):
    response = await server.generate(...)
    tool_calls = parse_tool_calls(response)

    if tool_calls:
        tool_call_count += len(tool_calls)
        if tool_call_count > max_tool_calls:
            logger.warning("Exceeded tool call budget")
            break
    ...
```

### 8.4 Reward Shaping for Agent RL

#### 技巧 1：中间步骤奖励

```python
def calc_gsm8k_reward_with_steps(solution: str, ground_truth: str):
    score = 0.0

    # 基础分：答案正确
    answer = extract_solution(solution)
    if answer == ground_truth:
        score += 1.0

    # 额外分：使用了工具
    if "<tool_call>" in solution:
        score += 0.1

    # 额外分：推理步骤数量
    steps = solution.count("Step")
    score += min(steps * 0.05, 0.3)  # 最多 +0.3

    return {"score": score}
```

#### 技巧 2：格式化奖励

```python
def calc_reward_with_format(solution: str, ground_truth: str):
    # 答案正确：1.0
    # 格式正确但答案错误：0.2
    # 格式错误：0.0
    return compute_score(
        solution,
        ground_truth,
        method="strict",
        format_score=0.2,  # ← 格式分
        score=1.0
    )
```

#### 技巧 3：长度惩罚

```python
def calc_reward_with_length_penalty(solution: str, ground_truth: str, max_length=500):
    base_score = compute_score(solution, ground_truth)

    # 过长惩罚
    if len(solution) > max_length:
        penalty = (len(solution) - max_length) / 1000
        base_score -= penalty

    return {"score": max(0.0, base_score)}
```

### 8.5 超参数调优

#### 关键超参数

| 参数 | 作用 | 推荐值（GSM8K）| 调优建议 |
|------|------|----------------|----------|
| `learning_rate` | Actor 学习率 | 1e-6 ~ 1e-5 | 从小开始（1e-6） |
| `clip_ratio` | PPO Clipping | 0.1 ~ 0.2 | GRPO 可以更大（0.3） |
| `beta` | KL 惩罚系数 | 0.01 ~ 0.05 | 根据 approx_kl 调整 |
| `gamma` | Discount factor | 1.0 | Episodic 任务用 1.0 |
| `lam` | GAE lambda | 0.95 | PPO 专用 |
| `batch_size` | Batch 大小 | 128 ~ 512 | 越大越稳定 |
| `max_turns` | 最大轮数 | 3 ~ 5 | 避免过长 |

#### 调优流程

```
1. 固定其他参数，调 learning_rate
   - 观察 policy/approx_kl
   - 目标：approx_kl < 0.1

2. 调整 clip_ratio
   - 观察 policy/clipfrac
   - 目标：clipfrac = 0.1 ~ 0.3

3. 调整 beta（KL 惩罚）
   - 观察 reward/mean
   - 目标：平衡 reward 和 KL

4. 增大 batch_size（如果资源允许）
   - 提升稳定性
```

---

## 9. 总结

### 9.1 核心要点回顾

1. **Agent Loop 架构**
   - AsyncLLMServerManager：负载均衡 + Sticky Session
   - AgentLoopWorker：并发执行多个 Agent Loop
   - AgentLoopManager：协调 Workers 和 LLM Servers

2. **工具调用机制**
   - 工具生命周期：create → execute → calc_reward → release
   - response_mask 区分 LLM token (1) 和 Tool token (0)
   - 工具响应不参与 loss 计算

3. **多轮对话**
   - Token-based API 保证 trajectory 一致性
   - Sticky Session 提升 KV Cache 命中率
   - Chat history 可以 Server 端或 Client 端管理

4. **训练流程**
   - Rollout → Reward → Ref → Value → Advantage → Actor Update → Critic Update
   - GRPO：无 Critic，Group Baseline
   - PPO：有 Critic，GAE Advantage

5. **调试和最佳实践**
   - 检查 response_mask 正确性
   - 使用 Trace 系统调试
   - System Prompt 明确指示工具使用
   - Reward Shaping 提升训练效果

### 9.2 进阶方向

1. **自定义 Agent Loop**
   - 实现复杂的多工具调用
   - 支持并行工具执行
   - 集成外部环境（如代码执行沙箱）

2. **工具学习**
   - Few-shot 工具使用示例
   - 工具选择策略优化
   - 工具组合优化

3. **高级 Reward**
   - 过程奖励（Process Reward Model）
   - 对比学习（Contrastive Learning）
   - 自我修正奖励

4. **分布式优化**
   - 多节点 Agent Loop
   - 异步 Reward 计算
   - Pipeline 并行

---

**🎉 恭喜！你已经完成了 Agent Loop 的深度学习！**

下一步建议：
1. 实践：运行 GSM8K Tool Agent 训练
2. 实验：修改 System Prompt 观察效果变化
3. 扩展：实现自己的 Agent Loop 和工具

继续加油！🚀
