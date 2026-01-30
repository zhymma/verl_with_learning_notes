# Generation - 批量文本生成工具

[English](README.md) | 简体中文

## 概述

Generation 示例展示了如何使用 veRL 进行大规模批量文本生成（推理）。这是一个独立的生成工具，不涉及训练，专门用于从预训练或微调后的模型批量生成文本。

本工具基于 veRL 的分布式推理引擎，支持多节点多 GPU 的高效生成，适合大规模数据集的批量处理。

## 主要特点

- **批量生成**: 高效处理大规模数据集的批量推理任务
- **分布式推理**: 支持多节点多 GPU 的分布式生成
- **灵活配置**: 支持各种采样参数（temperature, top_k, top_p 等）
- **多推理引擎**: 支持 vLLM 和 HuggingFace Transformers
- **数据格式灵活**: 输入输出使用 Parquet 格式，易于处理大规模数据
- **张量并行**: 支持模型张量并行以处理大型模型

## 适用场景

Generation 工具适用于以下场景：

1. **数据集增强**: 为训练数据生成多个响应
2. **模型评估**: 在测试集上批量生成结果进行评估
3. **A/B 测试**: 对比不同模型或参数设置的生成效果
4. **生成合成数据**: 为下游任务生成大量合成数据
5. **SFT 数据准备**: 为监督微调准备高质量的生成数据
6. **RLHF 数据准备**: 为强化学习准备初始化数据
7. **推理性能测试**: 测试模型在不同配置下的推理性能

## 快速开始

### 环境准备

1. 安装 veRL 及其依赖：

```bash
# 安装 veRL 和 vLLM
pip install -e .[test,vllm]

# 或使用 SGLang
pip install -e .[test,sglang]
```

2. 准备数据：

```bash
# 数据应该是 Parquet 格式，包含提示文本
# 示例数据结构:
# {
#   "prompt": "请解答以下数学问题: ...",
#   "other_fields": "..."  # 可选的其他字段
# }
```

### 基本运行

#### 单节点运行

使用单节点 8 GPU 进行生成：

```bash
# 修改脚本中的路径
data_path=$HOME/data/gsm8k/test.parquet
save_path=$HOME/data/gsm8k/deepseek_v2_lite_gen_test.parquet
model_path=deepseek-ai/deepseek-llm-7b-chat

# 运行
bash examples/generation/run_deepseek_v2_lite_math.sh
```

#### 多节点运行

使用多节点进行大规模生成：

```bash
# 修改脚本中的路径和节点配置
data_path=$HOME/data/rlhf/gsm8k/test.parquet
save_path=$HOME/data/rlhf/math/deepseek_v2_lite_gen_test.parquet
model_path=deepseek-ai/deepseek-llm-7b-chat

# 运行 (2 节点, 16 GPU)
bash examples/generation/run_deepseek7b_mutli_node.sh
```

## 详细配置说明

### 1. 训练器配置

```bash
trainer.nnodes=1                    # 节点数量
trainer.n_gpus_per_node=8           # 每个节点的 GPU 数量
trainer.device=cuda                 # 设备类型
```

### 2. 数据配置

```bash
data.path=$data_path                # 输入数据路径（Parquet 格式）
data.prompt_key=prompt              # Parquet 中提示文本的字段名
data.n_samples=1                    # 每个提示生成的样本数
data.output_path=$save_path         # 输出数据路径（Parquet 格式）
data.batch_size=128                 # 批次大小
```

**输入数据格式示例**:
```python
# input.parquet 应包含类似的结构
{
    "prompt": "What is 2+2?",
    "id": 123,
    # ... 其他字段
}
```

**输出数据格式**:
```python
# output.parquet 将包含
{
    "prompt": "What is 2+2?",
    "generated_text": "2+2 equals 4.",
    "id": 123,
    # ... 输入的其他字段会被保留
}
```

### 3. 模型配置

```bash
model.path=$model_path              # 模型路径（本地或 HuggingFace）
+model.trust_remote_code=True       # 信任远程代码（某些模型需要）
model.external_lib=null             # 外部库（如特殊模型实现）
```

### 4. 推理引擎配置 (vLLM)

```bash
rollout.name=vllm                           # 推理引擎名称
rollout.mode=async                          # 异步模式
rollout.temperature=1.0                     # 采样温度
rollout.top_k=50                            # Top-K 采样 (-1 for vLLM)
rollout.top_p=0.7                           # Top-P (nucleus) 采样
rollout.prompt_length=2048                  # 最大提示长度
rollout.response_length=1024                # 最大响应长度
rollout.tensor_model_parallel_size=2        # 张量并行大小
rollout.gpu_memory_utilization=0.8          # GPU 内存利用率
rollout.dtype=bfloat16                      # 数据类型
rollout.ignore_eos=False                    # 是否忽略 EOS token
rollout.enforce_eager=True                  # 强制 eager 模式
rollout.max_num_batched_tokens=8192         # 最大批次 token 数
rollout.max_num_seqs=1024                   # 最大序列数
rollout.enable_chunked_prefill=True         # 启用分块预填充
```

### 5. 采样参数详解

#### Temperature (温度)

```bash
rollout.temperature=1.0                     # 默认: 1.0
```

- **0.0**: 贪心解码（deterministic）
- **0.1-0.7**: 更确定性的输出，适合事实性任务
- **0.8-1.0**: 平衡的创造性和准确性
- **1.0+**: 更随机、更有创造性的输出

#### Top-K Sampling

```bash
rollout.top_k=50                            # 默认: 50
```

- 从概率最高的 K 个 token 中采样
- `top_k=-1`: 不使用 top-k（vLLM 推荐设置）
- `top_k=0`: 不使用 top-k（HF Transformers）

#### Top-P (Nucleus Sampling)

```bash
rollout.top_p=0.7                           # 默认: 0.7
```

- 从累积概率达到 P 的最小 token 集合中采样
- `top_p=0.9`: 更多样化的输出
- `top_p=0.5`: 更保守的输出

### 6. 并行策略配置

#### 张量并行 (Tensor Parallelism)

```bash
rollout.tensor_model_parallel_size=2        # TP 大小
```

**选择指南**:
- **7B 模型**: TP=1 或 2
- **13B 模型**: TP=2 或 4
- **30B+ 模型**: TP=4 或 8
- **70B+ 模型**: TP=8 或 16

**示例**:
```bash
# 单节点 8 GPU, 7B 模型
rollout.tensor_model_parallel_size=2        # 使用 4 个数据并行副本

# 2 节点 16 GPU, 70B 模型
rollout.tensor_model_parallel_size=16       # 使用完整的 TP
```

#### 数据并行

数据并行是自动的：
```
data_parallel_size = total_gpus / tensor_model_parallel_size
```

例如:
- 8 GPU, TP=2 → DP=4
- 16 GPU, TP=8 → DP=2

## 运行示例

### 示例 1: 单节点基本生成

生成 GSM8K 测试集的响应：

```bash
data_path=$HOME/data/gsm8k/test.parquet
save_path=$HOME/data/gsm8k/model_gen_test.parquet
model_path=Qwen/Qwen2-7B-Instruct

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    rollout.temperature=1.0 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8
```

### 示例 2: 多样本生成（用于 RLHF）

为每个提示生成多个响应：

```bash
python3 -m verl.trainer.main_generation \
    data.n_samples=8 \                      # 每个提示生成 8 个响应
    rollout.temperature=1.0 \               # 使用较高温度增加多样性
    rollout.top_p=0.9 \                     # 更宽松的 top-p
    # ... 其他配置
```

### 示例 3: 确定性生成（用于评估）

生成确定性的输出用于评估：

```bash
python3 -m verl.trainer.main_generation \
    data.n_samples=1 \
    rollout.temperature=0.0 \               # 贪心解码
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    # ... 其他配置
```

### 示例 4: 多节点大规模生成

使用 2 节点 16 GPU 处理大规模数据：

```bash
python3 -m verl.trainer.main_generation \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    data.batch_size=256 \                   # 增加批次大小
    rollout.tensor_model_parallel_size=4 \  # 使用 TP=4
    rollout.gpu_memory_utilization=0.9 \    # 更高的内存利用率
    # ... 其他配置
```

### 示例 5: 大模型生成（70B+）

处理大型模型：

```bash
model_path=meta-llama/Llama-3-70B-Instruct

python3 -m verl.trainer.main_generation \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    model.path=$model_path \
    rollout.tensor_model_parallel_size=16 \  # 使用所有 16 GPU 进行 TP
    rollout.gpu_memory_utilization=0.95 \    # 最大化内存使用
    rollout.prompt_length=4096 \             # 支持更长的上下文
    rollout.response_length=2048 \
    rollout.dtype=bfloat16 \
    rollout.enforce_eager=True \
    # ... 其他配置
```

### 示例 6: 使用 HuggingFace Transformers

使用 HF 后端而非 vLLM：

```bash
python3 -m verl.trainer.main_generation \
    rollout.name=hf \                       # 使用 HF 后端
    rollout.do_sample=True \                # 启用采样
    rollout.top_k=0 \                       # HF 的 top-k 格式
    # ... 其他配置
```

## 性能优化

### 1. 内存优化

```bash
# 调整 GPU 内存利用率
rollout.gpu_memory_utilization=0.8          # 保守: 0.7-0.8
rollout.gpu_memory_utilization=0.95         # 激进: 0.9-0.95

# 使用分块预填充
rollout.enable_chunked_prefill=True

# 调整批次大小
rollout.max_num_batched_tokens=8192         # 减小以降低内存使用
rollout.max_num_seqs=512                    # 减小以降低内存使用
```

### 2. 吞吐量优化

```bash
# 增加批次大小
data.batch_size=256                         # 根据 GPU 内存调整

# 优化序列长度
rollout.prompt_length=2048                  # 根据实际需求设置
rollout.response_length=1024                # 不要设置过大

# 使用最优的张量并行大小
rollout.tensor_model_parallel_size=2        # 平衡并行和通信开销
```

### 3. 延迟优化

```bash
# 使用更小的批次
data.batch_size=32

# 使用更大的张量并行
rollout.tensor_model_parallel_size=4        # 减少每个 GPU 的计算量

# 启用 eager 模式
rollout.enforce_eager=True
```

## 工作流程集成

### 1. 数据准备流程

```bash
# 步骤 1: 准备提示数据
python prepare_prompts.py --output prompts.parquet

# 步骤 2: 批量生成
python3 -m verl.trainer.main_generation \
    data.path=prompts.parquet \
    data.output_path=generations.parquet \
    # ...

# 步骤 3: 后处理
python postprocess_generations.py --input generations.parquet
```

### 2. 模型比较流程

```bash
# 生成模型 A 的输出
python3 -m verl.trainer.main_generation \
    model.path=model_a \
    data.output_path=model_a_outputs.parquet \
    # ...

# 生成模型 B 的输出
python3 -m verl.trainer.main_generation \
    model.path=model_b \
    data.output_path=model_b_outputs.parquet \
    # ...

# 比较结果
python compare_models.py \
    --model_a model_a_outputs.parquet \
    --model_b model_b_outputs.parquet
```

### 3. RLHF 数据准备流程

```bash
# 步骤 1: 生成多个响应
python3 -m verl.trainer.main_generation \
    data.n_samples=8 \                      # 每个提示 8 个响应
    rollout.temperature=1.0 \
    data.output_path=rlhf_candidates.parquet

# 步骤 2: 使用奖励模型评分
python score_generations.py \
    --input rlhf_candidates.parquet \
    --output rlhf_scored.parquet

# 步骤 3: 用于 PPO/GRPO 训练
python3 -m verl.trainer.main_ppo \
    data.train_files=rlhf_scored.parquet \
    # ...
```

## 常见问题

### 1. 生成速度慢怎么办？

**解决方案**:
```bash
# 增加批次大小
data.batch_size=256

# 增加数据并行（减少张量并行）
rollout.tensor_model_parallel_size=2        # 而不是 4 或 8

# 减少序列长度
rollout.response_length=512                 # 如果不需要长文本

# 使用更高的 GPU 内存利用率
rollout.gpu_memory_utilization=0.95
```

### 2. 内存不足怎么办？

**解决方案**:
```bash
# 减小批次大小
data.batch_size=32

# 增加张量并行
rollout.tensor_model_parallel_size=4        # 或更大

# 减小序列长度
rollout.prompt_length=1024
rollout.response_length=512

# 降低 GPU 内存利用率
rollout.gpu_memory_utilization=0.7

# 减小最大批次 token 数
rollout.max_num_batched_tokens=4096
rollout.max_num_seqs=256
```

### 3. 如何生成多样化的输出？

**配置方案**:
```bash
# 多样化采样
data.n_samples=8                            # 多个样本
rollout.temperature=1.0                     # 或更高 (1.2)
rollout.top_p=0.9                           # 更宽松
rollout.top_k=-1                            # 不限制 top-k
```

### 4. 如何生成确定性输出？

**配置方案**:
```bash
# 确定性生成
data.n_samples=1
rollout.temperature=0.0                     # 贪心解码
rollout.top_p=1.0
rollout.top_k=-1
```

### 5. vLLM 异步模式问题

**注意**: 根据配置文件注释，`main_generation.py` 在 vLLM 异步模式下可能存在兼容性问题（issue #4682）。

**临时解决方案**:
- 参考 issue #4682 的讨论
- 考虑使用 `main_generation_server.py`（支持异步）
- 或暂时使用 HF 后端: `rollout.name=hf`

### 6. 如何处理大型数据集？

**策略**:
```bash
# 分批处理
python split_data.py --input large.parquet --chunks 10

# 并行生成每个批次
for i in {1..10}; do
    python3 -m verl.trainer.main_generation \
        data.path=chunk_$i.parquet \
        data.output_path=output_$i.parquet \
        # ...
done

# 合并结果
python merge_results.py --output final.parquet
```

### 7. 如何监控生成进度？

生成过程会在控制台显示进度信息：
- 已处理的样本数
- 生成速度（tokens/sec）
- 预计剩余时间

可以查看日志：
```bash
# 查看详细日志
tail -f logs/generation.log
```

### 8. 输出数据格式说明

输出的 Parquet 文件会包含：
- 所有输入字段（保持不变）
- 新增的生成字段（generated_text 等）

如果 `n_samples > 1`，会为每个提示生成多行输出。

## 与训练工作流的集成

### 生成数据用于 SFT

```bash
# 1. 生成高质量响应
python3 -m verl.trainer.main_generation \
    model.path=teacher_model \
    data.n_samples=1 \
    rollout.temperature=0.7 \
    data.output_path=sft_data.parquet

# 2. 使用生成的数据进行 SFT
python3 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=sft_data.parquet \
    # ...
```

### 生成数据用于 RLHF

```bash
# 1. 生成候选响应
python3 -m verl.trainer.main_generation \
    data.n_samples=8 \
    rollout.temperature=1.0 \
    data.output_path=rlhf_candidates.parquet

# 2. 使用 PPO/GRPO 训练
python3 -m verl.trainer.main_ppo \
    data.train_files=rlhf_candidates.parquet \
    # ...
```

## 参考资料

### 相关工具

- **vLLM**: 高性能推理引擎 - https://docs.vllm.ai/
- **SGLang**: 结构化生成语言 - https://github.com/sgl-project/sglang

### 相关示例

- `examples/ppo_trainer/`: PPO 训练示例
- `examples/sft/`: SFT 训练示例
- `examples/data_preprocess/`: 数据预处理示例

### 配置文件

- `verl/trainer/config/generation.yaml`: 生成配置
- `verl/trainer/main_generation.py`: 生成主程序
- `verl/trainer/main_generation_server.py`: 异步生成服务

## 总结

Generation 工具提供了一个完整的批量文本生成解决方案，适合：

1. 大规模批量推理
2. 数据增强和合成
3. 模型评估和比较
4. RLHF 数据准备
5. 生产环境部署

通过合理配置并行策略和采样参数，您可以高效地处理各种规模的生成任务。

## 支持与反馈

如果遇到问题或有改进建议：

1. 查看 [veRL 文档](https://verl.readthedocs.io/)
2. 提交 [GitHub Issue](https://github.com/volcengine/verl/issues)
3. 参考 issue #4682 关于 vLLM 异步模式的讨论

祝生成顺利！
