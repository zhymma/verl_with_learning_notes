# MTP Trainer - 多令牌预测训练

[English](README.md) | 简体中文

## 概述

MTP (Multiple Token Prediction) Trainer 展示了如何在 RLHF 训练中集成多令牌预测技术。MTP 是一种先进的训练方法，它让模型在每个时间步不仅预测下一个 token，还预测后续的多个 token，从而提高模型的推理效率和质量。

本示例基于 DAPO (Dual-Advantage Policy Optimization) 算法，结合 MTP 技术训练 MiMo-7B 模型。

## 主要特点

- **多令牌预测 (MTP)**: 在训练阶段启用多令牌预测，增强模型的长期规划能力
- **DAPO 算法**: 使用双优势策略优化，提供更稳定的策略更新
- **Megatron-LM 后端**: 使用 Megatron-LM 进行高效的分布式训练
- **SGLang 推理**: 使用 SGLang 作为推理引擎，支持 MTP 的 EAGLE 推测解码
- **大规模训练**: 支持 16 节点 128 GPU 的大规模训练配置

## 适用场景

MTP Trainer 适用于以下场景：

1. **推理加速**: 通过多令牌预测减少推理延迟
2. **长文本生成**: 提高长序列生成的质量和效率
3. **数学推理任务**: 特别适合需要多步推理的数学问题求解
4. **大规模模型训练**: 利用高效的并行策略训练大型语言模型

## MTP 技术详解

### 什么是 MTP？

MTP (Multiple Token Prediction) 是一种训练技术，它扩展了标准的语言模型训练目标：

- **标准语言模型**: 在每个位置 t 预测下一个 token t+1
- **MTP 模型**: 在每个位置 t 同时预测 token t+1, t+2, ..., t+n

### MTP 的优势

1. **推理加速**: 一次前向传播可以生成多个 token
2. **更好的长期规划**: 模型学习到更长范围的依赖关系
3. **训练效率**: 从每个训练样本中提取更多的监督信号
4. **推测解码**: 可以与 EAGLE 等推测解码算法结合使用

### MTP 配置参数

```yaml
actor_rollout_ref.model.mtp:
  enable: true                    # 启用 MTP 参数加载/保存
  enable_train: true              # 训练时使用 MTP
  enable_rollout: false           # 推理时使用 MTP（可选）
  mtp_loss_scaling_factor: 0.1   # MTP 损失的缩放因子
  detach_encoder: true            # 是否分离编码器梯度
```

## 快速开始

### 环境准备

1. 安装 veRL 及其依赖：

```bash
# 安装 veRL 和 SGLang
pip install -e .[test,sglang]

# 安装 Megatron-LM
pip install megatron-core
```

2. 准备数据和模型：

```bash
# 设置数据路径
export RAY_DATA_HOME="${HOME}/verl"

# 下载 MiMo-7B-RL 模型
# 注意: 下载后需要修改 config.json 中的 max_position_embeddings 为 32768

# 准备训练数据 (DAPO Math 17K 数据集)
# 准备测试数据 (AIME 2024 数据集)
```

### 基本运行

使用提供的脚本启动训练：

```bash
# 设置环境变量
export NNODES=16              # 节点数量
export NGPUS_PER_NODE=8       # 每个节点的 GPU 数量
export RAY_DATA_HOME="${HOME}/verl"

# 运行训练
bash examples/mtp_trainer/test_dapo_mimo_7b_with_mtp_math_megatron.sh
```

## 详细配置说明

### 1. MTP 相关配置

```bash
# MTP 核心参数
actor_rollout_ref.model.mtp.enable=True               # 启用 MTP
actor_rollout_ref.model.mtp.enable_train=True         # 训练时使用 MTP
actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=0.1  # MTP 损失权重
actor_rollout_ref.model.mtp.detach_encoder=True       # 分离编码器梯度
```

### 2. DAPO 算法配置

```bash
adv_estimator=grpo                    # 使用 GRPO 作为优势估计器
use_kl_in_reward=False                # 不在奖励中使用 KL 散度
kl_coef=0.0                           # KL 系数
use_kl_loss=False                     # 不使用 KL 损失
kl_loss_coef=0.0                      # KL 损失系数
clip_ratio_low=0.2                    # 策略裁剪下界
clip_ratio_high=0.28                  # 策略裁剪上界
```

### 3. 数据和序列长度配置

```bash
max_prompt_length=$((1024 * 2))       # 最大提示长度: 2048
max_response_length=$((1024 * 8))     # 最大响应长度: 8192
enable_overlong_buffer=True           # 启用超长序列缓冲
overlong_buffer_len=$((1024 * 4))     # 超长缓冲区长度: 4096
overlong_penalty_factor=1.0           # 超长惩罚因子
```

### 4. 批次配置

```bash
train_prompt_bsz=128                  # 训练批次大小
n_resp_per_prompt=16                  # 每个提示的响应数量
train_prompt_mini_bsz=32              # Mini-batch 大小
```

### 5. 并行策略配置

```bash
# 推理并行
gen_tp=4                              # 推理时的张量并行度

# 训练并行
train_tp=2                            # 训练时的张量并行度
train_pp=2                            # 流水线并行度
train_cp=2                            # 上下文并行度

# 内存优化
offload=True                          # 启用参数卸载
```

### 6. 推理配置 (SGLang)

```bash
actor_rollout_ref.rollout.name=sglang                    # 使用 SGLang
actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
actor_rollout_ref.rollout.enable_chunked_prefill=True    # 启用分块预填充
actor_rollout_ref.rollout.gpu_memory_utilization=0.80    # GPU 内存利用率
```

### 7. 奖励模型配置 (DAPO)

```bash
reward_model.reward_manager=dapo                         # 使用 DAPO 奖励管理器
+reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
+reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
+reward_model.reward_kwargs.max_resp_len=${max_response_length}
```

### 8. 训练器配置

```bash
trainer.total_epochs=10                                  # 总训练轮数
trainer.total_training_steps=400                         # 总训练步数
trainer.test_freq=10                                     # 测试频率
trainer.save_freq=-1                                     # 保存频率 (-1 表示不保存)
trainer.val_before_train=False                           # 训练前不验证
trainer.log_val_generations=10                           # 记录验证生成样本数
```

## 运行示例

### 示例 1: 基本训练

使用默认配置进行训练：

```bash
bash examples/mtp_trainer/test_dapo_mimo_7b_with_mtp_math_megatron.sh
```

### 示例 2: 自定义并行策略

调整并行配置以适应不同的硬件：

```bash
# 修改脚本中的并行参数
gen_tp=8        # 增加推理张量并行度
train_tp=4      # 增加训练张量并行度
train_pp=1      # 禁用流水线并行
train_cp=1      # 禁用上下文并行
```

### 示例 3: 调整 MTP 参数

修改 MTP 相关配置：

```bash
# 在训练脚本中添加或修改
common_params=(
    actor_rollout_ref.model.mtp.enable=True
    actor_rollout_ref.model.mtp.enable_train=True
    actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=0.2  # 增加 MTP 损失权重
    actor_rollout_ref.model.mtp.detach_encoder=False         # 不分离编码器梯度
)
```

### 示例 4: 在推理时启用 MTP

启用 MTP 进行推测解码：

```bash
# 训练时启用 MTP
actor_rollout_ref.model.mtp.enable_train=True

# 推理时也启用 MTP（使用 EAGLE 推测解码）
actor_rollout_ref.model.mtp.enable_rollout=True

# SGLang EAGLE 配置
actor_rollout_ref.model.mtp.speculative_algorithm=EAGLE
actor_rollout_ref.model.mtp.speculative_num_steps=2
actor_rollout_ref.model.mtp.speculative_eagle_topk=2
actor_rollout_ref.model.mtp.speculative_num_draft_tokens=4
```

## Runtime Environment 配置

`runtime_env.yaml` 文件配置 Ray 运行时环境：

```yaml
working_dir: ./
excludes:
  - ".git/"

env_vars:
  VLLM_USE_V1: "1"                    # 使用 vLLM v1 API
  HYDRA_FULL_ERROR: "1"               # Hydra 完整错误信息
  NCCL_NVLS_ENABLE: "0"               # 禁用 NCCL NVLS
  NCCL_SOCKET_IFNAME: "eth0"          # NCCL 网络接口
  TMPDIR: "/tmp"                      # 临时目录
  CUDA_HOME: "/usr/local/cuda"        # CUDA 路径
  HF_HOME: "/tmp/hf_home_mimo"        # HuggingFace 缓存
  PYTHONPATH: "/tmp/hf_home_mimo/modules/"
```

## 常见问题

### 1. MTP 训练不稳定怎么办？

**问题**: MTP 损失导致训练不稳定。

**解决方案**:
- 降低 `mtp_loss_scaling_factor` (例如从 0.1 降到 0.05)
- 启用 `detach_encoder=True` 来稳定训练
- 减小学习率
- 使用梯度裁剪 (`optim.clip_grad=1.0`)

### 2. 如何验证 MTP 是否生效？

**验证方法**:
```bash
# 检查日志中的 MTP 损失
# 应该看到类似 "mtp_loss" 的日志项

# 在训练时启用详细日志
trainer.logger='["console","tensorboard"]'

# 查看 TensorBoard 中的 MTP 相关指标
```

### 3. MTP 模型如何加载？

MTP 参数通常存储在模型权重中，作为额外的预测头：
- 标准 LM head: 预测下一个 token
- MTP heads: 预测后续的 2, 3, ... n 个 token

确保模型已经用 MTP 进行了预训练，或者从头开始训练 MTP heads。

### 4. 推理时是否必须使用 MTP？

不是必须的：
- `enable_train=True, enable_rollout=False`: 只在训练时使用 MTP
- `enable_train=True, enable_rollout=True`: 训练和推理都使用 MTP
- 推理时使用 MTP 可以加速生成，但需要支持推测解码的推理引擎（如 SGLang 的 EAGLE）

### 5. 内存不足怎么办？

**解决方案**:
```bash
# 启用所有卸载选项
offload=True

# 减小批次大小
train_prompt_bsz=64
train_prompt_mini_bsz=16

# 增加张量并行度
train_tp=4

# 减小序列长度
max_response_length=$((1024 * 6))  # 从 8K 降到 6K

# 降低 GPU 内存利用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.70
```

### 6. DAPO 奖励模型如何工作？

DAPO (Dual-Advantage Policy Optimization) 使用特殊的奖励计算方式：
- 支持超长序列缓冲区管理
- 对超长生成进行惩罚
- 结合多个优势估计

配置超长缓冲区：
```bash
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))    # 4K 缓冲区
overlong_penalty_factor=1.0          # 惩罚因子
```

### 7. 如何选择并行策略？

**推荐配置** (16 节点, 128 GPU):
```bash
# 推理并行 (需要更高的 TP 以减少延迟)
gen_tp=4

# 训练并行 (平衡计算和通信)
train_tp=2     # 张量并行
train_pp=2     # 流水线并行
train_cp=2     # 上下文并行 (长序列必需)
```

**调整原则**:
- `gen_tp`: 推理延迟敏感，使用较大的值
- `train_tp`: 平衡计算效率和通信开销
- `train_pp`: 大模型必需，但会增加流水线气泡
- `train_cp`: 长序列 (>4K) 时必需

### 8. 如何监控训练进度？

启用 Prometheus 和 TensorBoard：
```bash
# Prometheus 监控
actor_rollout_ref.rollout.prometheus.enable=True
actor_rollout_ref.rollout.prometheus.port=44398

# TensorBoard 日志
trainer.logger='["console","tensorboard"]'

# 查看 TensorBoard
tensorboard --logdir=${CKPTS_DIR}/tensorboard
```

## 性能优化建议

### 1. 内存优化

- 启用参数卸载: `offload=True`
- 使用分块预填充: `enable_chunked_prefill=True`
- 调整 GPU 内存利用率: `gpu_memory_utilization=0.80`

### 2. 计算优化

- 使用动态批次大小: `use_dynamic_bsz=True`
- 启用 Megatron 桥接: `megatron.use_mbridge=True`
- 优化 token 长度限制

### 3. 通信优化

- 合理设置并行度 (TP, PP, CP)
- 配置 NCCL 参数
- 使用高速网络接口

## 参考资料

### 相关论文

1. **MTP 技术**:
   - [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)

2. **DAPO 算法**:
   - DAPO: Dual-Advantage Policy Optimization for RL

3. **推测解码**:
   - [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)

### 相关示例

- `examples/ppo_trainer/`: 标准 PPO 训练
- `examples/grpo_trainer/`: GRPO 训练
- `examples/sglang_multiturn/`: SGLang 多轮对话

### 配置文件

- `verl/trainer/config/ppo_megatron_trainer.yaml`: Megatron PPO 配置
- `verl/workers/config/model.py`: MTP 配置定义
- `verl/models/mcore/mtp_patch.py`: MTP 实现

## 总结

MTP Trainer 提供了一个完整的多令牌预测训练示例，展示了如何：

1. 配置和启用 MTP 技术
2. 使用 DAPO 算法进行强化学习
3. 在大规模分布式环境中训练
4. 优化内存和计算效率
5. 集成推测解码进行快速推理

通过 MTP 技术，您可以训练出更高效、更强大的语言模型，特别适合需要长序列生成和复杂推理的任务。

## 支持与反馈

如果遇到问题或有改进建议：

1. 查看 [veRL 文档](https://verl.readthedocs.io/)
2. 提交 [GitHub Issue](https://github.com/volcengine/verl/issues)
3. 加入社区讨论

祝训练顺利！
