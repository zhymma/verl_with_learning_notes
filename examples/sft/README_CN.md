# SFT 训练 (Supervised Fine-Tuning)

> 监督微调 - RL 训练的前置步骤

---

## 📋 概述

**SFT (Supervised Fine-Tuning)** 是强化学习训练的重要前置步骤。通过在高质量的标注数据上进行监督学习，可以为后续的 RL 训练提供更好的初始化。

### 为什么需要 SFT？

- ✅ **更好的初始化**：RL 从 SFT 模型开始，收敛更快
- ✅ **数据格式适应**：让模型熟悉特定的数据格式
- ✅ **基础能力建立**：在 RL 之前建立基本的任务能力
- ✅ **减少探索成本**：缩小 RL 的搜索空间

### SFT → RL 的完整流程

```
1. 预训练模型
   ↓
2. SFT 训练（本目录）
   ├─ 在高质量数据上监督学习
   ├─ 学习任务特定格式
   └─ 建立基础能力
   ↓
3. RL 训练
   ├─ 从 SFT checkpoint 开始
   ├─ 通过奖励信号优化
   └─ 获得更好的性能
```

### 适用场景

| 场景 | 是否需要 SFT | 说明 |
|------|-------------|------|
| **数学推理** | ⭐⭐⭐⭐⭐ 强烈推荐 | SFT 建立推理模式 |
| **代码生成** | ⭐⭐⭐⭐⭐ 强烈推荐 | SFT 学习代码语法 |
| **工具调用** | ⭐⭐⭐⭐⭐ 必需 | SFT 学习工具格式 |
| **对话质量** | ⭐⭐⭐⭐ 推荐 | SFT 建立对话模式 |
| **通用 RLHF** | ⭐⭐⭐ 可选 | 预训练模型已经很好 |

---

## 🔧 前置条件

### 硬件要求

```
最低配置：
- GPU: 1 张 24GB GPU（如 RTX 3090）
- 内存: 32GB
- 存储: 50GB

推荐配置：
- GPU: 4-8 张 40GB GPU（如 A100）
- 内存: 128GB+
- 存储: 200GB+
```

### 软件依赖

```bash
# 安装 verl
pip install -e .[test]

# 验证安装
python -c "import verl; print(verl.__version__)"
```

### 数据准备

SFT 需要高质量的标注数据：

#### 1. GSM8K SFT 数据

```bash
# 处理 GSM8K SFT 数据（包含完整解题过程）
python examples/data_preprocess/gsm8k_multiturn_sft.py \
    --local_save_dir ~/data/gsm8k_sft

# 数据格式：
# {
#   "prompt": [{"role": "user", "content": "Question..."}],
#   "response": "Step 1: ... Step 2: ... #### 42",  # 完整的解题过程
#   "data_source": "gsm8k_sft"
# }
```

#### 2. 多轮对话 SFT 数据

```bash
# 处理多轮对话数据
python examples/data_preprocess/multiturn.py \
    --local_save_dir ~/data/multiturn_sft

# 数据格式（多轮）：
# {
#   "prompt": [
#     {"role": "user", "content": "Q1"},
#     {"role": "assistant", "content": "A1"},
#     {"role": "user", "content": "Q2"}
#   ],
#   "response": "A2",
#   "data_source": "multiturn_sft"
# }
```

---

## 🚀 快速开始

### 示例 1：Qwen2.5-0.5B SFT（最简单）

```bash
# GSM8K SFT 训练
cd examples/sft/gsm8k

bash run_qwen_05_sp2.sh

# 预期输出：
# Epoch 0: loss=2.134
# Epoch 1: loss=1.567
# Epoch 2: loss=1.234
# ...
# Epoch 9: loss=0.567
# ✅ SFT 训练完成！
# 模型保存到: ./checkpoints/qwen0.5b_sft/
```

### 示例 2：使用 LoRA（节省显存）

```bash
# LoRA SFT 训练（显存占用更少）
bash run_qwen_05_peft.sh

# LoRA 配置：
# - r=16（rank）
# - alpha=32
# - target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
#
# 显存占用：~12GB（相比全量的 ~20GB）
```

### 示例 3：Gemma 2B SFT

```bash
# Gemma 模型 SFT
bash examples/sft/gsm8k/run_gemma_2b.sh

# 或自定义配置
python3 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k_sft/train.parquet \
    data.val_files=$HOME/data/gsm8k_sft/test.parquet \
    model.path=google/gemma-2-2b-it \
    trainer.default_local_dir=./checkpoints/gemma2b_sft \
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=10
```

### 示例 4：多模态 VLM SFT

```bash
# 多模态 SFT（图像 + 文本）
cd examples/sft/vlm

python run_vlm_sft.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --data_path ~/data/vlm_sft/train.parquet \
    --output_dir ./checkpoints/qwen2.5_vl_sft
```

---

## 📖 详细配置

### SFT 训练配置

#### 1. 数据配置

```yaml
data:
  train_files: ~/data/gsm8k_sft/train.parquet
  val_files: ~/data/gsm8k_sft/test.parquet
  train_batch_size: 128              # 批次大小
  max_prompt_length: 1024            # 最大 prompt 长度
  max_response_length: 512           # 最大 response 长度
  num_workers: 4                     # 数据加载线程数
```

#### 2. 模型配置

```yaml
model:
  path: Qwen/Qwen2.5-7B-Instruct     # 模型路径

  # FSDP 配置
  fsdp_config:
    param_offload: False             # 参数卸载到 CPU
    optimizer_offload: False         # 优化器卸载
    gradient_checkpointing: True     # 梯度检查点（省显存）

  # LoRA 配置（可选）
  peft_config:
    enable: True                     # 启用 LoRA
    r: 16                            # LoRA rank
    alpha: 32                        # LoRA alpha
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    lora_dropout: 0.05
```

#### 3. 优化器配置

```yaml
optim:
  lr: 5e-6                           # 学习率（SFT 通常比 RL 小）
  weight_decay: 0.01                 # 权重衰减
  warmup_steps: 100                  # Warmup 步数
  lr_scheduler: cosine               # 学习率调度器
```

#### 4. 训练配置

```yaml
trainer:
  n_gpus_per_node: 4                 # 每节点 GPU 数
  nnodes: 1                          # 节点数
  total_epochs: 10                   # 总轮数
  save_freq: 2                       # 保存频率
  eval_freq: 1                       # 评估频率
  default_local_dir: ./checkpoints   # 保存目录
  gradient_accumulation_steps: 1     # 梯度累积
  max_grad_norm: 1.0                 # 梯度裁剪
```

---

## 💡 运行示例

### 示例 1：标准 SFT 训练流程

```bash
# 第 1 步：准备数据
python examples/data_preprocess/gsm8k_multiturn_sft.py \
    --local_save_dir ~/data/gsm8k_sft

# 第 2 步：SFT 训练
python3 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k_sft/train.parquet \
    data.val_files=$HOME/data/gsm8k_sft/test.parquet \
    data.train_batch_size=128 \
    model.path=Qwen/Qwen2.5-7B-Instruct \
    optim.lr=5e-6 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=10 \
    trainer.default_local_dir=./checkpoints/qwen7b_sft

# 第 3 步：评估 SFT 模型
python evaluate_sft.py \
    --model_path ./checkpoints/qwen7b_sft \
    --test_file ~/data/gsm8k/test.parquet

# 第 4 步：使用 SFT 模型进行 RL 训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=./checkpoints/qwen7b_sft \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    # ... 其他 RL 参数 ...
```

### 示例 2：使用 Liger Kernel 加速

```bash
# Liger Kernel 可以加速训练并减少显存
bash examples/sft/gsm8k/run_qwen_05_sp2_liger.sh

# Liger 特点：
# ✅ 更快的训练速度（~20% 提升）
# ✅ 更少的显存占用（~15% 减少）
# ✅ 数值稳定性更好
```

### 示例 3：NPU 训练（华为 Ascend）

```bash
# 在华为 Ascend NPU 上训练
bash examples/sft/gsm8k/run_qwen3_8b_sft_peft_sp2_npu.sh

# NPU 配置：
# - 需要安装 torch_npu
# - 支持 FSDP 和 LoRA
# - 性能与 A100 相当
```

### 示例 4：超大模型 SFT（36B+）

```bash
# SEED-OSS 36B SFT
bash examples/sft/gsm8k/run_seed_oss_36b_sft.sh

# 配置：
# - 8 张 80GB GPU
# - FSDP + 混合精度
# - 梯度检查点
# - 预计训练时间：~4 小时
```

---

## 🎯 SFT 最佳实践

### 1. 学习率选择

```yaml
# SFT 学习率通常比预训练小，比 RL 小
小模型（<7B）:
  lr: 1e-5 到 5e-5

中等模型（7B-70B）:
  lr: 5e-6 到 1e-5

大模型（>70B）:
  lr: 1e-6 到 5e-6

# 推荐使用 warmup
warmup_steps: 总步数的 5-10%
```

### 2. 训练轮数

```yaml
# SFT 不需要太多轮次
标准任务: 5-10 epochs
简单任务: 3-5 epochs
复杂任务: 10-15 epochs

# 注意过拟合！
# 使用验证集监控，loss 不再下降时停止
```

### 3. Batch Size 选择

```yaml
# 根据 GPU 显存调整
24GB GPU（单卡）:
  batch_size: 8-16（全量）
  batch_size: 32-64（LoRA）

40GB GPU（单卡）:
  batch_size: 16-32（全量）
  batch_size: 64-128（LoRA）

# 使用梯度累积增大有效 batch size
gradient_accumulation_steps: 4
# 有效 batch_size = batch_size × accumulation_steps
```

### 4. 数据质量 > 数据数量

```yaml
# 宁可少而精，不要多而杂
高质量数据（1000 条）> 低质量数据（10000 条）

# 数据清洗
- 移除重复数据
- 移除格式错误的数据
- 移除不相关的数据
- 验证答案的正确性
```

### 5. SFT 后验证

```bash
# SFT 训练完成后，务必验证效果

# 方法 1: 在验证集上计算 loss
python3 -m verl.trainer.fsdp_sft_trainer \
    --eval_only \
    --checkpoint ./checkpoints/qwen7b_sft

# 方法 2: 生成样例并人工检查
python generate_samples.py \
    --model ./checkpoints/qwen7b_sft \
    --num_samples 100

# 方法 3: 在测试集上评估准确率
python evaluate.py \
    --model ./checkpoints/qwen7b_sft \
    --test_file ~/data/gsm8k/test.parquet
```

---

## 🐛 常见问题

### Q1: SFT loss 不下降怎么办？

**可能原因：**

```bash
# 1. 学习率太小
optim.lr=1e-5  # 尝试增大到 5e-5

# 2. 学习率太大（loss 震荡）
optim.lr=5e-7  # 尝试减小到 1e-6

# 3. Batch size 太小
data.train_batch_size=128  # 增大 batch size
gradient_accumulation_steps=4  # 或使用梯度累积

# 4. 数据质量问题
# 检查数据：
python -c "
import pandas as pd
df = pd.read_parquet('~/data/gsm8k_sft/train.parquet')
print(df.head())
print(df['response'].apply(len).describe())  # 检查长度分布
"

# 5. 模型权重加载问题
# 检查是否正确加载了预训练权重
model.path=Qwen/Qwen2.5-7B-Instruct  # 确认路径正确
```

### Q2: SFT 过拟合怎么办？

**症状：**
```
训练 loss 持续下降，但验证 loss 上升
```

**解决方案：**

```bash
# 1. 减少训练轮数
trainer.total_epochs=5  # 从 10 减小到 5

# 2. 增加正则化
optim.weight_decay=0.1  # 从 0.01 增大到 0.1

# 3. 使用 Dropout
model.dropout=0.1

# 4. 使用 LoRA（天然正则化）
model.peft_config.enable=True
model.peft_config.r=8  # 减小 rank

# 5. 增加数据量
# 使用数据增强或获取更多数据

# 6. Early stopping
# 监控验证 loss，不再下降时停止
```

### Q3: OOM（显存不足）怎么办？

**解决方案：**

```bash
# 1. 减小 batch size
data.train_batch_size=64  # 从 128 减小

# 2. 启用梯度检查点
model.fsdp_config.gradient_checkpointing=True

# 3. 使用 LoRA
model.peft_config.enable=True

# 4. 启用参数卸载（会变慢）
model.fsdp_config.param_offload=True
model.fsdp_config.optimizer_offload=True

# 5. 使用混合精度
trainer.mixed_precision=True

# 6. 减小序列长度
data.max_prompt_length=512  # 从 1024 减小
data.max_response_length=256  # 从 512 减小

# 7. 使用梯度累积
gradient_accumulation_steps=4
# 这样可以减小每步的 batch size，但保持有效 batch size
```

### Q4: SFT 后 RL 效果反而变差？

**可能原因：**

```bash
# 1. SFT 过拟合导致模型失去多样性
# 解决：减少 SFT 训练轮数

# 2. SFT 数据分布与 RL 任务不匹配
# 解决：检查数据一致性

# 3. SFT 学习率太大，破坏了预训练知识
# 解决：减小学习率（如 1e-6）

# 4. RL 学习率设置不当
# 解决：RL 学习率应该比 SFT 更小
actor_rollout_ref.actor.optim.lr=5e-7  # SFT 用的是 5e-6

# 5. 没有正确加载 SFT checkpoint
# 解决：确认路径和加载方式
actor_rollout_ref.model.path=./checkpoints/qwen7b_sft/checkpoint-final
```

### Q5: 多轮对话 SFT 如何配置？

**数据格式：**

```python
# 多轮对话 SFT 数据
{
    "prompt": [
        {"role": "user", "content": "第一个问题"},
        {"role": "assistant", "content": "第一个回答"},
        {"role": "user", "content": "第二个问题"}
    ],
    "response": "第二个回答",  # 只需要标注最后一轮
    "data_source": "multiturn_sft"
}
```

**训练配置：**

```bash
python3 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/multiturn_sft/train.parquet \
    data.max_prompt_length=2048 \  # 多轮对话需要更长
    # 其他参数同单轮
```

---

## 📊 性能基准

### Qwen2.5-0.5B GSM8K SFT

```
训练前准确率: 36.4%
SFT 后准确率: 52.8%
SFT + RL 准确率: 56.7%

SFT 配置:
- lr: 1e-5
- epochs: 10
- batch_size: 128
- 训练时间: ~20 分钟（2x A100）
```

### Qwen2-7B GSM8K SFT

```
训练前准确率: 65.2%
SFT 后准确率: 72.1%
SFT + RL 准确率: 78.5%

SFT 配置:
- lr: 5e-6
- epochs: 8
- batch_size: 64
- 训练时间: ~1 小时（4x A100）
```

---

## 🔗 参考资料

### 官方文档

- [FSDP 文档](https://pytorch.org/docs/stable/fsdp.html)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)

### 学习笔记

- [01_快速上手](../../learning_notes/01_快速上手/) - 环境安装
- [02_数据准备](../../learning_notes/02_数据准备/) - SFT 数据格式

### 相关示例

- `examples/ppo_trainer/` - PPO RL 训练（SFT 后的下一步）
- `examples/grpo_trainer/` - GRPO RL 训练
- `examples/data_preprocess/` - 数据预处理

---

**创建时间**: 2026-01-28
**适用版本**: verl v0.2+
**维护者**: verl team
