# 数据预处理 (Data Preprocess)

> 将各种格式的数据集转换为 verl 训练所需的 Parquet 格式

---

## 📋 概述

本目录包含了将常见数据集转换为 verl 训练格式的预处理脚本。verl 使用 **Parquet** 格式存储训练数据，支持单轮对话、多轮对话、工具调用和多模态等多种任务类型。

### 适用场景

- 准备数学推理数据集（GSM8K、MATH等）
- 准备对话数据集（HH-RLHF等）
- 准备多轮对话数据（Agent训练）
- 准备工具调用数据
- 准备多模态数据

### 支持的数据集

| 脚本 | 数据集 | 任务类型 | 说明 |
|------|--------|---------|------|
| `gsm8k.py` | GSM8K | 数学推理 | 单轮数学问题 |
| `gsm8k_multiturn_w_tool.py` | GSM8K + Tool | 工具调用 | 带计算器工具的多轮对话 |
| `gsm8k_multiturn_w_interaction.py` | GSM8K | 多轮对话 | 交互式解题 |
| `gsm8k_multiturn_sft.py` | GSM8K | SFT | 监督微调数据 |
| `gsm8k_tool_agent_loop.py` | GSM8K + Tool | Agent RL | Agent Loop 训练数据 |
| `math_dataset.py` | MATH | 数学推理 | 高级数学问题 |
| `geo3k.py` | GEO3K | 几何推理 | 几何问题求解 |
| `geo3k_multiturn_w_tool.py` | GEO3K + Tool | 工具调用 | 带工具的几何问题 |
| `full_hh_rlhf.py` | HH-RLHF | 对话安全 | 人类偏好对齐 |
| `hellaswag.py` | HellaSwag | 常识推理 | 句子补全任务 |
| `multiturn.py` | 通用 | 多轮对话 | 通用多轮对话格式 |
| `pokemon.py` | Pokemon | 游戏对话 | 示例数据集 |
| `aime2024_multiturn_w_tool.py` | AIME 2024 | 数学竞赛 | 高难度数学题 |
| `dapo_multiturn_w_tool.py` | DAPO | 多轮工具 | DAPO 算法数据 |
| `preprocess_search_r1_dataset.py` | Search R1 | 搜索推理 | R1 模型数据 |

---

## 🔧 前置条件

### 环境依赖

```bash
# 基础依赖
pip install datasets pandas pyarrow

# HDFS 支持（如果使用分布式存储）
pip install hdfs

# 特定数据集依赖
pip install openai  # 用于某些数据集的 API 调用
```

### 数据集访问

大部分脚本会自动从 HuggingFace Hub 下载数据集，但需要：

1. **网络连接**：能访问 HuggingFace Hub
2. **HuggingFace Token**（可选）：某些私有数据集需要
   ```bash
   export HF_TOKEN=your_token_here
   ```

### 本地数据集（可选）

如果已有本地数据集，可以使用 `--local_dataset_path` 参数：

```bash
python gsm8k.py --local_dataset_path /path/to/gsm8k
```

---

## 🚀 快速开始

### 示例 1：处理 GSM8K 数据集

```bash
# 下载并处理 GSM8K 数据集
python examples/data_preprocess/gsm8k.py \
    --local_save_dir ~/data/gsm8k

# 查看生成的文件
ls ~/data/gsm8k/
# 输出：train.parquet  test.parquet
```

### 示例 2：处理多轮对话数据

```bash
# 处理带工具调用的 GSM8K 数据
python examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_save_dir ~/data/gsm8k_multiturn

# 处理通用多轮对话数据
python examples/data_preprocess/multiturn.py \
    --local_save_dir ~/data/multiturn
```

### 示例 3：处理 HH-RLHF 数据集

```bash
# 处理完整的 HH-RLHF 数据集（需要更多时间和存储）
python examples/data_preprocess/full_hh_rlhf.py \
    --local_save_dir ~/data/hh_rlhf
```

---

## 📖 详细配置

### 通用参数

所有预处理脚本支持以下参数：

```bash
python <script_name>.py \
    --local_save_dir <保存目录> \        # 必需：本地保存路径
    --local_dataset_path <数据集路径> \  # 可选：本地数据集路径
    --hdfs_dir <HDFS路径> \             # 可选：HDFS 存储路径
    --local_dir <本地缓存目录>           # 可选：临时缓存目录
```

### 参数详解

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--local_save_dir` | 处理后数据的保存目录 | `~/data/<dataset_name>` | `~/data/gsm8k` |
| `--local_dataset_path` | 本地原始数据集路径 | `None`（从 HF 下载） | `/data/raw/gsm8k` |
| `--hdfs_dir` | HDFS 分布式存储路径 | `None` | `hdfs://cluster/data` |
| `--local_dir` | 临时文件缓存目录 | `None` | `/tmp/preprocess` |

---

## 📊 数据格式说明

### verl 标准格式

所有预处理脚本生成的 Parquet 文件都包含以下字段：

```python
{
    # ========== 必需字段 ==========
    "data_source": str,           # 数据来源标识，如 "openai/gsm8k"
    "prompt": list or str,        # 用户输入（支持多种格式）

    # ========== 推荐字段 ==========
    "ability": str,               # 任务能力类别，如 "math", "chat"
    "reward_model": {             # Reward 计算信息
        "style": str,             # "rule" 或 "model"
        "ground_truth": str,      # 标准答案（rule-based 需要）
    },
    "extra_info": {               # 额外元数据
        "split": str,             # "train" 或 "test"
        "index": int,             # 数据索引
    }
}
```

### Prompt 字段的 4 种格式

#### 格式 1：单轮对话（字符串）

```python
{
    "prompt": "What is 2 + 2?",
}
```

#### 格式 2：单轮对话（Chat 格式，推荐）

```python
{
    "prompt": [
        {"role": "user", "content": "What is 2 + 2?"}
    ],
}
```

#### 格式 3：多轮对话

```python
{
    "prompt": [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What about 3 + 3?"}
    ],
}
```

#### 格式 4：工具调用

```python
{
    "prompt": [
        {"role": "user", "content": "Calculate 123 * 456"},
        {"role": "assistant", "content": "Let me calculate that."},
        {"role": "tool", "content": "56088", "name": "calculator"}
    ],
}
```

---

## 💡 运行示例

### GSM8K 单轮数学问题

```bash
# 1. 处理数据
python examples/data_preprocess/gsm8k.py \
    --local_save_dir ~/data/gsm8k

# 2. 验证数据（使用学习笔记中的脚本）
python learning_notes/01_快速上手/check_data.py ~/data/gsm8k/train.parquet

# 输出示例：
# ✅ 文件存在: ~/data/gsm8k/train.parquet
# ✅ 数据集大小: 7473 条
# ✅ 必需字段检查通过: data_source, prompt
# ✅ Prompt 格式: Chat 格式（推荐）
# ✅ 包含 reward_model 字段
#
# 样例数据:
# {
#   "data_source": "openai/gsm8k",
#   "prompt": [{"role": "user", "content": "Natalia sold clips to..."}],
#   "ability": "math",
#   "reward_model": {"style": "rule", "ground_truth": "48"}
# }
```

### GSM8K 多轮工具调用

```bash
# 1. 处理数据
python examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_save_dir ~/data/gsm8k_tool

# 2. 查看生成的数据
python -c "
import pandas as pd
df = pd.read_parquet('~/data/gsm8k_tool/train.parquet')
print('数据集大小:', len(df))
print('字段:', df.columns.tolist())
print('\n第一条数据:')
print(df.iloc[0]['prompt'])
"

# 输出示例：
# 数据集大小: 7473
# 字段: ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
#
# 第一条数据:
# [
#   {'role': 'user', 'content': 'Natalia sold clips to...'},
#   {'role': 'assistant', 'content': 'Let me use calculator...'},
#   {'role': 'tool', 'content': '48', 'name': 'calculator'}
# ]
```

### HH-RLHF 对话数据

```bash
# 1. 处理数据（较大，需要时间）
python examples/data_preprocess/full_hh_rlhf.py \
    --local_save_dir ~/data/hh_rlhf

# 2. 查看数据统计
python -c "
import pandas as pd
train_df = pd.read_parquet('~/data/hh_rlhf/train.parquet')
test_df = pd.read_parquet('~/data/hh_rlhf/test.parquet')
print(f'训练集: {len(train_df)} 条')
print(f'测试集: {len(test_df)} 条')
print(f'数据源: {train_df.iloc[0][\"data_source\"]}')
"
```

### 自定义数据集

如果你有自己的数据集，可以参考现有脚本创建预处理脚本：

```python
# my_dataset.py
import argparse
import pandas as pd
import datasets

def process_data(raw_data):
    processed = []
    for idx, item in enumerate(raw_data):
        processed.append({
            "data_source": "my_dataset",
            "prompt": [
                {"role": "user", "content": item["question"]}
            ],
            "ability": "custom",
            "reward_model": {
                "style": "rule",
                "ground_truth": item["answer"]
            },
            "extra_info": {
                "split": "train",
                "index": idx
            }
        })
    return processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", required=True)
    args = parser.parse_args()

    # 加载原始数据
    raw_dataset = datasets.load_dataset("your_dataset")

    # 处理数据
    train_data = process_data(raw_dataset["train"])
    test_data = process_data(raw_dataset["test"])

    # 保存为 Parquet
    pd.DataFrame(train_data).to_parquet(f"{args.local_save_dir}/train.parquet")
    pd.DataFrame(test_data).to_parquet(f"{args.local_save_dir}/test.parquet")

    print(f"✅ 数据处理完成！保存到: {args.local_save_dir}")
```

---

## ❓ 常见问题

### Q1: 下载数据集失败怎么办？

**问题：** 无法访问 HuggingFace Hub

**解决方案：**
```bash
# 方法 1: 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com
python gsm8k.py --local_save_dir ~/data/gsm8k

# 方法 2: 手动下载后使用本地路径
# 1. 从 https://huggingface.co/datasets/openai/gsm8k 手动下载
# 2. 使用 --local_dataset_path 参数
python gsm8k.py \
    --local_dataset_path /path/to/downloaded/gsm8k \
    --local_save_dir ~/data/gsm8k
```

### Q2: 如何验证生成的数据格式？

**使用内置验证脚本：**

```bash
# 使用学习笔记中的数据检查脚本
python learning_notes/01_快速上手/check_data.py ~/data/gsm8k/train.parquet

# 或使用更详细的质量检查脚本
python learning_notes/02_数据准备/data_quality_check.py ~/data/gsm8k/train.parquet
```

**手动检查：**

```python
import pandas as pd

# 读取数据
df = pd.read_parquet('~/data/gsm8k/train.parquet')

# 检查字段
print("字段:", df.columns.tolist())
assert 'data_source' in df.columns
assert 'prompt' in df.columns

# 检查第一条数据
print("\n第一条数据:")
print(df.iloc[0].to_dict())

# 检查 prompt 格式
first_prompt = df.iloc[0]['prompt']
if isinstance(first_prompt, list):
    print("✅ Chat 格式")
    assert first_prompt[0]['role'] == 'user'
else:
    print("⚠️  字符串格式（建议改为 Chat 格式）")
```

### Q3: 数据集太大，处理很慢怎么办？

**方法 1: 使用采样**

修改脚本，添加采样逻辑：

```python
# 在脚本中添加采样
train_dataset = dataset["train"].select(range(1000))  # 只取前 1000 条
```

**方法 2: 分批处理**

```python
# 分批保存
batch_size = 10000
for i in range(0, len(dataset), batch_size):
    batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
    # 处理并保存
    processed = process_batch(batch)
    pd.DataFrame(processed).to_parquet(f"train_part_{i//batch_size}.parquet")
```

### Q4: 如何处理多模态数据？

参考 `geo3k.py` 脚本，处理图像：

```python
{
    "prompt": [
        {
            "type": "image",
            "image": "/path/to/image.jpg"  # 或 base64 编码
        },
        {
            "type": "text",
            "text": "What's in this image?"
        }
    ],
    # ... 其他字段
}
```

### Q5: 如何自定义 Reward 函数？

在 `reward_model` 字段中指定：

```python
# Rule-based Reward（需要 ground_truth）
"reward_model": {
    "style": "rule",
    "ground_truth": "42"
}

# Model-based Reward（使用 RM 模型）
"reward_model": {
    "style": "model",
    "model_path": "path/to/reward/model"
}

# 在训练时，verl 会根据 data_source 路由到相应的 Reward 函数
# 详见 learning_notes/04_Reward设计/自定义Reward实践指南.md
```

### Q6: Parquet 文件太大怎么办？

**启用压缩：**

```python
# 使用更高的压缩级别
df.to_parquet(
    "train.parquet",
    compression='snappy',  # 或 'gzip', 'brotli', 'zstd'
    compression_level=9    # 最高压缩
)
```

**拆分文件：**

```python
# 按大小拆分
chunk_size = 100_000
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk.to_parquet(f"train_part_{i//chunk_size}.parquet")
```

### Q7: 如何使用 HDFS 分布式存储？

```bash
# 使用 --hdfs_dir 参数
python gsm8k.py \
    --local_save_dir /tmp/gsm8k \
    --hdfs_dir hdfs://your-cluster/data/gsm8k

# 脚本会先保存到本地，然后自动上传到 HDFS
```

---

## 🔗 参考资料

### 官方文档

- [verl 数据格式文档](../../docs/data/)
- [HuggingFace Datasets 文档](https://huggingface.co/docs/datasets/)
- [Parquet 格式说明](https://parquet.apache.org/docs/)

### 学习笔记

- [02_数据准备/02_数据准备.md](../../learning_notes/02_数据准备/02_数据准备.md) - 数据格式详解
- [04_Reward设计/自定义Reward实践指南.md](../../learning_notes/04_Reward设计/自定义Reward实践指南.md) - Reward 函数设计

### 数据集链接

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) - 小学数学应用题
- [MATH](https://huggingface.co/datasets/lighteval/MATH) - 高级数学问题
- [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) - 人类偏好对齐
- [HellaSwag](https://huggingface.co/datasets/hellaswag) - 常识推理

### 相关脚本

- `learning_notes/01_快速上手/check_data.py` - 数据格式验证
- `learning_notes/02_数据准备/data_quality_check.py` - 数据质量检查

---

## 📝 最佳实践

### 1. 数据质量检查

处理完数据后，始终进行质量检查：

```bash
# 自动检查
python learning_notes/02_数据准备/data_quality_check.py ~/data/gsm8k/train.parquet

# 手动抽查
python -c "
import pandas as pd
df = pd.read_parquet('~/data/gsm8k/train.parquet')
print(df.sample(5))  # 随机查看 5 条
"
```

### 2. 版本管理

为数据集添加版本标识：

```python
"extra_info": {
    "version": "v1.0",
    "processed_date": "2026-01-28",
    "preprocessing_script": "gsm8k.py"
}
```

### 3. 数据切分

为调试准备小规模数据：

```bash
# 创建 mini 版本用于快速测试
python -c "
import pandas as pd
df = pd.read_parquet('~/data/gsm8k/train.parquet')
df_mini = df.head(100)
df_mini.to_parquet('~/data/gsm8k/train_mini.parquet')
print('✅ 创建 mini 数据集: 100 条')
"
```

### 4. 数据备份

处理完成后备份原始和处理后的数据：

```bash
# 备份到云存储或其他位置
cp -r ~/data/gsm8k ~/data/backups/gsm8k_$(date +%Y%m%d)
```

---

**创建时间**: 2026-01-28
**适用版本**: verl v0.2+
