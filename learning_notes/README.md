# verl 学习笔记

> 从零开始掌握 verl - LLM 强化学习训练框架

---

## 📚 学习笔记目录

### ✅ 第一部分：快速上手
📁 **01_快速上手/** - [进入目录](01_快速上手/)

**内容概览：**
- 📖 3 个深度学习笔记（超 40000 字）
- 🛠️ 3 个核心脚本
- 📋 详细的检查清单

**核心文件：**
- `01_快速上手.md` - 基础入门（8000+ 字）
- `ray_trainer_详解.md` - 训练主循环深度解析（新！）
- `配置系统详解.md` - Hydra 配置深度解析（新！）
- `check_env.sh` - 环境检查
- `check_data.py` - 验证数据格式
- `run_first_training.sh` - 第一次训练

**学习目标：**
- ✅ 环境安装与验证
- ✅ 理解训练流程的 7 个阶段
- ✅ 掌握 RayPPOTrainer 架构
- ✅ 熟练使用 Hydra 配置系统
- ✅ 运行第一次训练并监控

---

### ✅ 第二部分：数据准备
📁 **02_数据准备/** - [进入目录](02_数据准备/)

**内容概览：**
- 📖 2 个深度学习笔记（超 35000 字）
- 🛠️ 1 个核心脚本
- 📋 详细的检查清单

**核心文件：**
- `02_数据准备.md` - 数据格式详解（10000+ 字）
- `reward_系统详解.md` - Reward 系统深度解析（新！）
- `data_quality_check.py` - 数据质量检查

**学习目标：**
- ✅ 理解 verl 数据格式（Parquet + 4 种 prompt 格式）
- ✅ 掌握 RewardManager 架构和调用流程
- ✅ 深入理解 GSM8K Reward 实现
- ✅ 准备单轮和多轮对话数据
- ✅ 实现自定义 Reward 函数

---

### ✅ 第三部分：RL 算法
📁 **03_RL算法/** - [进入目录](03_RL算法/)

**内容概览：**
- 📖 3 个深度学习笔记（超 50000 字）
- 📋 详细的检查清单和对比表

**核心文件：**
- `03_RL算法概览.md` - 算法对比与选择指南
- `GRPO_详解.md` - GRPO 算法深度解析（新！）
- `PPO_详解.md` - PPO 算法深度解析（新！）

**学习目标：**
- ✅ 理解 GRPO、PPO、RLOO 等算法原理
- ✅ 掌握 GRPO 的分组机制和优势计算
- ✅ 掌握 PPO 的 GAE 和 Clipping 机制
- ✅ 能够根据任务选择合适的算法
- ✅ 熟练配置和切换不同算法

### ✅ 第四部分：Reward 设计
📁 **04_Reward设计/** - [进入目录](04_Reward设计/)

**内容概览：**
- 📖 详细学习笔记（超 40000 字）
- 🛠️ 10+ 个实战 Reward 示例
- 📋 完整的调试和优化指南

**核心文件：**
- `自定义Reward实践指南.md` - 完整实战教程（新！）

**学习目标：**
- ✅ 深入理解 Reward 函数的作用和设计原理
- ✅ 掌握 3 种 Reward 类型（Rule-based, Model-based, Sandbox）
- ✅ 实现 10+ 种自定义 Reward 函数
- ✅ 掌握 Reward Shaping 和多目标优化技巧
- ✅ 能够调试和优化 Reward 计算

---

### ✅ 第五部分：Agent RL
📁 **05_Agent_RL/** - [进入目录](05_Agent_RL/)

**内容概览：**
- 📖 2 个深度学习笔记（超 45000 字）
- 📋 完整的调试和最佳实践指南

**核心文件：**
- `Agent_Loop详解.md` - Agent Loop 系统深度解析（新！）
- `README.md` - Agent RL 概览和学习路径

**学习目标：**
- ✅ 理解 Agent Loop 架构（AsyncLLMServerManager + Workers）
- ✅ 掌握工具调用机制和生命周期
- ✅ 理解多轮对话的 Token-based API 设计
- ✅ 掌握 response_mask 和 trajectory 一致性
- ✅ 熟练调试和优化 Agent RL 训练

---

### 📝 全部完成！

---

## 🚀 快速开始

### 从第一部分开始

```bash
# 1. 进入第一部分目录
cd learning_notes/01_快速上手

# 2. 阅读 README
cat README.md

# 3. 检查环境
bash check_env.sh

# 4. 开始学习
# 详见 01_快速上手/README.md
```

---

## 📖 学习路线

### 推荐学习顺序

```
第 1-2 天：快速上手
    ↓
第 3-4 天：数据准备
    ↓
第 5-7 天：RL 算法
    ↓
第 8-11 天：Agent RL
    ↓
进阶：自定义任务
```

### 时间安排

| 阶段 | 时间 | 核心内容 |
|------|------|---------|
| **01 快速上手** | 1-2 天 | 环境安装、第一次训练 |
| **02 数据准备** | 1-2 天 | 数据格式、自定义数据 |
| **03 RL 算法** | 2-3 天 | GRPO/PPO、参数调优 |
| **04 Reward 设计** | 1 天 | 自定义 Reward 函数 |
| **05 Agent RL** | 3-5 天 | 工具调用、Agent Loop |

---

## 📊 学习进度跟踪

### 第 01 部分：快速上手 ✓
- [ ] 完成环境安装
- [ ] 下载模型和数据
- [ ] 完成第一次训练
- [ ] 学会使用 TensorBoard
- [ ] 评估训练效果

### 第 02 部分：数据准备 ✓
- [ ] 理解 Parquet 格式
- [ ] 准备单轮数据
- [ ] 准备多轮数据
- [ ] 实现数据检查

### 第 03 部分：RL 算法 ✓
- [ ] 理解 GRPO 算法原理和实现
- [ ] 理解 PPO 算法和 GAE
- [ ] 对比不同算法的优缺点
- [ ] 实现算法切换和参数调优

### 第 04 部分：Reward 设计 ✓
- [ ] 理解 Reward 函数基础和签名
- [ ] 掌握 RewardManager 调用流程
- [ ] 实现 Rule-based Reward
- [ ] 了解 Model-based 和 Sandbox Reward
- [ ] 掌握 Reward Shaping 技巧

### 第 05 部分：Agent RL
- [ ] 理解 Agent Loop
- [ ] 准备工具调用数据
- [ ] 训练第一个 Agent
- [ ] 实现自定义 Agent Loop

---

## 🎯 学习目标

完成所有部分后，你将能够：

✅ 独立搭建 verl 训练环境
✅ 准备和处理各种格式的训练数据
✅ 使用不同的 RL 算法训练模型
✅ 设计和实现自定义 Reward 函数
✅ 训练多轮对话和工具调用的 Agent
✅ 调优训练参数获得更好效果
✅ 部署和评估训练后的模型

---

## 🛠️ 全局工具脚本

### 环境管理
```bash
# 检查环境（任意目录）
bash 01_快速上手/check_env.sh
```

### 数据管理
```bash
# 检查数据格式
python 01_快速上手/check_data.py ~/data/gsm8k/train.parquet
python 02_数据准备/data_quality_check.py ~/data/gsm8k/train.parquet
```

### 训练管理
```bash
# 第一次训练
bash 01_快速上手/run_first_training.sh
```

---

## 📚 学习资源

### 本地文档

**基础文档：**
- **CLAUDE.md** - 项目概览和架构说明（面向开发者）
- **LEARNING_GUIDE.md** - 完整学习路线
  - 第 1-5 节：应用层面（数据准备、算法使用、Agent 训练）
  - 第 6 节：训练流程深度解析（RayPPOTrainer 架构）⭐
  - 第 7 节：Reward 系统深度解析（4 种 RewardManager）⭐
- **README.md** - 项目 README

**学习笔记：**
- `learning_notes/01_快速上手/` - 快速上手笔记和脚本
- `learning_notes/02_数据准备/` - 数据准备笔记和脚本

### 官方资源
- [verl 文档](https://verl.readthedocs.io/en/latest/)
- [GitHub 仓库](https://github.com/volcengine/verl)
- [HybridFlow 论文](https://arxiv.org/abs/2409.19256)

### 社区资源
- [Slack 社区](https://join.slack.com/t/verl-project)
- [GitHub Issues](https://github.com/volcengine/verl/issues)
- [GitHub Discussions](https://github.com/volcengine/verl/discussions)

---

## 💡 学习建议

### 学习方法
1. **边学边做**：每个知识点都要实际运行代码
2. **做好笔记**：记录遇到的问题和解决方法
3. **实验对比**：尝试不同参数，观察效果变化
4. **查阅文档**：遇到问题先查官方文档
5. **参与社区**：在 GitHub 上提问和分享经验

### 时间安排
- **每天 2-4 小时**持续学习
- **完整时间块**用于训练（训练时长较长）
- **碎片时间**阅读文档和笔记

### 硬件建议
- **最低配置**：1 张 24GB GPU（如 RTX 3090）
- **推荐配置**：2-4 张 40GB GPU（如 A100）
- **存储空间**：至少 100GB

---

## ❓ 常见问题

### Q1: 我应该从哪里开始？
从 `01_快速上手/` 开始，按顺序学习各个部分。

### Q2: 我没有 GPU 可以学习吗？
可以阅读笔记理解原理，但实际训练需要 GPU。可以考虑使用云服务（AWS、阿里云等）。

### Q3: 学习需要多长时间？
- **快速上手**：1-2 天
- **基础掌握**：1-2 周
- **熟练使用**：1 个月
- **深入精通**：2-3 个月

### Q4: 遇到问题怎么办？
1. 查看对应章节的"常见问题"部分
2. 搜索官方文档
3. 在 GitHub Issues 搜索类似问题
4. 加入 Slack 社区提问

---

## 📝 更新日志

### 2026-01-25
- ✅ 创建学习笔记目录结构
- ✅ 完成第 01 部分：快速上手
  - 完整学习笔记（8000+ 字）
  - 深度技术解析（20000+ 字）
  - 实用脚本和检查清单
- ✅ 完成第 02 部分：数据准备
  - 完整学习笔记（10000+ 字）
  - Reward 系统深度解析（15000+ 字）
  - 数据格式详解和示例

### 2026-01-26 ⭐ 最新进展
- ✅ **LEARNING_GUIDE.md** 添加深度解析章节
  - 第 6 节：训练流程深度解析（RayPPOTrainer）
  - 第 7 节：Reward 系统架构深度解析
- ✅ **CLAUDE.md** 全面更新
  - 扩展训练命令示例（RL/SFT/评估/生成）
  - 详细的算法对比表（12+ 算法）
  - Inference Backend 详细说明
  - Examples 目录结构说明
  - 最新功能和改进
- ✅ **完成第 03 部分：RL 算法**（超 50000 字）
  - 算法概览与决策树
  - GRPO 深度解析（源码级别）
  - PPO 深度解析（GAE + Clipping）
  - 算法对比和配置切换指南
- ✅ **PPO_详解.md 深度问答** ⭐
  - 3.6 节：old_log_prob vs new_log_prob 与 Importance Sampling
    - 两个 log_prob 的计算时机和作用
    - Importance sampling ratio 详细公式推导
    - 三种策略模式对比（Decoupled/Bypass/Rollout Correction）
    - 完整示例：追踪一个 batch 的训练过程
  - 6.4 节：ref_log_probs 计算与 KL 双重惩罚机制
    - ref_log_probs 的两种实现方式（LoRA/独立模型）
    - use_kl_in_reward vs use_kl_loss 详细对比
    - KL in Reward 和 KL in Loss 的完整训练流程
    - 三种使用场景建议（RLHF/强正则化/双重约束）
- ✅ **reward_系统详解.md 深度问答** ⭐
  - 2.3 节：Reward Token 放置机制与 Advantage 广播
    - 为什么 Reward 只在最后一个 token？（Outcome Supervision 设计）
    - GAE 如何处理：递归反向传播（TD-error 链）
    - GRPO 如何处理：显式广播（unsqueeze + mask）
    - response_mask 的关键作用（过滤 padding）
    - 完整数据流图示：从 Reward 到 Loss 的全流程
    - GAE vs GRPO 对比表（6 个维度详细比较）
- ✅ **完成第 04 部分：Reward 设计**（超 40000 字）
  - Reward 函数基础和设计原理
  - RewardManager 完整调用流程（源码级）
  - 3 种 Reward 类型详解（Rule-based, Model-based, Sandbox）
  - 10+ 实战示例（从简单到复杂）
  - Reward Shaping 和多目标优化
  - 完整的调试验证流程
- ✅ **完成第 05 部分：Agent RL**（超 45000 字）⭐
  - Agent Loop 系统架构深度解析
  - 工具调用完整流程（create → execute → calc_reward → release）
  - Token-based API vs Chat Completion API
  - 多轮对话的 Trajectory 一致性保证
  - response_mask 机制详解（Loss 计算、Advantage 广播）
  - 端到端训练流程追踪（GSM8K Tool Agent）
  - 完整的调试技巧（Reward=0、PPO ratio 爆炸、response_mask 错误）
  - 最佳实践（System Prompt 设计、工具设计原则、Reward Shaping）
  - 超参数调优指南

### 待完成
- 🎉 **所有部分已完成！**

---

## 🎓 结语

verl 是一个强大的 LLM RL 训练框架。通过系统学习这些笔记，你将掌握从数据准备到模型训练的完整流程，并能够应用到实际项目中。

**记住：实践是最好的老师！** 不要只是阅读，一定要动手运行代码，做实验，遇到问题就解决问题。

祝学习顺利！🚀

---

*创建时间: 2026-01-25*
*当前进度: 🎉 全部完成 - 所有 5 个部分已完成（超 210000 字）*
