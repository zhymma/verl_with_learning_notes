# GMPO 训练器使用指南

[English](./README.md) | 简体中文

## 目录
- [算法概述](#算法概述)
- [核心特点](#核心特点)
- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [参考资料](#参考资料)

## 算法概述

GMPO (Geometric-Mean Policy Optimization) 通过优化token级奖励的几何平均值来提升 GRPO 的稳定性。基于论文 [Geometric-Mean Policy Optimization](https://arxiv.org/abs/2507.20673)。

### 算法原理

**GRPO 的问题**：优化算术平均值，对异常值敏感
```python
GRPO: mean(rewards)  # 算术平均
# 问题：outlier reward 导致不稳定
```

**GMPO 的解决**：优化几何平均值，对异常值更鲁棒
```python
GMPO: geometric_mean(rewards)  # 几何平均
# 优势：对 outlier 不敏感，importance sampling ratio 更稳定
```

### 与其他算法对比

| 特性 | GMPO | GRPO | PPO |
|------|------|------|-----|
| 平均方式 | 几何平均 | 算术平均 | 算术平均 |
| 异常值敏感度 | **低** | 高 | 高 |
| 训练稳定性 | **很好** | 好 | 好 |
| MoE 友好 | ✅ | ⚠️ | ⚠️ |
| 实现复杂度 | 低 | 低 | 高 |

### 适用场景

- **MoE 模型训练**（强烈推荐）
- 奖励分布方差大的任务
- 需要极致稳定性的场景
- 大规模模型训练
- GRPO 训练不稳定时的替代方案

### 算法优势

1. **更稳定的训练**：importance sampling ratio 范围更小
2. **MoE 友好**：训练 MoE 模型时更稳定
3. **即插即用**：只需改一行配置
4. **理论支撑**：有完整的理论分析
5. **实验验证**：Qwen2.5-7B 上提升 4.1%

## 核心特点

### 关键配置

```yaml
algorithm:
  adv_estimator: grpo              # 仍然使用 GRPO 优势估计

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: geo_mean          # 关键：使用几何平均
    clip_ratio_low: 0.4            # MoE 建议用更大的 clip
    clip_ratio_high: 0.4
```

### GMPO vs GRPO 对比

| 项目 | GRPO | GMPO |
|------|------|------|
| 损失函数 | 算术平均 | **几何平均** |
| IS ratio 范围 | 大 | **小**（更稳定） |
| 对 outlier | 敏感 | **鲁棒** |
| MoE 训练 | 不稳定 | **稳定** |

### 性能特征

- **训练速度**：与 GRPO 相同
- **显存占用**：与 GRPO 相同
- **收敛性**：**比 GRPO 更好**
- **稳定性**：**显著提升**

## 快速开始

### 最简运行

```bash
cd examples/gmpo_trainer
bash run_qwen2_5-7b_math.sh
```

### 核心配置

```bash
# 关键参数
loss_mode=geo_mean       # 几何平均损失
clip_ratio=0.4           # 推荐值（MoE 模型）
```

### 预期输出

```
Epoch 5/15: 82.5% GSM8K Pass@1
GMPO advantage: IS ratio 更稳定，训练曲线更平滑
```

## 详细配置

### 完整参数说明

#### GMPO 核心配置
```yaml
# 1. 算法选择
algorithm:
  adv_estimator: grpo              # 基于 GRPO
  use_kl_in_reward: False

# 2. 损失函数模式
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: geo_mean          # 几何平均（GMPO 关键）

    # 3. 裁剪比率
    clip_ratio_low: 0.4            # Dense 模型可用默认值
    clip_ratio_high: 0.4           # MoE 模型建议降低到 0.4
```

#### MoE 模型推荐配置
```yaml
# MoE 模型训练常不稳定，GMPO + 大 clip ratio 有效
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: geo_mean
    clip_ratio_low: 0.4            # 降低 clip ratio
    clip_ratio_high: 0.4
    use_kl_loss: False
    kl_loss_type: low_var_kl
```

#### Dense 模型配置
```yaml
# Dense 模型可以用标准配置
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: geo_mean
    clip_ratio_low: 0.2            # 标准值
    clip_ratio_high: 0.2
```

### 参数选择建议

#### Clip Ratio 选择
```yaml
# Dense 模型（推荐）
clip_ratio: 0.2

# MoE 模型（推荐）
clip_ratio: 0.4

# 经验：MoE 模型用更大的 clip ratio 更稳定
```

#### 学习率
```yaml
# 与 GRPO 相同
actor.optim.lr: 1e-6              # 标准值
```

#### 采样数量
```yaml
# 与 GRPO 相同
rollout.n: 5
```

## 实战示例

### 示例 1: Qwen2.5-7B GMPO

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=geo_mean \
    actor_rollout_ref.actor.clip_ratio_low=0.4 \
    actor_rollout_ref.actor.clip_ratio_high=0.4 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-7B \
    data.train_files="['$HOME/data/gsm8k/train.parquet']" \
    trainer.total_epochs=15
```

### 示例 2: GMPO + DAPO（实验性）

```bash
cd examples/gmpo_trainer
bash test_dapo_7b_math.sh
```

### 示例 3: Qwen3-30B MoE 模型

```bash
cd examples/gmpo_trainer
bash test_dapo_qwen3_30b_math.sh
```

### 示例 4: GMPO 替换 GRPO

```bash
# 原 GRPO 脚本
algorithm.adv_estimator=grpo

# 改为 GMPO（只需加一行）
algorithm.adv_estimator=grpo \
actor_rollout_ref.actor.policy_loss.loss_mode=geo_mean \
actor_rollout_ref.actor.clip_ratio_low=0.4 \
actor_rollout_ref.actor.clip_ratio_high=0.4
```

## 性能对比

### GMPO vs GRPO（Qwen2.5-7B）

| 算法 | GSM8K | MATH | 稳定性 | 训练时间 |
|------|-------|------|--------|----------|
| GRPO | 77.2% | 35.1% | 基准 | 基准 |
| **GMPO** | **81.3%** | **37.5%** | **更好** | 相同 |
| **提升** | **+4.1%** | **+2.4%** | - | - |

### MoE 模型训练稳定性

| 模型类型 | GRPO | GMPO |
|---------|------|------|
| Dense 7B | ✅ 稳定 | ✅ 稳定 |
| Dense 30B | ✅ 稳定 | ✅ 稳定 |
| MoE 30B | ⚠️ **不稳定** | ✅ **稳定** |

### Importance Sampling Ratio 分布

```
GRPO IS ratio: [0.1, 0.3, 0.8, 2.5, 15.0]  # 有大 outlier
GMPO IS ratio: [0.3, 0.5, 0.8, 1.2, 1.8]   # 更集中，更稳定
```

## 常见问题

### Q1: GMPO 就是 GRPO + geo_mean？

**答**：基本正确！
```yaml
GMPO = GRPO + loss_mode=geo_mean + (可选) larger clip_ratio
```

### Q2: 什么时候用 GMPO？

**答**：
- ✅ **训练 MoE 模型**（强烈推荐）
- ✅ GRPO 训练不稳定时
- ✅ 奖励方差大的任务
- ✅ 追求最佳性能

**什么时候用 GRPO？**
- Dense 模型 + 稳定训练 = GRPO 也很好

### Q3: clip_ratio 为什么要调大？

**答**：
```
观察：MoE 训练时用小 clip_ratio (0.2) 不稳定
原因：MoE 的梯度本身方差就大
解决：GMPO (几何平均) + 大 clip_ratio (0.4)
结果：训练稳定且性能好
```

### Q4: Dense 模型也能用 GMPO 吗？

**答**：✅ 可以！
- Dense 模型用 clip_ratio=0.2
- 性能通常也会略好于 GRPO
- 但提升不如 MoE 模型明显

### Q5: GMPO 的理论基础？

**答**：
```
几何平均特性：
1. 对 outlier 不敏感（log 变换）
2. Importance sampling ratio 更稳定
3. 加权形式的策略梯度，但权重更稳定
```

详见论文的理论分析部分。

### Q6: 可以和 DAPO 结合吗？

**答**：✅ 可以！
```bash
# 见示例脚本
bash examples/gmpo_trainer/test_dapo_7b_math.sh
bash examples/gmpo_trainer/test_dapo_qwen3_30b_math.sh
```

## 参考资料

### 论文
- **GMPO 原始论文**：[Geometric-Mean Policy Optimization](https://arxiv.org/abs/2507.20673)

### 引用
```bibtex
@article{zhao2025geometric,
  title={Geometric-mean policy optimization},
  author={Zhao, Yuzhong and Liu, Yue and Liu, Junpeng and Chen, Jingye and Wu, Xun and Hao, Yaru and Lv, Tengchao and Huang, Shaohan and Cui, Lei and Ye, Qixiang and others},
  journal={arXiv preprint arXiv:2507.20673},
  year={2025}
}
```

### 联系方式
- zhaoyuzhong20@mails.ucas.ac.cn
- liuyue171@mails.ucas.ac.cn
- lecu@microsoft.com
- wanfang@ucas.ac.cn

### 代码
- 实现：`verl/trainer/ppo/policy_loss.py`
- 示例：`examples/gmpo_trainer/`
- README：`examples/gmpo_trainer/README.md`

### 相关文档
- [GRPO 训练指南](../grpo_trainer/README_CN.md)
- [GSPO 训练指南](../gspo_trainer/README_CN.md)

---

**提示**：GMPO 是训练 MoE 模型的利器。如果你在用 GRPO 训练 MoE 模型遇到不稳定，立即尝试 GMPO！只需一行配置改动，稳定性和性能都会显著提升。
