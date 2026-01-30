#!/bin/bash
# 第一次训练脚本 - GRPO on GSM8K

# ===== 配置区域 =====
MODEL_PATH=~/models/Qwen2.5-7B-Instruct
DATA_DIR=~/data/gsm8k
OUTPUT_DIR=./outputs/first_training

# GPU 配置
N_GPUS=1  # 修改为你的 GPU 数量

# 训练配置
EPOCHS=3
BATCH_SIZE=256
LEARNING_RATE=1e-6

# ===== 开始训练 =====
echo "========================================"
echo "verl 第一次训练 - GRPO on GSM8K"
echo "========================================"
echo "模型: ${MODEL_PATH}"
echo "数据: ${DATA_DIR}"
echo "GPU 数量: ${N_GPUS}"
echo "训练轮数: ${EPOCHS}"
echo "========================================"
echo ""

# 检查模型是否存在
if [ ! -d "${MODEL_PATH}" ]; then
    echo "错误: 模型路径不存在: ${MODEL_PATH}"
    echo "请先下载模型或修改 MODEL_PATH"
    exit 1
fi

# 检查数据是否存在
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "错误: 数据文件不存在: ${DATA_DIR}/train.parquet"
    echo "请先运行: python examples/data_preprocess/gsm8k.py --local_dir ${DATA_DIR}"
    exit 1
fi

# 运行训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.group_size=4 \
    algorithm.kl_penalty=0.001 \
    data.train_files="['${DATA_DIR}/train.parquet']" \
    data.val_files="['${DATA_DIR}/test.parquet']" \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=0.6 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.total_epochs=${EPOCHS} \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=my_first_verl_training \
    trainer.experiment_name=grpo_gsm8k_qwen7b

echo ""
echo "========================================"
echo "训练完成！"
echo "查看 TensorBoard: tensorboard --logdir=./outputs"
echo "模型保存在: ${OUTPUT_DIR}"
echo "========================================"
