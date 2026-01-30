# 数据检查脚本 - 验证 GSM8K 数据格式

import pandas as pd
import sys
from pathlib import Path

def check_data(data_path):
    """检查数据格式是否正确"""
    print("=" * 60)
    print(f"检查数据文件: {data_path}")
    print("=" * 60)
    print()

    # 检查文件是否存在
    if not Path(data_path).exists():
        print(f"❌ 错误: 文件不存在")
        return False

    try:
        # 读取数据
        df = pd.read_parquet(data_path)
        print(f"✓ 成功读取数据文件")
        print(f"✓ 样本数量: {len(df)}")
        print()

        # 检查必需字段
        required_fields = ['data_source', 'prompt']
        missing_fields = [f for f in required_fields if f not in df.columns]

        if missing_fields:
            print(f"❌ 缺少必需字段: {missing_fields}")
            return False

        print("✓ 字段检查:")
        print(f"  - 字段列表: {df.columns.tolist()}")
        print()

        # 检查第一个样本
        print("✓ 第一个样本:")
        first_sample = df.iloc[0]
        print(f"  - data_source: {first_sample.get('data_source', 'N/A')}")
        print(f"  - prompt (前100字符): {str(first_sample.get('prompt', 'N/A'))[:100]}...")

        if 'reward_model' in first_sample:
            reward_info = first_sample['reward_model']
            if isinstance(reward_info, dict):
                print(f"  - reward_model.ground_truth: {reward_info.get('ground_truth', 'N/A')}")
            else:
                print(f"  - reward_model: {reward_info}")
        print()

        # 统计信息
        print("✓ 统计信息:")
        if df['prompt'].dtype == 'object':
            df['prompt_len'] = df['prompt'].str.len()
            print(f"  - Prompt 长度统计:")
            print(f"    最小: {df['prompt_len'].min()}")
            print(f"    最大: {df['prompt_len'].max()}")
            print(f"    平均: {df['prompt_len'].mean():.0f}")
            print(f"    中位数: {df['prompt_len'].median():.0f}")
        print()

        # 显示更多样本
        print("✓ 前3个样本预览:")
        for idx in range(min(3, len(df))):
            row = df.iloc[idx]
            print(f"\n  样本 {idx + 1}:")
            print(f"  问题: {str(row['prompt'])[:80]}...")
            if 'reward_model' in row and isinstance(row['reward_model'], dict):
                print(f"  答案: {row['reward_model'].get('ground_truth', 'N/A')}")

        print()
        print("=" * 60)
        print("✓ 数据格式检查通过！")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 默认检查路径
    default_paths = [
        "~/data/gsm8k/train.parquet",
        "~/data/gsm8k/test.parquet",
    ]

    # 如果命令行提供了路径，使用命令行参数
    paths = sys.argv[1:] if len(sys.argv) > 1 else default_paths

    all_passed = True
    for path in paths:
        # 展开 ~ 路径
        expanded_path = Path(path).expanduser()
        if not check_data(str(expanded_path)):
            all_passed = False
        print()

    if all_passed:
        print("✓ 所有数据文件检查通过！")
        sys.exit(0)
    else:
        print("❌ 部分数据文件检查失败")
        sys.exit(1)
