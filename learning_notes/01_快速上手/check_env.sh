#!/bin/bash
# 环境验证脚本

echo "========================================="
echo "verl 环境验证脚本"
echo "========================================="
echo ""

# 1. 检查 verl
echo "1. 检查 verl..."
python -c "import verl; print('✓ verl 已安装:', verl.__file__)" 2>/dev/null || echo "✗ verl 未安装"

# 2. 检查 Ray
echo "2. 检查 Ray..."
python -c "import ray; print('✓ Ray 版本:', ray.__version__)" 2>/dev/null || echo "✗ Ray 未安装"

# 3. 检查 vLLM
echo "3. 检查 vLLM..."
python -c "import vllm; print('✓ vLLM 版本:', vllm.__version__)" 2>/dev/null || echo "✗ vLLM 未安装"

# 4. 检查 PyTorch 和 CUDA
echo "4. 检查 PyTorch..."
python -c "import torch; print('✓ PyTorch 版本:', torch.__version__); print('✓ CUDA 可用:', torch.cuda.is_available()); print('✓ GPU 数量:', torch.cuda.device_count())" 2>/dev/null || echo "✗ PyTorch 未安装"

# 5. 检查 transformers
echo "5. 检查 Transformers..."
python -c "from transformers import __version__; print('✓ Transformers 版本:', __version__)" 2>/dev/null || echo "✗ Transformers 未安装"

# 6. 检查 pandas
echo "6. 检查 Pandas..."
python -c "import pandas; print('✓ Pandas 版本:', pandas.__version__)" 2>/dev/null || echo "✗ Pandas 未安装"

echo ""
echo "========================================="
echo "环境检查完成"
echo "========================================="
