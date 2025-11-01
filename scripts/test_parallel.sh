#!/bin/bash
# 并行训练快速测试（小规模）

set -e

echo "========================================="
echo "AlphaZero 并行训练快速测试"
echo "========================================="
echo "配置: 2个工作进程, 每进程5局, 1轮训练"
echo "预计时间: ~2-3分钟"
echo "========================================="
echo ""

# 清理旧数据
rm -rf training_data_test model_test.pt

# 设置环境变量 (Apple Silicon with Homebrew PyTorch)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="/Users/yukang/.local/share/mise/installs/python/3.13.3/lib/python3.13/site-packages/torch/lib:$DYLD_LIBRARY_PATH"

# 运行测试
cd backend
cargo run --release --bin parallel_trainer --features alphazero -- \
    ../model_test.pt \
    ../training_data_test \
    2 \
    5 \
    1

echo ""
echo "========================================="
echo "测试完成！"
echo "检查生成的文件:"
ls -lh ../model_test.pt ../training_data_test/ 2>/dev/null || true
echo "========================================="
