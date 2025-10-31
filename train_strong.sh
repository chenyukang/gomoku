#!/bin/bash
# 训练一个较强的 AlphaZero 模型

set -e

MODEL_PATH="data/az_strong.pt"
NUM_GAMES=${1:-50}      # 默认50场游戏
NUM_ITERS=${2:-200}     # 默认200次训练迭代

echo "🚀 Training Strong AlphaZero Model"
echo "   Games: $NUM_GAMES"
echo "   Iterations: $NUM_ITERS"
echo "   Model: $MODEL_PATH"
echo ""

# 设置环境变量
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="/Users/yukang/.local/share/mise/installs/python/3.13.3/lib/python3.13/site-packages/torch/lib:$DYLD_LIBRARY_PATH"
# 设置 PyTorch 线程数以提升 CPU 利用率
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)
export TORCH_NUM_THREADS=$(sysctl -n hw.ncpu)

echo "🔧 Using $(sysctl -n hw.ncpu) CPU threads"

# 训练
cd backend
cargo run --release --bin train_alphazero --features alphazero -- ../$MODEL_PATH $NUM_GAMES $NUM_ITERS
cd ..

# 转换模型
echo ""
echo "🔄 Converting model format..."
python3 convert_model.py $MODEL_PATH ${MODEL_PATH%.pt}_converted.pt

echo ""
echo "✅ Training complete!"
echo "   Model: ${MODEL_PATH%.pt}_converted.pt"
echo ""
echo "📊 Test the model:"
echo "   ./play_match.sh ${MODEL_PATH%.pt}_converted.pt 10 500"
