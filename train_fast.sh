#!/bin/bash
# 快速训练 - 使用更少的MCTS模拟次数

set -e

MODEL_PATH=${1:-"data/az_fast_train.pt"}
NUM_GAMES=${2:-30}
NUM_ITERS=${3:-150}

echo "⚡ Fast AlphaZero Training"
echo "   Games: $NUM_GAMES"
echo "   Iterations: $NUM_ITERS"
echo "   Model: $MODEL_PATH"
echo "   Strategy: Fewer MCTS simulations for faster training"
echo ""

# 设置环境变量
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="/Users/yukang/.local/share/mise/installs/python/3.13.3/lib/python3.13/site-packages/torch/lib:$DYLD_LIBRARY_PATH"
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)

echo "🔧 System: $(sysctl -n hw.ncpu) CPU cores"
echo ""

# 训练
cd backend
time cargo run --release --bin train_alphazero --features alphazero -- ../$MODEL_PATH $NUM_GAMES $NUM_ITERS
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
echo "   ./play_match.sh ${MODEL_PATH%.pt}_converted.pt 6 500"
