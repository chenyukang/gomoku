#!/bin/bash
# AlphaZero 迭代训练脚本（改进版）
# 每轮都生成新数据并训练，避免 loss 停滞

set -e

# 配置
MODEL_PATH="${1:-data/alphazero_iterative.pt}"
NUM_ITERATIONS="${2:-10}"    # 迭代次数
GAMES_PER_ITER="${3:-100}"   # 每轮自我对弈游戏数
TRAIN_ITERS="${4:-500}"      # 每轮训练迭代数

echo "🚀 AlphaZero 迭代训练"
echo "================================"
echo "模型路径: $MODEL_PATH"
echo "迭代次数: $NUM_ITERATIONS"
echo "每轮游戏数: $GAMES_PER_ITER"
echo "每轮训练迭代: $TRAIN_ITERS"
echo "================================"
echo

# 设置环境变量
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="$(python3 -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib:$DYLD_LIBRARY_PATH"

# 进入 backend 目录
cd backend

# 运行训练（使用改进的 train_loop）
echo "📚 开始迭代训练..."
time cargo run --release --bin train_alphazero --features alphazero,random -- \
  "$MODEL_PATH" "$GAMES_PER_ITER" "$TRAIN_ITERS" "$NUM_ITERATIONS"

echo
echo "✅ 训练完成！"
echo
echo "💡 下一步："
echo "   1. 查看检查点: ls -lh data/alphazero_model_iter_*.pt"
echo "   2. 转换模型: python3 convert_model.py $MODEL_PATH"
echo "   3. 测试对战: ./play_match.sh data/alphazero_iterative_converted.pt 10 500"
