#!/bin/bash
# 并行训练启动脚本

set -e

# 配置参数
MODEL_PATH=${1:-"model_parallel.pt"}
DATA_DIR=${2:-"./training_data"}
NUM_WORKERS=${3:-4}
GAMES_PER_WORKER=${4:-25}
NUM_ROUNDS=${5:-10}

echo "========================================="
echo "AlphaZero 并行训练"
echo "========================================="
echo "模型路径: $MODEL_PATH"
echo "数据目录: $DATA_DIR"
echo "工作进程数: $NUM_WORKERS"
echo "每进程游戏数: $GAMES_PER_WORKER"
echo "训练轮数: $NUM_ROUNDS"
echo "总游戏数/轮: $((NUM_WORKERS * GAMES_PER_WORKER))"
echo "========================================="
echo ""

# 创建数据目录
mkdir -p "$DATA_DIR"

# 设置 PyTorch 环境变量 (Apple Silicon with Homebrew PyTorch)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export LIBTORCH_USE_PYTORCH=1

# 运行并行训练
cd backend
cargo run --release --bin parallel_trainer --features alphazero -- \
    "$MODEL_PATH" \
    "$DATA_DIR" \
    "$NUM_WORKERS" \
    "$GAMES_PER_WORKER" \
    "$NUM_ROUNDS"

echo ""
echo "========================================="
echo "训练完成！"
echo "模型已保存到: $MODEL_PATH"
echo "========================================="
