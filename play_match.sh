#!/bin/bash
# AlphaZero vs Monte Carlo 对弈脚本

# 设置环境变量
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="$(python3 -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib:$DYLD_LIBRARY_PATH"

# 进入 backend 目录
cd backend

# 默认参数
MODEL_PATH="../data/az_model.pt"
NUM_GAMES=10
MC_SIMS=500

# 解析参数
if [ $# -ge 1 ]; then
    MODEL_PATH="$1"
fi

if [ $# -ge 2 ]; then
    NUM_GAMES="$2"
fi

if [ $# -ge 3 ]; then
    MC_SIMS="$3"
fi

echo "🎮 AlphaZero vs Monte Carlo"
echo "Model: $MODEL_PATH"
echo "Games: $NUM_GAMES"
echo "Monte Carlo simulations: $MC_SIMS"
echo ""

# 运行对弈
cargo run --release --features alphazero --bin play_match -- "$MODEL_PATH" "$NUM_GAMES" "$MC_SIMS"
