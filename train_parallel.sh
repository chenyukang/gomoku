#!/bin/bash
# 并行训练 - 使用多进程生成自对弈数据

set -e

MODEL_PATH="data/az_parallel.pt"
NUM_GAMES_TOTAL=${1:-50}
NUM_WORKERS=${2:-4}        # 并行进程数
NUM_ITERS=${3:-200}

echo "🚀 Parallel AlphaZero Training"
echo "   Total Games: $NUM_GAMES_TOTAL"
echo "   Workers: $NUM_WORKERS"
echo "   Iterations: $NUM_ITERS"
echo "   Model: $MODEL_PATH"
echo ""

# 设置环境变量
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="/Users/yukang/.local/share/mise/installs/python/3.13.3/lib/python3.13/site-packages/torch/lib:$DYLD_LIBRARY_PATH"

# 计算每个worker的游戏数
GAMES_PER_WORKER=$((NUM_GAMES_TOTAL / NUM_WORKERS))
EXTRA_GAMES=$((NUM_GAMES_TOTAL % NUM_WORKERS))

echo "🎮 Generating training data with $NUM_WORKERS parallel workers..."
echo "   Games per worker: $GAMES_PER_WORKER"
if [ $EXTRA_GAMES -gt 0 ]; then
    echo "   Extra games: $EXTRA_GAMES (for worker 0)"
fi
echo ""

# 启动并行训练
START_TIME=$(date +%s)

# 为每个worker生成临时模型
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    GAMES=$GAMES_PER_WORKER
    if [ $i -eq 0 ]; then
        GAMES=$((GAMES_PER_WORKER + EXTRA_GAMES))
    fi

    TEMP_MODEL="data/temp_worker_${i}.pt"

    (
        cd backend
        echo "  Worker $i: Starting $GAMES games..."
        cargo run --release --bin train_alphazero --features alphazero -- ../$TEMP_MODEL $GAMES 0 > /dev/null 2>&1
        echo "  Worker $i: ✅ Complete"
    ) &
done

# 等待所有worker完成
wait

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "✅ All workers complete in ${ELAPSED}s"
echo "   Speed: $(echo "scale=2; $NUM_GAMES_TOTAL / $ELAPSED" | bc) games/s"

# 合并所有训练数据并训练最终模型
echo ""
echo "🎓 Training final model with $NUM_ITERS iterations..."

cd backend
cargo run --release --bin train_alphazero --features alphazero -- ../$MODEL_PATH $NUM_GAMES_TOTAL $NUM_ITERS
cd ..

# 清理临时文件
rm -f data/temp_worker_*.pt

# 转换模型
echo ""
echo "🔄 Converting model format..."
python3 convert_model.py $MODEL_PATH ${MODEL_PATH%.pt}_converted.pt

echo ""
echo "✅ Parallel training complete!"
echo "   Model: ${MODEL_PATH%.pt}_converted.pt"
echo "   Total time: ${ELAPSED}s + training time"
echo ""
echo "📊 Test the model:"
echo "   ./play_match.sh ${MODEL_PATH%.pt}_converted.pt 10 500"
