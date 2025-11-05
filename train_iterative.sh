#!/bin/bash
# 持续迭代训练 AlphaZero Connect4 模型
# 用法: ./train_iterative.sh [起始代数] [总代数]

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置参数
START_GEN=${1:-0}           # 起始代数（默认0）
MAX_GENS=${2:-50}           # 最大代数（默认50）
GAMES_PER_GEN=100           # 每代自我对弈游戏数
TRAIN_ITERS=300             # 每代训练迭代数
MODEL_DIR="data/generations"
BEST_MODEL="data/best_model.pt"
CHECKPOINT_INTERVAL=5       # 每N代保存一次完整检查点

# 设置 PyTorch 库路径
export DYLD_LIBRARY_PATH="$(python3 -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib:$DYLD_LIBRARY_PATH"
export LIBTORCH_USE_PYTORCH=1

# 创建目录
mkdir -p "$MODEL_DIR"
mkdir -p "data/checkpoints"
mkdir -p "data/logs"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}🚀 AlphaZero Connect4 迭代训练${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${GREEN}配置:${NC}"
echo "  起始代数: $START_GEN"
echo "  目标代数: $MAX_GENS"
echo "  每代游戏数: $GAMES_PER_GEN"
echo "  每代训练迭代: $TRAIN_ITERS"
echo "  模型目录: $MODEL_DIR"
echo ""

# 进入 backend 目录
cd backend

# 如果是从0开始，初始化第一个模型
if [ $START_GEN -eq 0 ]; then
    echo -e "${YELLOW}📝 初始化第0代（随机模型）...${NC}"
    INIT_MODEL="../$MODEL_DIR/gen_0000.pt"
    INIT_TEMP="../$MODEL_DIR/gen_0000_temp.pt"

    # 运行1次训练以生成初始模型
    cargo run --release --features alphazero --bin train_alphazero "$INIT_TEMP" 10 50 1 2>&1 | tee "../data/logs/gen_0000.log"

    # 转换为兼容格式
    echo -e "${BLUE}🔄 转换第0代模型格式...${NC}"
    python3 ../convert_model.py "$INIT_TEMP" "$INIT_MODEL"
    rm "$INIT_TEMP"

    cp "$INIT_MODEL" "../$BEST_MODEL"
    echo -e "${GREEN}✅ 第0代模型已创建并转换${NC}"
    echo ""
    START_GEN=1
fi

# 主训练循环
for GEN in $(seq $START_GEN $MAX_GENS); do
    GEN_NUM=$(printf "%04d" $GEN)
    PREV_GEN_NUM=$(printf "%04d" $((GEN - 1)))

    PREV_MODEL="../$MODEL_DIR/gen_${PREV_GEN_NUM}.pt"
    CURRENT_MODEL="../$MODEL_DIR/gen_${GEN_NUM}.pt"
    LOG_FILE="../data/logs/gen_${GEN_NUM}.log"

    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}📍 训练第 ${GEN}/${MAX_GENS} 代${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${YELLOW}父模型: gen_${PREV_GEN_NUM}.pt${NC}"
    echo -e "${YELLOW}目标模型: gen_${GEN_NUM}.pt${NC}"
    echo ""

    # 检查父模型是否存在
    if [ ! -f "$PREV_MODEL" ]; then
        echo -e "${RED}❌ 错误: 父模型不存在: $PREV_MODEL${NC}"
        exit 1
    fi

    # 复制父模型作为起点（train_alphazero会自动加载它并继续训练）
    cp "$PREV_MODEL" "$CURRENT_MODEL"

    # 开始时间
    START_TIME=$(date +%s)

    # 运行训练（1个epoch，train_alphazero会检测到模型存在并自动加载）
    echo -e "${GREEN}🎮 开始自我对弈和训练...${NC}"
    cargo run --release --features alphazero --bin train_alphazero \
        "$CURRENT_MODEL" \
        $GAMES_PER_GEN \
        $TRAIN_ITERS \
        1 \
        2>&1 | tee "$LOG_FILE"    # 计算耗时
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))

    echo -e "${GREEN}✅ 第${GEN}代训练完成 (用时: ${MINUTES}分${SECONDS}秒)${NC}"

    # 转换模型为兼容格式（重要：下一代将从转换后的模型继续训练）
    echo -e "${BLUE}🔄 转换模型格式以确保兼容性...${NC}"
    TEMP_MODEL="${CURRENT_MODEL}.temp"
    cp "$CURRENT_MODEL" "$TEMP_MODEL"

    python3 ../convert_model.py "$TEMP_MODEL" "$CURRENT_MODEL" 2>&1 | grep -E "✅|❌|Converting"

    if [ $? -eq 0 ]; then
        rm "$TEMP_MODEL"
        echo -e "${GREEN}   模型格式转换成功${NC}"
    else
        echo -e "${YELLOW}   ⚠️ 模型转换失败，使用原始格式（可能影响下一代加载）${NC}"
        mv "$TEMP_MODEL" "$CURRENT_MODEL"
    fi

    # 提取最终损失（从日志中）
    FINAL_LOSS=$(grep "Best val loss:" "$LOG_FILE" | tail -1 | awk '{print $4}')
    if [ -n "$FINAL_LOSS" ]; then
        echo -e "${GREEN}   最佳验证损失: ${FINAL_LOSS}${NC}"
        echo "$GEN,$FINAL_LOSS,$DURATION" >> "../data/training_history.csv"
    fi

    # 每隔一定代数进行评估对战
    if [ $((GEN % 5)) -eq 0 ]; then
        echo ""
        echo -e "${BLUE}🎯 评估第${GEN}代 vs 第${PREV_GEN_NUM}代${NC}"

        # 对战测试（6局快速测试）
        # 注意：这需要 play_match 支持两个模型对战，暂时注释
        # ./play_arena.sh "$CURRENT_MODEL" "$PREV_MODEL" 6

        echo -e "${YELLOW}   (跳过对战评估，直接接受新模型)${NC}"
    fi

    # 更新最佳模型
    cp "$CURRENT_MODEL" "../$BEST_MODEL"

    # 定期保存检查点
    if [ $((GEN % CHECKPOINT_INTERVAL)) -eq 0 ]; then
        CHECKPOINT="../data/checkpoints/checkpoint_gen_${GEN_NUM}.pt"
        cp "$CURRENT_MODEL" "$CHECKPOINT"
        echo -e "${GREEN}💾 检查点已保存: $CHECKPOINT${NC}"
    fi

    # 清理旧的中间模型（保留每10代的模型）
    if [ $((GEN % 10)) -ne 0 ] && [ $GEN -gt 10 ]; then
        OLD_GEN=$((GEN - 5))
        if [ $((OLD_GEN % 10)) -ne 0 ]; then
            OLD_GEN_NUM=$(printf "%04d" $OLD_GEN)
            OLD_MODEL="../$MODEL_DIR/gen_${OLD_GEN_NUM}.pt"
            if [ -f "$OLD_MODEL" ]; then
                rm "$OLD_MODEL"
                echo -e "${YELLOW}🗑️  清理旧模型: gen_${OLD_GEN_NUM}.pt${NC}"
            fi
        fi
    fi

    echo ""
done

cd ..

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}🎉 训练完成！${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}最终代数: $MAX_GENS${NC}"
echo -e "${GREEN}最佳模型: $BEST_MODEL${NC}"
echo ""
echo -e "${BLUE}📊 查看训练历史:${NC}"
echo "  cat data/training_history.csv"
echo ""
echo -e "${BLUE}🎮 测试最佳模型:${NC}"
echo "  ./play_match.sh $BEST_MODEL 10 500"
echo ""
