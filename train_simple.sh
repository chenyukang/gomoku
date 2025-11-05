#!/bin/bash
# 简单的训练脚本 - 一次性训练多个epochs
# 用法: ./train_simple.sh [epochs] [games_per_epoch] [train_iters]

set -e

EPOCHS=${1:-50}
GAMES=${2:-100}
ITERS=${3:-300}
MODEL_NAME="connect4_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="data/${MODEL_NAME}.pt"
CONVERTED_PATH="data/${MODEL_NAME}_converted.pt"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 设置 PyTorch 库路径
export DYLD_LIBRARY_PATH="$(python3 -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib:$DYLD_LIBRARY_PATH"
export LIBTORCH_USE_PYTORCH=1

# 创建目录
mkdir -p data
mkdir -p data/logs

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}🚀 AlphaZero Connect4 训练${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${GREEN}配置:${NC}"
echo "  总轮数(Epochs): $EPOCHS"
echo "  每轮游戏数: $GAMES"
echo "  每轮训练迭代: $ITERS"
echo "  输出模型: $MODEL_PATH"
echo ""
echo -e "${YELLOW}预计用时: $((EPOCHS * 6 / 60)) - $((EPOCHS * 10 / 60)) 小时${NC}"
echo ""

# 进入 backend 目录
cd backend

# 开始训练
START_TIME=$(date +%s)

echo -e "${GREEN}🎮 开始训练...${NC}"
echo ""

cargo run --release --features alphazero --bin train_alphazero \
    "../$MODEL_PATH" \
    $GAMES \
    $ITERS \
    $EPOCHS \
    2>&1 | tee "../data/logs/${MODEL_NAME}.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

cd ..

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✅ 训练完成！${NC}"
echo -e "${GREEN}================================================${NC}"
echo "  用时: ${HOURS}小时${MINUTES}分钟"
echo "  原始模型: $MODEL_PATH"
echo ""

# 转换模型
echo -e "${BLUE}🔄 转换模型格式...${NC}"
python3 convert_model.py "$MODEL_PATH" "$CONVERTED_PATH"

if [ -f "$CONVERTED_PATH" ]; then
    echo -e "${GREEN}✅ 模型转换成功: $CONVERTED_PATH${NC}"
    echo ""
    echo -e "${BLUE}📊 下一步:${NC}"
    echo "  1. 测试模型:"
    echo "     ./play_match.sh $CONVERTED_PATH 10 500"
    echo ""
    echo "  2. 使用模型（需要先部署Web UI）:"
    echo "     在浏览器中选择该模型文件"
    echo ""
    echo "  3. 继续训练（如果效果不够好）:"
    echo "     ./train_simple.sh $((EPOCHS + 20)) $GAMES $ITERS"
else
    echo -e "${YELLOW}⚠️  模型转换失败，请检查 convert_model.py${NC}"
fi

echo ""
