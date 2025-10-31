#!/bin/bash
# AlphaZero 训练脚本

set -e

# 设置环境变量
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="$(python3 -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib:$DYLD_LIBRARY_PATH"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AlphaZero Training Script${NC}\n"

# 默认参数
FILTERS=32
BLOCKS=2
GAMES=2
ITERATIONS=20
SIMULATIONS=10
EPOCHS=1
OUTPUT="../data/alphazero_model.pt"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            echo -e "${GREEN}✨ Quick test mode (3 minutes)${NC}"
            FILTERS=32
            BLOCKS=2
            GAMES=2
            ITERATIONS=20
            SIMULATIONS=10
            EPOCHS=1
            shift
            ;;
        --standard)
            echo -e "${GREEN}✨ Standard training mode (1-2 hours)${NC}"
            FILTERS=128
            BLOCKS=10
            GAMES=50
            ITERATIONS=500
            SIMULATIONS=200
            EPOCHS=5
            shift
            ;;
        --quality)
            echo -e "${GREEN}✨ High quality mode (8-12 hours)${NC}"
            FILTERS=256
            BLOCKS=15
            GAMES=200
            ITERATIONS=2000
            SIMULATIONS=800
            EPOCHS=10
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick|--standard|--quality] [--output PATH]"
            exit 1
            ;;
    esac
done

# 确保数据目录存在
mkdir -p data

# 进入 backend 目录
cd backend

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Filters: $FILTERS"
echo "  Blocks: $BLOCKS"
echo "  Games: $GAMES"
echo "  Iterations: $ITERATIONS"
echo "  Simulations: $SIMULATIONS"
echo "  Epochs: $EPOCHS"
echo "  Output: $OUTPUT"
echo ""

# 开始训练
echo -e "${GREEN}🎓 Starting training...${NC}\n"

cargo run --release --features alphazero --bin alphazero_cli -- train \
    --filters $FILTERS \
    --blocks $BLOCKS \
    --games $GAMES \
    --iterations $ITERATIONS \
    --simulations $SIMULATIONS \
    --epochs $EPOCHS \
    --output $OUTPUT

echo -e "\n${GREEN}✅ Training complete!${NC}"

# 测试模型
if [ -f "$OUTPUT" ]; then
    echo -e "\n${BLUE}🧪 Testing model...${NC}\n"
    cargo run --release --features alphazero --bin alphazero_cli -- test "$OUTPUT"

    echo -e "\n${BLUE}📊 Running benchmark...${NC}\n"
    cargo run --release --features alphazero --bin alphazero_cli -- benchmark "$OUTPUT"
fi

echo -e "\n${GREEN}🎉 All done!${NC}"
