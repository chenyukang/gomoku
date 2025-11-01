#!/bin/bash
# 监控并行训练进度

DATA_DIR=${1:-"./training_data"}

echo "========================================="
echo "AlphaZero 训练监控"
echo "========================================="
echo ""

# 检测工作进程
echo "工作进程状态:"
ps aux | grep "parallel_worker" | grep -v grep || echo "  无活动工作进程"
echo ""

# 统计数据文件
if [ -d "$DATA_DIR" ]; then
    echo "数据目录: $DATA_DIR"

    for worker_dir in "$DATA_DIR"/worker_*; do
        if [ -d "$worker_dir" ]; then
            worker_name=$(basename "$worker_dir")
            file_count=$(ls "$worker_dir"/*.json 2>/dev/null | wc -l | tr -d ' ')
            echo "  $worker_name: $file_count 个数据文件"
        fi
    done

    total_files=$(find "$DATA_DIR" -name "*.json" | wc -l | tr -d ' ')
    echo "  总计: $total_files 个数据文件"
else
    echo "数据目录不存在: $DATA_DIR"
fi

echo ""
echo "========================================="
