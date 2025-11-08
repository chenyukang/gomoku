#!/usr/bin/env python3
"""测试转换后的模型是否正确"""

import torch
import torch.nn.functional as F

# 加载转换后的模型
model_path = "connect4_resnet_converted.pt"
state_dict = torch.load(model_path, map_location='cpu')

print(f"模型参数数量: {len(state_dict)}")
print(f"前5个参数键: {list(state_dict.keys())[:5]}")

# 检查参数是否全是零或者异常值
total_params = 0
zero_params = 0
for key, value in state_dict.items():
    param_count = value.numel()
    total_params += param_count
    zero_count = (value == 0).sum().item()
    zero_params += zero_count
    
    # 打印每个参数的统计信息
    print(f"{key:50s} shape={str(list(value.shape)):20s} mean={value.mean():.6f} std={value.std():.6f} zeros={zero_count}/{param_count}")

print(f"\n总参数: {total_params}, 零参数: {zero_params} ({100*zero_params/total_params:.2f}%)")

# 测试一个简单的输入
print("\n=== 测试模型推理 ===")
print("空棋盘的策略预测应该偏向中间列（column 3）")

# 创建一个空棋盘 [1, 3, 6, 7]
# Channel 0: 当前玩家的棋子 (全0)
# Channel 1: 对手的棋子 (全0)  
# Channel 2: 空位 (全1)
empty_board = torch.zeros(1, 3, 6, 7)
empty_board[0, 2, :, :] = 1.0  # 所有位置都是空的

print(f"输入形状: {empty_board.shape}")
print(f"空位通道 sum: {empty_board[0, 2].sum()}")  # 应该是 42 (6*7)
