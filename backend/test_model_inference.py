#!/usr/bin/env python3
"""测试模型对不同棋盘状态的推理"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 简单的ResNet块（需要匹配训练时的架构）
class ResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class Connect4ResNet(nn.Module):
    def __init__(self, num_blocks=10, filters=128):
        super().__init__()
        self.conv_init = nn.Conv2d(3, filters, 3, padding=1)
        self.bn_init = nn.BatchNorm2d(filters)
        self.res_blocks = nn.ModuleList([ResBlock(filters) for _ in range(num_blocks)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 6 * 7, 7)  # Connect4: 7列
        
        # Value head  
        self.value_conv = nn.Conv2d(filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(6 * 7, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 6 * 7)
        policy = self.policy_fc(policy)
        
        # Value
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 6 * 7)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

# 加载模型
model = Connect4ResNet(num_blocks=10, filters=128)
state_dict = torch.load('connect4_resnet_converted.pt', map_location='cpu')

# 转换参数名
new_state_dict = {}
for key, value in state_dict.items():
    # 去掉 'net.' 前缀，转换 '|' 为 '.'
    new_key = key.replace('|', '.')
    # 处理 res_X 到 res_blocks.X 的映射
    if new_key.startswith('res_'):
        parts = new_key.split('.')
        block_num = parts[0].split('_')[1]
        new_key = f"res_blocks.{block_num}." + '.'.join(parts[1:])
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict, strict=False)
model.eval()

print("✅ 模型加载成功\n")

# 测试场景
test_cases = [
    ("空棋盘", "0" * 42, "应该偏好中间列(3)"),
    ("中间有一子", "0" * 24 + "1" + "0" * 17, "应该考虑在中间附近下"),
    ("边角有一子", "0" * 6 + "1" + "0" * 35, "不应该过度偏好右边"),
]

for name, board_str, expected in test_cases:
    print(f"=== {name} ===")
    print(f"期望: {expected}")
    
    # 转换为tensor [1, 3, 6, 7]
    board_tensor = torch.zeros(1, 3, 6, 7)
    for i, ch in enumerate(board_str):
        row, col = i // 7, i % 7
        if ch == '1':
            board_tensor[0, 0, row, col] = 1.0  # 当前玩家
        elif ch == '2':
            board_tensor[0, 1, row, col] = 1.0  # 对手
        else:
            board_tensor[0, 2, row, col] = 1.0  # 空位
    
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
        policy_probs = F.softmax(policy_logits, dim=1)[0]
    
    print(f"Value: {value.item():.4f}")
    print(f"Policy probabilities:")
    for col in range(7):
        bar = "█" * int(policy_probs[col].item() * 50)
        print(f"  Col {col}: {policy_probs[col].item():.4f} {bar}")
    
    best_col = policy_probs.argmax().item()
    print(f"最佳列: {best_col} (概率={policy_probs[best_col].item():.4f})")
    print()
