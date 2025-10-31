#!/usr/bin/env python3
"""
将 Rust tch 保存的模型转换为兼容格式
"""
import torch
import sys

if len(sys.argv) < 3:
    print("Usage: python convert_model.py <input.pt> <output.pt>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

print(f"Converting {input_path} to {output_path}...")

# 加载 TorchScript 模型
try:
    model = torch.jit.load(input_path, map_location='cpu')
    print("Loaded as TorchScript model")

    # 提取状态字典
    state_dict = {}
    for name, param in model.named_parameters():
        state_dict[name] = param.data

    print(f"Found {len(state_dict)} parameters")

    # 保存为普通状态字典
    torch.save(state_dict, output_path)
    print(f"✅ Converted model saved to {output_path}")

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
