#!/usr/bin/env python3
"""
将训练保存的完整模型转换为只包含权重的模型
"""
import torch
import sys

def convert_model(input_path, output_path):
    print(f"Loading model from {input_path}")
    
    # 尝试加载模型
    try:
        # 方式1: 尝试直接加载 (PyTorch 2.6+ 需要 weights_only=False 来加载 TorchScript)
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        print(f"Model type: {type(checkpoint)}")
        
        # 如果是 TorchScript 模块
        if isinstance(checkpoint, torch.jit.ScriptModule):
            print("Detected TorchScript module, extracting state_dict...")
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict):
            print(f"Keys in checkpoint: {checkpoint.keys()}")
            
            # 如果有 'model_state_dict' 或类似的键
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # 假设整个 checkpoint 就是 state_dict
                state_dict = checkpoint
        else:
            # 如果是 OrderedDict 或直接是 state_dict
            state_dict = checkpoint
        
        print(f"\nSaving weights to {output_path}")
        print(f"Number of parameters: {len(state_dict)}")
        
        # 只保存权重
        torch.save(state_dict, output_path)
        print("✅ Conversion successful!")
        
        # 验证保存的文件
        print("\nVerifying saved file...")
        loaded = torch.load(output_path, map_location='cpu', weights_only=False)
        print(f"Loaded type: {type(loaded)}")
        if isinstance(loaded, dict):
            print(f"Number of keys: {len(loaded)}")
            print(f"Sample keys: {list(loaded.keys())[:5]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_model.py <input_model.pt> <output_model.pt>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    success = convert_model(input_path, output_path)
    sys.exit(0 if success else 1)
