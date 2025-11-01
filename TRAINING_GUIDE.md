# AlphaZero 迭代训练指南

## 🎯 解决 Loss 停滞问题

如果你发现训练时 loss 下降到一定程度后就不再降低，这是因为：

1. **数据质量固化**：一直用同样的数据训练，网络已经"记住"了
2. **缺乏新策略**：没有用最新的网络生成新的对弈数据

## ✅ 解决方案：迭代训练

改进后的 `train_alphazero` 现在支持**迭代训练**：

### 训练流程

```
每轮迭代 (Epoch):
  1. 🎮 自我对弈：用当前网络生成新的训练数据
  2. 🎓 训练：用新数据训练网络
  3. 💾 保存：每5轮保存一次检查点

→ 循环往复，loss 持续下降
```

## 📝 使用方法

### 基本用法

```bash
cd backend
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="$(python3 -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib:$DYLD_LIBRARY_PATH"

# 语法：
cargo run --release --bin train_alphazero --features alphazero,random -- \
  <模型路径> [每轮游戏数] [每轮训练次数] [迭代轮数]

# 快速测试（5分钟）
cargo run --release --bin train_alphazero --features alphazero,random -- \
  ../data/az_test.pt 50 300 5

# 标准训练（30分钟-1小时）
cargo run --release --bin train_alphazero --features alphazero,random -- \
  ../data/az_model.pt 100 500 10

# 完整训练（2-4小时）
cargo run --release --bin train_alphazero --features alphazero,random -- \
  ../data/az_strong.pt 200 1000 20
```

### 使用脚本

```bash
# 使用提供的训练脚本
./train_iterative.sh data/az_model.pt 10 100 500
```

## 📊 训练输出示例

```
🚀 AlphaZero Training Pipeline (Improved)
============================================================

Configuration:
  Iterations: 10
  Games per iteration: 100
  Training iterations: 500
  MCTS simulations: 25
  Replay buffer size: 100000
  Current buffer samples: 0

============================================================
📍 Iteration 1/10
============================================================

🎮 Phase 1: Self-Play
🎮 Generating 100 self-play games...
  Progress: 50/100 (0.4 games/s, ~125s remaining, 5234 samples)
✅ Generated 10532 training samples in 250.3s (0.4 games/s)
   Buffer size: 10532 samples

🎓 Phase 2: Training
🎓 Training for 500 iterations...
Iter 0/500: Loss=165.3421 (Policy=164.1234, Value=1.2187) | Avg=165.3421, Best=165.3421
Iter 100/500: Loss=98.5432 (Policy=98.2341, Value=0.3091) | Avg=112.4567, Best=95.2341
...

💾 Saving checkpoint: data/alphazero_model_iter_5.pt
   ✅ Model saved

✅ Iteration 5 complete
```

## 🔍 监控训练质量

改进后的训练会显示：

- **Loss**: 当前批次的损失
- **Avg**: 最近100次迭代的平均损失
- **Best**: 迄今为止的最佳损失

如果看到：
- ✅ **Avg 持续下降** → 训练正常
- ⚠️ **Avg 波动很大** → 学习率可能过高
- ❌ **Avg 不再下降** → 需要更多新数据或降低学习率

## 💡 训练技巧

### 1. 渐进式训练

```bash
# 第一阶段：快速探索（小网络，少迭代）
cargo run --release --bin train_alphazero --features alphazero,random -- \
  ../data/stage1.pt 50 300 5

# 第二阶段：加载第一阶段模型，增加数据
# （需要修改代码支持加载现有模型继续训练）
```

### 2. 调整参数

修改 `backend/src/bin/train_alphazero.rs` 中的配置：

```rust
let config = AlphaZeroConfig {
    num_filters: 64,           // 增加网络容量
    num_res_blocks: 5,         // 增加深度
    learning_rate: 0.0005,     // 降低学习率
    batch_size: 64,            // 增大批次
    replay_buffer_size: 200000, // 更大的回放缓冲区
    num_mcts_simulations: 50,  // 更多MCTS模拟
    temperature: 1.0,
    // ...
};
```

### 3. 检查点管理

训练会自动保存检查点：

```
data/alphazero_model_iter_5.pt
data/alphazero_model_iter_10.pt
data/alphazero_model_iter_15.pt
...
```

你可以测试不同检查点的性能：

```bash
# 转换并测试第10轮的模型
python3 convert_model.py data/alphazero_model_iter_10.pt
./play_match.sh data/alphazero_model_iter_10_converted.pt 10 500
```

## 🎮 测试训练效果

训练完成后：

1. **转换模型**
```bash
python3 convert_model.py data/az_model.pt
```

2. **对战测试**
```bash
./play_match.sh data/az_model_converted.pt 20 500
```

3. **查看胜率**
- 如果 AlphaZero 胜率 > 60% → 训练有效
- 如果 AlphaZero 胜率 < 40% → 需要更多训练
- 如果接近 50% → 需要增加网络容量或训练时间

## 🐛 常见问题

### Q: Loss 仍然不下降？

A: 尝试：
1. 增加每轮游戏数（100 → 200）
2. 降低学习率（修改代码中的 `learning_rate`）
3. 增加网络容量（`num_filters`, `num_res_blocks`）

### Q: 训练太慢？

A: 优化方案：
1. 减少 MCTS 模拟次数（25 → 15）
2. 减少每轮游戏数（100 → 50）
3. 减少训练迭代（500 → 300）

### Q: 内存不足？

A: 调整：
1. 减小回放缓冲区（100000 → 50000）
2. 减小批次大小（32 → 16）
3. 减小网络规模

## 📈 预期训练时间

| 配置 | 每轮时间 | 10轮总时间 | 预期效果 |
|------|---------|-----------|---------|
| 快速 (50游戏/300迭代) | ~5分钟 | ~50分钟 | 能下棋但较弱 |
| 标准 (100游戏/500迭代) | ~10分钟 | ~100分钟 | 中等水平 |
| 完整 (200游戏/1000迭代) | ~20分钟 | ~200分钟 | 较强水平 |

## 🎯 下一步优化

1. **并行自我对弈**：需要解决权重共享问题
2. **学习率衰减**：训练后期自动降低学习率
3. **早停机制**：loss 不再下降时自动停止
4. **模型评估**：每轮自动评估模型强度
