# AlphaZero Connect4 模型评估指南

## 评估方法

我们已经实现了完整的模型评估系统，包括以下几种评估方式：

### 1. 对抗不同难度的对手

#### 1.1 对抗随机玩家
```bash
cargo run --features alphazero --bin test_eval_system
```
- **用途**：最基础的评估，检查模型是否比随机强
- **预期**：训练良好的模型应该有 **90%+** 胜率

#### 1.2 对抗纯MCTS（无神经网络）
```bash
cargo run --features alphazero --bin test_eval_system
```
- **用途**：中级评估，检查神经网络是否有效提升MCTS
- **预期**：训练良好的模型应该有 **60%+** 胜率
- **说明**：纯MCTS已经很强（对随机玩家100%胜率）

#### 1.3 不同搜索深度的AlphaZero对比
- AlphaZero(100模拟) vs AlphaZero(50模拟)
- **用途**：测试搜索深度的影响
- **预期**：更多模拟次数应该更强

### 2. 训练过程中持续评估

#### 运行带评估的训练
```bash
cd backend
cargo run --features alphazero --bin train_with_eval
```

这会在每轮训练后自动评估模型，输出包括：
- 初始模型（未训练）评分
- 每轮迭代后的评分
- vs 随机玩家的胜率
- vs 纯MCTS的胜率

**示例输出：**
```
初始评估：
  vs 随机: 30%
  vs MCTS: 0%

迭代1后：
  vs 随机: 10%
  vs MCTS: 0%

迭代2后：
  vs 随机: 40%
  vs MCTS: 10%

迭代5后：
  vs 随机: 80%
  vs MCTS: 50%
```

### 3. 对称评估（消除先手优势）

在 `az_eval.rs` 中实现了 `evaluate_symmetric` 函数：
- 双方各执先后手
- 更公平的评估
- 适合精确测量棋力差距

### 4. 观看对局过程

设置 `verbose=true` 可以看到完整的棋盘变化：
```rust
gomoku::az_eval::play_game(&player1, &player2, true);
```

## 评估指标

### 基本指标
- **胜率**：最直接的强度指标
- **平均步数**：反映对局质量（强手对决通常步数较少）
- **平局率**：Connect4平局较少，高平局率可能说明双方实力接近

### 进阶指标（可扩展）
- **ELO评级**：量化棋力差距
- **策略质量**：与最佳走法的匹配度
- **价值准确性**：价值网络预测vs实际结果

## 当前测试结果

### 未训练模型（随机初始化）
- vs 随机玩家：30% 胜率
- vs 纯MCTS：0% 胜率
- **结论**：纯MCTS非常强，神经网络必须训练才有用

### 训练中模型（1轮后）
- vs 随机玩家：10% 胜率
- vs 纯MCTS：0% 胜率
- **结论**：早期训练可能暂时变弱（探索阶段）

### 预期完全训练后（10+轮）
- vs 随机玩家：>90% 胜率
- vs 纯MCTS：>60% 胜率
- **结论**：AlphaZero应该超越纯MCTS

## 使用建议

### 快速测试
```bash
# 测试评估系统是否工作
cargo run --features alphazero --bin test_eval_system
```

### 完整训练+评估
```bash
# 训练并实时评估（推荐）
cargo run --features alphazero --bin train_with_eval

# 或者分开：先训练
cargo run --features alphazero --bin test_connect4

# 然后评估（注意：当前模型加载有兼容性问题）
cargo run --features alphazero --bin eval_connect4
```

### 自定义评估
编辑 `src/bin/train_with_eval.rs` 可以调整：
- 评估频率
- 对手配置
- 对局数量

## 已知问题

### 模型加载失败
**问题**：`tch-rs` 的 `VarStore::load()` 有版本兼容性问题
```
Internal torch error: isGenericDict() INTERNAL ASSERT FAILED
```

**临时方案**：
1. 使用 `train_with_eval` 在训练过程中直接评估
2. 不保存/加载模型，每次训练新模型并评估

**未来改进**：
- 使用 `save_to_stream/load_from_stream`
- 或者自己实现参数序列化

## 评估代码结构

```
backend/src/
├── az_eval.rs          # 评估核心逻辑
│   ├── Player enum     # 不同类型玩家
│   ├── play_game()     # 单局对战
│   ├── evaluate()      # 多局评估
│   └── evaluate_symmetric()  # 对称评估
│
└── bin/
    ├── eval_connect4.rs      # 完整评估程序（加载模型）
    ├── test_eval_system.rs   # 测试评估系统（不加载模型）
    ├── train_with_eval.rs    # 训练+评估集成
    └── quick_eval.rs         # 快速评估示例
```

## 下一步改进

1. **修复模型加载**：解决 tch-rs 兼容性问题
2. **ELO系统**：实现评级系统量化棋力
3. **对局数据库**：保存精彩对局用于分析
4. **可视化**：绘制训练曲线和胜率变化
5. **并行评估**：加速评估过程
6. **自对弈检查点**：定期评估不同checkpoint的相对强度
