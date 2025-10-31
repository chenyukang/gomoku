# AlphaZero 实现指南

## 概述

本项目实现了 AlphaZero 算法用于五子棋（Gomoku）游戏。AlphaZero 结合了深度神经网络和蒙特卡洛树搜索（MCTS），通过自我对弈不断学习和提高棋力。

## 架构

### 核心组件

1. **神经网络** (`alphazero_net.rs`)
   - ResNet 架构（残差网络）
   - 双输出头：策略网络（Policy）+ 价值网络（Value）
   - 输入：3通道 15x15 棋盘（当前玩家、对手、玩家标记）
   - 输出：225个位置的概率分布 + 局面评估值

2. **MCTS 搜索** (`alphazero_mcts.rs`)
   - 神经网络引导的树搜索
   - UCB 公式平衡探索与利用
   - 温度参数控制探索程度

3. **训练管道** (`alphazero_trainer.rs`)
   - 自我对弈生成训练数据
   - 经验回放缓冲区
   - 策略损失 + 价值损失联合优化

4. **求解器** (`alphazero_solver.rs`)
   - 实现 `GomokuSolver` 接口
   - 支持标准和自适应模拟次数

## 安装依赖

### PyTorch (LibTorch)

AlphaZero 需要 PyTorch C++ 库（LibTorch）。

**macOS 安装：**

```bash
# 下载 LibTorch
cd ~
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.0.zip
unzip libtorch-macos-2.0.0.zip

# 设置环境变量
export LIBTORCH=~/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH
```

**Linux 安装：**

```bash
cd ~
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

export LIBTORCH=~/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

## 编译

```bash
cd backend

# 编译 AlphaZero（启用 alphazero feature）
cargo build --release --features alphazero --bin alphazero_cli
```

## 使用方法

### 1. 训练模型

```bash
# 基础训练（默认参数）
cargo run --release --features alphazero --bin alphazero_cli -- train

# 自定义参数训练
cargo run --release --features alphazero --bin alphazero_cli -- train \
  --filters 128 \           # 卷积滤波器数量
  --blocks 10 \             # 残差块数量
  --lr 0.001 \              # 学习率
  --batch-size 32 \         # 批次大小
  --buffer-size 10000 \     # 回放缓冲区大小
  --games 100 \             # 每轮自我对弈游戏数
  --iterations 1000 \       # 每轮训练迭代次数
  --simulations 400 \       # MCTS 模拟次数
  --epochs 10 \             # 训练轮数
  --output data/my_model.pt # 输出路径
```

### 2. 测试模型

```bash
# 测试训练好的模型
cargo run --release --features alphazero --bin alphazero_cli -- \
  test data/alphazero_final.pt
```

### 3. 对战基准测试

```bash
# 与其他算法对战
cargo run --release --features alphazero --bin alphazero_cli -- \
  benchmark data/alphazero_final.pt
```

输出示例：
```
🎮 AlphaZero vs Minimax (10 games)
Results:
  AlphaZero wins: 7
  Minimax wins: 2
  Draws: 1
  Win rate: 70.0%
```

## 训练流程

### 迭代训练循环

1. **自我对弈**：使用当前模型进行自我对弈，生成训练数据
2. **数据存储**：将游戏状态、策略、结果存入回放缓冲区
3. **网络训练**：从缓冲区采样批次数据，训练神经网络
4. **模型评估**：定期评估模型性能
5. **重复迭代**：继续下一轮训练

### 训练数据格式

每个训练样本包含：
- **Board**: 3x15x15 的棋盘状态张量
- **Policy**: 225 维的策略目标（MCTS 访问次数分布）
- **Value**: 游戏结果（1=胜，-1=负，0=平）

### 损失函数

```
Total Loss = Policy Loss + Value Loss
- Policy Loss: 交叉熵损失（策略网络）
- Value Loss: 均方误差（价值网络）
```

## 参数调优

### 网络架构

- **滤波器数量** (128-256)：更多滤波器 = 更强表达能力，但训练慢
- **残差块数量** (10-20)：更深网络 = 更强能力，但容易过拟合

### MCTS 参数

- **模拟次数** (400-1600)：
  - 训练时：400-800（速度优先）
  - 对战时：1600+（质量优先）
- **温度** (0.0-1.0)：
  - 前30步：1.0（探索）
  - 后续：0.0（利用）

### 训练参数

- **学习率** (0.0001-0.01)：
  - 初期：0.001（快速学习）
  - 后期：0.0001（精细调整）
- **批次大小** (32-128)：
  - 较大批次更稳定
  - 较小批次更新快
- **回放缓冲区** (10000-100000)：
  - 更大缓冲区 = 更多样化数据

## 快速测试

### 小规模训练（5分钟）

```bash
cargo run --release --features alphazero --bin alphazero_cli -- train \
  --filters 32 \
  --blocks 2 \
  --games 10 \
  --iterations 100 \
  --simulations 50 \
  --epochs 2 \
  --output data/test_model.pt
```

### 中等规模训练（1小时）

```bash
cargo run --release --features alphazero --bin alphazero_cli -- train \
  --filters 64 \
  --blocks 5 \
  --games 50 \
  --iterations 500 \
  --simulations 200 \
  --epochs 5
```

### 完整训练（8-24小时）

```bash
cargo run --release --features alphazero --bin alphazero_cli -- train \
  --filters 128 \
  --blocks 10 \
  --games 100 \
  --iterations 1000 \
  --simulations 400 \
  --epochs 20
```

## 性能优化

### GPU 加速

修改 `alphazero_net.rs` 中的设备：

```rust
// 改为
let device = Device::Cuda(0);  // 使用第一块 GPU
```

### 并行自我对弈

当前实现是串行的。可以通过 `rayon` 并行化：

```rust
use rayon::prelude::*;

// 在 alphazero_trainer.rs 中
(0..num_games).into_par_iter()
    .map(|_| self.self_play_game())
    .collect()
```

## 常见问题

### Q: 训练很慢怎么办？

A:
1. 减少模拟次数 (`--simulations`)
2. 减少自我对弈游戏数 (`--games`)
3. 使用 GPU 加速
4. 减小网络规模 (`--filters`, `--blocks`)

### Q: 模型不收敛？

A:
1. 降低学习率 (`--lr 0.0001`)
2. 增加批次大小 (`--batch-size 64`)
3. 增加训练迭代次数 (`--iterations 2000`)

### Q: 内存不足？

A:
1. 减小回放缓冲区 (`--buffer-size 5000`)
2. 减小批次大小 (`--batch-size 16`)
3. 减小网络规模

## 进阶功能

### 自定义网络架构

编辑 `alphazero_net.rs`，修改网络结构。

### 保存中间模型

训练会自动每10轮保存模型：
```
data/alphazero_model_iter_10.pt
data/alphazero_model_iter_20.pt
...
```

### 继续训练

```rust
// 在代码中加载已有模型继续训练
pipeline.load_model("data/alphazero_model_iter_10.pt")?;
pipeline.train_loop(10);  // 继续训练10轮
```

## 与 Python 版本对比

| 特性 | Rust 版本 | Python 版本 |
|------|-----------|-------------|
| 速度 | ⚡️ 非常快 | 较慢 |
| 内存 | 💾 高效 | 占用多 |
| 部署 | 📦 单文件 | 需要依赖 |
| 开发 | 🔧 编译复杂 | 简单 |
| 调试 | 🐛 较难 | 容易 |

## 参考资源

- [AlphaZero 论文](https://arxiv.org/abs/1712.01815)
- [tch-rs 文档](https://github.com/LaurentMazare/tch-rs)
- [MCTS 算法](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

## 下一步

1. ✅ 实现基础 AlphaZero
2. 🚧 GPU 加速
3. 🚧 并行自我对弈
4. 🚧 模型评估和对比
5. 🚧 迁移学习和预训练
