# 🚀 开始训练 AlphaZero Connect4

## ✅ 更新说明（已修复）

**模型加载问题已解决！**

通过在每代训练后使用 `convert_model.py` 转换模型格式，现在 `train_iterative.sh` 可以正常工作了！

**关键发现**：
- ✅ 转换后的模型格式可以被 `tch-rs` 成功加载
- ✅ 每代训练后自动转换，下一代可以继续训练
- ✅ 支持真正的代际迭代和断点续传

**推荐使用**：`train_iterative.sh`（已修复）

## 快速开始指南

### 步骤 1: 选择训练模式

#### 模式 A: 快速测试（推荐首次使用）✅
训练5代，快速验证流程：
```bash
./train_iterative.sh 0 5
```
预计耗时：约30-60分钟
✨ **已修复**：每代会自动加载上一代模型并继续训练

#### 模式 B: 标准训练（推荐）✅
训练50代，获得有一定实力的模型：
```bash
./train_iterative.sh 0 50
```
预计耗时：约5-10小时（可以后台运行或夜间挂机）
✨ **已修复**：真正的迭代训练，每代基于上一代进步

#### 模式 C: 深度训练（高级）✅
训练100+代，追求强力模型：
```bash
./train_iterative.sh 0 100
```
预计耗时：约20-40小时
✨ **已修复**：支持断点续传

#### 模式 D: 一次性训练（备选方案）
如果不想用迭代训练，可以使用简单模式：
```bash
./train_simple.sh 50 100 300
```
在单次运行中完成所有epochs

### 步骤 2: 后台运行（推荐）

如果训练时间较长，建议使用 `nohup` 或 `screen` 在后台运行：

```bash
# 使用 nohup（输出会保存到 nohup.out）
nohup ./train_iterative.sh 0 50 &

# 查看训练输出
tail -f nohup.out

# 查看训练进度（查看最新的代数）
ls -lt data/generations/ | head -10
```

或者使用 `screen`（更灵活）：
```bash
# 创建新会话
screen -S alphazero

# 在 screen 中运行训练
./train_iterative.sh 0 50

# 按 Ctrl+A 然后按 D 断开（训练继续在后台）

# 重新连接会话
screen -r alphazero
```

### 步骤 3: 监控训练进度

#### 实时查看训练日志
```bash
# 查看最新代数的日志
ls -t data/logs/*.log | head -1 | xargs tail -f
```

#### 查看训练历史
```bash
# 查看所有代数的损失和耗时
cat data/training_history.csv

# 显示为表格（如果安装了 column）
cat data/training_history.csv | column -t -s ','
```

#### 检查当前进度
```bash
# 查看已完成的代数
ls data/generations/ | sort | tail -5

# 查看最新模型
ls -lh data/best_model.pt
```

### 步骤 4: 测试模型

在训练过程中，随时可以测试当前最佳模型：

```bash
# 快速测试（5局，每局500次模拟）
./play_match.sh data/best_model.pt 5 500

# 完整测试（10局，每局1000次模拟）
./play_match.sh data/best_model.pt 10 1000
```

测试特定代数的模型：
```bash
# 测试第20代
./play_match.sh data/generations/gen_0020.pt 5 500

# 测试第40代
./play_match.sh data/generations/gen_0040.pt 5 500
```

## 训练参数调整

### 基本参数（编辑 train_iterative.sh）

```bash
GAMES_PER_GEN=100           # 每代游戏数（越多越好，但更慢）
TRAIN_ITERS=300             # 每代训练迭代（越多越好，但更慢）
CHECKPOINT_INTERVAL=5       # 每N代保存检查点
```

**推荐配置：**

| 目标 | GAMES_PER_GEN | TRAIN_ITERS | 单代耗时 |
|------|---------------|-------------|----------|
| 快速测试 | 50 | 150 | 2-4分钟 |
| 标准训练 | 100 | 300 | 5-10分钟 |
| 深度训练 | 200 | 500 | 15-30分钟 |
| 极致训练 | 300 | 800 | 30-60分钟 |

### 高级参数（编辑 backend/src/bin/train_alphazero.rs）

```rust
// 网络架构
num_filters: 128,        // 卷积通道数（64/128/256）
num_blocks: 6,           // 残差块数（4/6/8/10）

// MCTS
num_simulations: 200,    // 每步模拟次数（100/200/400/800）

// 训练
batch_size: 64,          // 批次大小（32/64/128）
buffer_size: 10000,      // 经验回放缓冲区（5000/10000/20000）
```

修改后需要重新编译：
```bash
cd backend
cargo build --release --features alphazero --bin train_alphazero
cd ..
```

## 常见问题

### Q: 训练太慢怎么办？

A: 可以尝试：
1. 减少 `GAMES_PER_GEN`（如改为50）
2. 减少 `TRAIN_ITERS`（如改为150）
3. 减少 `num_simulations`（如改为100）
4. 减少网络大小（num_filters: 64, num_blocks: 4）

### Q: 如何知道模型在进步？

A: 观察这些指标：
1. **验证损失下降**：`cat data/training_history.csv` 看第2列
2. **手动测试棋力**：定期用 `play_match.sh` 测试
3. **观察自我对弈**：后期训练的游戏应该更长、更复杂

### Q: 训练中断了怎么恢复？

A: 脚本设计了断点续传功能：
```bash
# 查看最后完成的代数
ls -lt data/generations/ | head -3

# 假设最后是 gen_0015.pt，从第16代继续
./train_iterative.sh 16 50
```

### Q: 我想从某个检查点重新开始？

A:
```bash
# 备份当前进度
cp -r data/generations data/generations_backup

# 从第20代重新开始
./train_iterative.sh 20 50
```

### Q: 内存不够怎么办？

A: 减小缓冲区和批次大小（编辑 `train_alphazero.rs`）：
```rust
buffer_size: 5000,      // 从10000减到5000
batch_size: 32,         // 从64减到32
```

## 预期效果

### 训练里程碑

| 代数 | 预期表现 | 验证方式 |
|------|----------|----------|
| 0-5 | 学会基本规则，不犯明显错误 | 不会随机落子 |
| 5-10 | 能识别简单的威胁和机会 | 能堵住对手的四连 |
| 10-20 | 掌握基本战术（双三、冲四） | 能创造简单的进攻 |
| 20-40 | 有一定的策略理解 | 开局不太差，中盘有想法 |
| 40-60 | 接近或达到中级人类水平 | 能和普通玩家竞争 |
| 60+ | 高级水平，难以击败 | 可能超越大多数人类 |

### 如何评估进步

定期对比不同代数：
```bash
# 第10代 vs 第20代
# （需要实现对战工具，或手动测试）
./play_match.sh data/generations/gen_0010.pt 3 500
./play_match.sh data/generations/gen_0020.pt 3 500

# 观察：第20代应该下得更"聪明"，犯错更少
```

## 推荐训练流程

### Day 1: 初始探索（1-2小时）
```bash
# 训练前10代，快速验证
./train_iterative.sh 0 10

# 测试第10代
./play_match.sh data/generations/gen_0010.pt 5 500
```

### Day 2-3: 持续训练（8-12小时）
```bash
# 后台运行到第50代
nohup ./train_iterative.sh 10 50 &

# 定期检查进度
tail -f nohup.out
cat data/training_history.csv
```

### Day 4+: 深度优化（可选）
```bash
# 继续训练到100代
nohup ./train_iterative.sh 50 100 &

# 或者增加难度（修改参数后重新开始）
# 编辑 train_iterative.sh: GAMES_PER_GEN=200, TRAIN_ITERS=500
./train_iterative.sh 0 50
```

## 下一步

训练完成后，你可以：

1. **部署模型**：将最佳模型集成到 Web UI
2. **对战人类**：通过浏览器界面与模型对弈
3. **继续改进**：实现代际评估、Ply penalty 等优化
4. **迁移到其他游戏**：尝试 Gomoku（五子棋）或其他棋类

## 相关文档

- `TRAINING_WORKFLOW.md` - 详细的训练工作流程说明
- `IMPROVEMENTS.md` - 已实现的改进列表
- `ALPHAZERO_QUICKSTART.md` - AlphaZero 快速入门
- `train_iterative.sh` - 主训练脚本

---

**现在就开始吧！**

```bash
# 推荐：先快速测试5代（约30分钟）
./train_iterative.sh 0 5

# 或者：直接开始标准训练（可以后台运行）
nohup ./train_iterative.sh 0 50 > training.log 2>&1 &
```

祝训练顺利！🎉
