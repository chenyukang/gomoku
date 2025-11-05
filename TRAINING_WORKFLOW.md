# AlphaZero Connect4 持续训练工作流程

## 快速开始

### 1. 从零开始训练（推荐新手）

```bash
# 训练50代，每代100局游戏
./train_iterative.sh 0 50
```

这将：
- 自动创建第0代（随机初始化模型）
- 持续训练到第50代
- 每代生成100局自我对弈游戏数据
- 每代进行300次训练迭代
- 自动保存每一代模型到 `data/generations/`
- 保存最佳模型到 `data/best_model.pt`

### 2. 继续已有训练

```bash
# 从第20代继续训练到第100代
./train_iterative.sh 20 100
```

### 3. 测试模型

```bash
# 测试最佳模型（10局，每局500次MCTS模拟）
./play_match.sh data/best_model.pt 10 500
```

## 训练策略建议

### 阶段1：快速探索（第0-20代）
- 目标：快速建立基础棋力
- 配置：
  - 每代游戏数：100局
  - 训练迭代：300次
  - MCTS模拟：200次
- 预期时间：每代约5-10分钟（取决于硬件）
- 预期效果：模型学会基本规则和简单战术

```bash
./train_iterative.sh 0 20
```

### 阶段2：深度优化（第20-50代）
- 目标：强化战术理解和策略深度
- 配置：
  - 每代游戏数：150局
  - 训练迭代：500次
  - MCTS模拟：400次（可以修改脚本增加）
- 预期时间：每代约15-30分钟
- 预期效果：掌握复杂战术和开局策略

编辑 `train_iterative.sh` 修改配置：
```bash
GAMES_PER_GEN=150
TRAIN_ITERS=500
```

### 阶段3：精英训练（第50+代）
- 目标：达到竞技级别
- 配置：
  - 每代游戏数：200局
  - 训练迭代：800次
  - MCTS模拟：800次
- 预期时间：每代约1-2小时
- 预期效果：接近或超越人类高级玩家

## 训练监控

### 查看训练历史

```bash
# 查看所有代数的训练损失和耗时
cat data/training_history.csv
```

格式：`代数,最佳验证损失,训练时长(秒)`

### 查看单代详细日志

```bash
# 查看第10代的详细日志
cat data/logs/gen_0010.log
```

### 可视化训练曲线（可选）

创建简单的Python脚本查看训练进度：

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/training_history.csv',
                 names=['generation', 'val_loss', 'duration_sec'])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df['generation'], df['val_loss'])
plt.xlabel('Generation')
plt.ylabel('Validation Loss')
plt.title('Training Progress')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df['generation'], df['duration_sec'] / 60)
plt.xlabel('Generation')
plt.ylabel('Duration (minutes)')
plt.title('Training Time per Generation')
plt.grid(True)

plt.tight_layout()
plt.savefig('data/training_progress.png')
plt.show()
```

## 模型管理

### 目录结构

```
data/
├── best_model.pt              # 当前最佳模型（始终指向最新代）
├── generations/               # 所有代数模型
│   ├── gen_0000.pt
│   ├── gen_0001.pt
│   ├── gen_0010.pt           # 每10代保留
│   ├── gen_0020.pt
│   └── ...
├── checkpoints/               # 定期检查点（每5代）
│   ├── checkpoint_gen_0005.pt
│   ├── checkpoint_gen_0010.pt
│   └── ...
└── logs/                      # 训练日志
    ├── gen_0000.log
    ├── gen_0001.log
    └── ...
```

### 清理策略

脚本会自动：
- 保留每10代的模型（gen_0010.pt, gen_0020.pt 等）
- 保留每5代的检查点
- 删除中间代数的模型以节省空间

手动清理：
```bash
# 只保留检查点和最佳模型
rm -rf data/generations/gen_*.pt
rm -rf data/logs/

# 重新开始（删除所有训练数据）
rm -rf data/generations data/checkpoints data/logs data/training_history.csv
```

## 高级配置

### 修改训练参数

编辑 `train_iterative.sh` 中的配置：

```bash
GAMES_PER_GEN=100           # 每代自我对弈游戏数
TRAIN_ITERS=300             # 每代训练迭代数
CHECKPOINT_INTERVAL=5       # 检查点保存间隔
```

### 修改网络架构和MCTS参数

编辑 `backend/src/bin/train_alphazero.rs`：

```rust
// 默认配置
num_filters: 128,        // 卷积通道数（更大=更强但更慢）
num_blocks: 6,           // 残差块数量（更多=更深但更慢）
num_simulations: 200,    // MCTS模拟次数（更多=更强但更慢）
batch_size: 64,          // 批次大小
buffer_size: 10000,      // 经验回放缓冲区大小
```

修改后需要重新编译：
```bash
cd backend
cargo build --release --features alphazero --bin train_alphazero
cd ..
```

## 评估和对比

### 对比不同代数

```bash
# 比较第10代和第30代（需要实现play_arena工具）
# ./play_arena.sh data/generations/gen_0010.pt data/generations/gen_0030.pt 20
```

### 对比模型强度

测试不同MCTS模拟次数的效果：

```bash
# 弱模式（100次模拟）
./play_match.sh data/best_model.pt 5 100

# 中等模式（500次模拟）
./play_match.sh data/best_model.pt 5 500

# 强模式（2000次模拟）
./play_match.sh data/best_model.pt 5 2000
```

## 训练建议

### ✅ 推荐做法

1. **从小规模开始**：先训练20代看效果，再决定是否继续
2. **定期测试**：每10-20代手动测试一次棋力
3. **监控损失**：验证损失应该逐渐下降（虽然可能有波动）
4. **保存检查点**：重要代数手动备份
5. **利用夜间**：训练可以在后台运行，适合晚上挂机

### ❌ 避免做法

1. **过早放弃**：前几代棋力可能很弱，需要耐心等待
2. **频繁中断**：训练中断会丢失当前代的进度
3. **过小数据**：每代少于50局游戏效果不佳
4. **忽略日志**：出现NaN或异常损失要及时检查

## 预期训练时间（M1 Mac参考）

| 代数 | 配置 | 单代耗时 | 累计耗时 |
|------|------|----------|----------|
| 0-20 | 100局/300iter | 5-8分钟 | 2-3小时 |
| 20-50 | 150局/500iter | 15-25分钟 | 10-15小时 |
| 50-100 | 200局/800iter | 1-2小时 | 50-100小时 |

**建议训练策略**：
- 第1天：训练第0-10代（约1小时）→ 测试基础棋力
- 第2-3天：训练第10-30代（约8小时）→ 测试战术理解
- 长期：持续训练到50+代（可以分批进行）

## 故障排除

### 训练中断怎么办？

```bash
# 查看最后完成的代数
ls -lt data/generations/ | head -5

# 假设最后完成到第15代，从第16代继续
./train_iterative.sh 16 50
```

### 损失不下降/模型不进步？

可能原因：
1. **学习率过大/过小**：检查 `alphazero_net.rs` 中的学习率设置
2. **数据不足**：增加每代游戏数（GAMES_PER_GEN）
3. **MCTS太弱**：增加模拟次数（num_simulations）
4. **过拟合**：早停机制会自动处理，但可能需要更多样化的数据

### 内存不足？

减小配置：
```bash
# 在 train_alphazero.rs 中
buffer_size: 5000,      # 从10000减小到5000
batch_size: 32,         # 从64减小到32
```

## 下一步优化

当前训练流程已经包含：
- ✅ 数据增强（水平翻转）
- ✅ 温度采样策略
- ✅ 训练/验证分离
- ✅ 早停机制

未来可以添加：
- [ ] 代际对战评估（只有胜率>55%才接受新模型）
- [ ] Ply penalty（鼓励快速获胜）
- [ ] 开局库（常见开局的先验知识）
- [ ] Solver集成（残局精确求解）
- [ ] 分布式训练（多机并行自我对弈）

## 相关文件

- `train_iterative.sh` - 主训练脚本
- `backend/src/bin/train_alphazero.rs` - 训练程序入口
- `backend/src/alphazero_trainer.rs` - 训练逻辑
- `backend/src/alphazero_mcts.rs` - MCTS搜索
- `backend/src/alphazero_net.rs` - 神经网络
- `IMPROVEMENTS.md` - 已实现的改进说明
