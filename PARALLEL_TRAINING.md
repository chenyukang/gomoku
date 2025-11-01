# AlphaZero 并行训练指南

## 概述

并行训练系统通过多进程方式加速 AlphaZero 训练：
- **多个工作进程**：独立生成自对弈数据
- **文件共享**：通过文件系统共享模型和数据
- **主训练器**：收集数据、训练网络、更新模型

## 架构

```
┌─────────────────────────────────────────┐
│         主训练器 (parallel_trainer)      │
│                                         │
│  1. 启动工作进程                         │
│  2. 等待数据生成                         │
│  3. 收集训练数据                         │
│  4. 训练神经网络                         │
│  5. 保存更新的模型                       │
└───┬─────────────────────────────────┬───┘
    │                                 │
    │    model.pt (共享模型文件)      │
    │                                 │
┌───▼─────────┐  ┌────────────┐  ┌───▼─────────┐
│  Worker 0   │  │  Worker 1  │  │  Worker N   │
│             │  │            │  │             │
│ 1. 加载模型 │  │ 1. 加载模型│  │ 1. 加载模型 │
│ 2. 自对弈   │  │ 2. 自对弈  │  │ 2. 自对弈   │
│ 3. 保存数据 │  │ 3. 保存数据│  │ 3. 保存数据 │
└─────┬───────┘  └──────┬─────┘  └──────┬──────┘
      │                 │                │
      ├─────────────────┴────────────────┘
      │
      ▼
 training_data/
   ├── worker_0/
   │   ├── batch_000000.json
   │   └── batch_000001.json
   ├── worker_1/
   │   └── ...
   └── worker_N/
       └── ...
```

## 快速开始

### 1. 基本用法

```bash
# 使用默认参数（4个工作进程，每进程25局，共10轮）
./scripts/parallel_train.sh

# 自定义参数
./scripts/parallel_train.sh model.pt data 8 50 20
#                          ^^^^^^^^  ^^^^ ^  ^^ ^^
#                          模型路径  目录 |  |  |
#                                     工作进程数 |  |
#                                          每进程游戏数 |
#                                                  训练轮数
```

### 2. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MODEL_PATH | model_parallel.pt | 模型保存路径 |
| DATA_DIR | ./training_data | 数据目录 |
| NUM_WORKERS | 4 | 工作进程数量 |
| GAMES_PER_WORKER | 25 | 每个进程生成的游戏数 |
| NUM_ROUNDS | 10 | 训练轮数 |

### 3. 监控训练

```bash
# 查看训练进度
./scripts/monitor_training.sh

# 查看数据目录（可选参数）
./scripts/monitor_training.sh ./training_data
```

## 训练流程

### 单轮训练

每一轮包含以下步骤：

1. **启动工作进程** (30-60秒)
   - 主训练器启动 N 个 `parallel_worker` 进程
   - 每个工作进程独立运行

2. **并行生成数据** (取决于 GAMES_PER_WORKER)
   - 每个工作进程：
     - 加载当前模型
     - 生成 M 局自对弈游戏
     - 每10局保存一个数据文件

3. **数据收集** (1-5秒)
   - 主训练器从所有 `worker_N/` 目录读取 JSON 文件
   - 合并到训练数据池
   - 删除已读取的文件

4. **模型训练** (30-120秒)
   - 在收集的数据上训练 500 次迭代
   - 每50次迭代打印损失
   - 保存更新的模型

5. **准备下一轮**
   - 工作进程将加载更新后的模型
   - 继续生成数据

## 性能优化

### CPU 核心数

建议工作进程数 = CPU 核心数 - 1：

```bash
# macOS 查看核心数
sysctl -n hw.ncpu

# 10核机器建议用8-9个工作进程
./scripts/parallel_train.sh model.pt data 8 25 10
```

### 训练速度估算

- 串行训练：~0.4 游戏/秒
- 4工作进程：~1.6 游戏/秒 (4倍加速)
- 8工作进程：~3.2 游戏/秒 (8倍加速)

实际速度取决于：
- CPU 性能
- MCTS 模拟次数（当前100次）
- 磁盘 I/O 速度

### 内存使用

每个工作进程约需：
- 模型加载：200-500 MB
- MCTS 内存：50-100 MB
- 总计：~300-600 MB/进程

示例：8进程 × 500MB = 4GB

## 数据格式

### TrainingSample

每个训练样本包含：

```json
{
  "board": [/* 675个浮点数: 3×15×15 */],
  "policy": [/* 225个浮点数: 15×15 动作概率 */],
  "value": 1.0  // -1.0 (输), 0.0 (平), 1.0 (赢)
}
```

### 文件组织

```
training_data/
├── worker_0/
│   ├── batch_000000.json  (10局游戏的样本)
│   ├── batch_000001.json
│   └── ...
├── worker_1/
│   └── ...
└── worker_N/
    └── ...
```

## 常见问题

### 1. 工作进程失败

**症状**：某个工作进程提前退出

**解决**：
- 检查模型文件是否存在且可读
- 确认 PyTorch 正确安装
- 查看错误日志

### 2. 数据未生成

**症状**：`training_data/` 目录为空

**解决**：
```bash
# 检查工作进程是否运行
ps aux | grep parallel_worker

# 检查文件权限
ls -la training_data/
```

### 3. 内存不足

**症状**：系统变慢或进程被杀死

**解决**：
- 减少工作进程数
- 降低 MCTS 模拟次数（修改 parallel_worker.rs）

### 4. 训练速度慢

**症状**：低于预期的游戏/秒

**解决**：
- 检查 CPU 占用率（应接近100%）
- 减少磁盘I/O（使用SSD）
- 增加 `--release` 编译优化

## 高级配置

### 修改 MCTS 模拟次数

编辑 `backend/src/bin/parallel_worker.rs`：

```rust
let mut mcts = AlphaZeroMCTS::new(board.clone(), 100); // 改为50-400
```

### 修改批量写入频率

```rust
if (game_idx + 1) % 10 == 0 || game_idx == num_games - 1 {
    // 改为每5局或每20局
}
```

### 修改训练迭代次数

编辑 `backend/src/bin/parallel_trainer.rs`：

```rust
ParallelTrainer::new(
    model_path,
    data_dir,
    num_workers,
    games_per_worker,
    500, // 改为100-1000
)
```

## 与串行训练对比

| 特性 | 串行训练 | 并行训练 |
|------|---------|---------|
| 速度 | 0.4 游戏/秒 | N×0.4 游戏/秒 |
| 内存 | 500 MB | N×500 MB |
| CPU | 单核 | N核 |
| 实现复杂度 | 简单 | 中等 |
| 数据多样性 | 低（同一模型） | 高（可能加载不同版本） |

## 下一步

1. **运行第一轮训练**：
   ```bash
   ./scripts/parallel_train.sh
   ```

2. **监控进度**：
   ```bash
   # 另一个终端
   watch -n 5 ./scripts/monitor_training.sh
   ```

3. **测试模型**：
   ```bash
   cargo run --release --bin play_alphazero --features alphazero -- model_parallel.pt
   ```

4. **调优参数**：
   - 增加工作进程数（如果CPU允许）
   - 增加每轮游戏数
   - 增加训练轮数

## 参考资料

- [AlphaGo Zero 论文](https://www.nature.com/articles/nature24270)
- [AlphaZero 论文](https://arxiv.org/abs/1712.01815)
- [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) - 串行训练指南
