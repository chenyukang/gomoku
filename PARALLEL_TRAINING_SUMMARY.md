# 并行训练实现总结

## ✅ 已完成

### 核心模块

1. **parallel_training.rs** - 数据管理
   - `TrainingDataPool`: 收集和管理来自多个工作进程的训练数据
   - `WorkerDataWriter`: 工作进程数据写入器，批量保存样本到JSON

2. **parallel_worker (binary)** - 工作进程
   - 独立进程生成自对弈游戏数据
   - 加载共享模型文件
   - 每10局保存一次数据到JSON

3. **parallel_trainer (binary)** - 主训练器
   - 协调多个工作进程
   - 收集训练数据
   - 训练神经网络
   - 保存更新的模型

4. **alphazero_net.rs** - 新增方法
   - `from_file()`: 从文件加载训练器
   - `train_on_samples()`: 在给定样本上训练
   - `save_model()`: 保存模型到文件

### 脚本工具

1. **parallel_train.sh**
   - 启动并行训练的主脚本
   - 参数：模型路径、数据目录、工作进程数、每进程游戏数、轮数

2. **test_parallel.sh**
   - 快速测试脚本（2进程 × 5局 × 1轮）
   - 用于验证系统正常工作

3. **monitor_training.sh**
   - 监控训练进度
   - 显示工作进程状态、数据文件数量、磁盘使用

### 文档

1. **PARALLEL_TRAINING.md**
   - 完整的使用指南
   - 架构说明
   - 参数配置
   - 性能优化建议
   - 常见问题解答

## 架构设计

```
主训练器 (parallel_trainer)
    ↓
启动 N 个工作进程 (parallel_worker)
    ↓
每个工作进程:
  1. 加载模型 (model.pt)
  2. 生成自对弈数据
  3. 保存到 training_data/worker_N/batch_*.json
    ↓
主训练器:
  1. 等待所有工作进程完成
  2. 收集所有 JSON 数据
  3. 训练神经网络
  4. 保存更新的模型
    ↓
重复下一轮
```

## 使用方法

### 快速测试（推荐先运行）

```bash
./scripts/test_parallel.sh
```

### 正式训练

```bash
# 基本用法（4进程 × 25局 × 10轮）
./scripts/parallel_train.sh

# 自定义参数（8进程 × 50局 × 20轮）
./scripts/parallel_train.sh model.pt data 8 50 20
```

### 监控进度

```bash
# 在另一个终端运行
./scripts/monitor_training.sh
```

## 性能预期

| 配置 | 速度 | 适用场景 |
|------|------|----------|
| 串行训练 | 0.4 游戏/秒 | 基准测试 |
| 2进程并行 | ~0.8 游戏/秒 | 快速测试 |
| 4进程并行 | ~1.6 游戏/秒 | 日常训练 |
| 8进程并行 | ~3.2 游戏/秒 | 密集训练 |

*实际速度取决于CPU性能、MCTS模拟次数等因素*

## 关键文件

```
backend/src/
├── parallel_training.rs          # 数据管理模块
├── alphazero_net.rs              # 添加了 train_on_samples 等方法
└── bin/
    ├── parallel_worker.rs        # 工作进程二进制
    └── parallel_trainer.rs       # 主训练器二进制

scripts/
├── parallel_train.sh             # 训练启动脚本
├── test_parallel.sh              # 快速测试脚本
└── monitor_training.sh           # 监控脚本

PARALLEL_TRAINING.md              # 详细文档
```

## 编译状态

✅ 所有二进制文件编译成功
✅ 无错误，仅有少量无害的警告

## 下一步

1. 运行快速测试验证系统:
   ```bash
   ./scripts/test_parallel.sh
   ```

2. 根据CPU核心数调整并行度:
   ```bash
   # 查看核心数
   sysctl -n hw.ncpu

   # 使用核心数-1作为工作进程数
   ./scripts/parallel_train.sh model.pt data 8 25 10
   ```

3. 长时间训练:
   ```bash
   # 大规模训练（8进程 × 100局 × 50轮）
   ./scripts/parallel_train.sh model.pt data 8 100 50
   ```

## 技术亮点

- ✅ **真正的并行**: 多进程而非多线程，充分利用多核CPU
- ✅ **文件共享**: 简单可靠的进程间通信
- ✅ **容错性**: 单个进程失败不影响其他进程
- ✅ **可扩展**: 轻松增加更多工作进程
- ✅ **监控友好**: 实时查看训练进度

## 已知限制

1. 工作进程之间是独立的，可能在同一轮中使用稍有不同的模型版本
2. 文件I/O可能成为瓶颈（建议使用SSD）
3. 内存使用随工作进程数线性增长

## 参考

- 详细使用说明: `PARALLEL_TRAINING.md`
- 串行训练指南: `TRAINING_GUIDE.md`
- AlphaZero论文: https://arxiv.org/abs/1712.01815
