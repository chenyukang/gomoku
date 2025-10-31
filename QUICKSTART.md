# 五子棋机器学习快速开始

这是一个帮助你学习机器学习方法下棋策略的指南。

## 📋 前置要求

### Rust 环境
```bash
# 检查 Rust 是否已安装
rustc --version

# 如果没有，安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Python 环境 (用于数据分析)
```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r ml_examples/requirements.txt
```

## 🚀 第一步：理解现有 AI

### 1. 运行现有的 AI
```bash
cd backend

# 构建项目
cargo build --release

# 测试 Minimax 算法
cargo run --release -- -i "............................................................................................................................................................................................................................................................." -d 5

# 启动 Web 服务器
cargo run --release --features server -- -s
# 然后在浏览器打开 client/index.html
```

### 2. 理解两种算法
- **Minimax (minimax.rs)**: 传统博弈树搜索 + Alpha-Beta 剪枝
- **MCTS (monte.rs)**: 蒙特卡洛树搜索

## 📊 第二步：生成训练数据

### 1. 运行自我对弈
```bash
cd backend

# 生成 10 局对局数据
cargo run --release --bin ml_trainer -- --selfplay 10

# 详细模式 (查看棋盘)
cargo run --release --bin ml_trainer -- --selfplay 1 -v

# 生成更多数据 (推荐至少 100 局)
cargo run --release --bin ml_trainer -- --selfplay 100
```

### 2. 锦标赛模式
```bash
# 让所有算法互相对战
cargo run --release --bin ml_trainer -- --tournament 5
```

### 3. 查看生成的数据
```bash
# 数据保存在 data/ 目录
ls -lh data/

# 查看 CSV 文件
head data/games.csv

# 查看 JSON 文件
cat data/games.json | head -n 20
```

## 🤖 第三步：数据分析与机器学习

### 1. 基础数据分析
```bash
# 进入项目根目录
cd ..

# 激活 Python 环境
source venv/bin/activate

# 运行分析脚本
python ml_examples/analyze_data.py
```

这会生成：
- 📊 数据可视化图表 (`data/analysis_basic.png`)
- 📈 特征重要性分析 (`data/feature_importance.png`)
- 🎯 简单预测模型

### 2. Q-Learning 演示
```bash
# 运行 Q-Learning 示例
python ml_examples/q_learning_demo.py
```

## 📚 第四步：学习进阶内容

查看完整学习指南：
```bash
cat ML_LEARNING_GUIDE.md
```

## 🎯 项目结构

```
gomoku/
├── backend/
│   ├── src/
│   │   ├── board.rs         # 棋盘逻辑
│   │   ├── minimax.rs       # Minimax 算法
│   │   ├── monte.rs         # MCTS 算法
│   │   ├── game_record.rs   # 游戏记录 (新)
│   │   ├── self_play.rs     # 自我对弈 (新)
│   │   └── bin/
│   │       └── ml_trainer.rs # 训练工具 (新)
│   └── Cargo.toml
├── ml_examples/             # Python ML 示例 (新)
│   ├── analyze_data.py      # 数据分析
│   ├── q_learning_demo.py   # Q-Learning 演示
│   └── requirements.txt
├── data/                    # 训练数据目录 (自动生成)
│   ├── games.csv
│   └── games.json
├── client/                  # Web 前端
└── ML_LEARNING_GUIDE.md    # 完整学习指南 (新)
```

## 💡 常见使用场景

### 场景1: 收集大量数据
```bash
# 后台运行，生成 1000 局数据
cd backend
nohup cargo run --release --bin ml_trainer -- --selfplay 1000 > training.log 2>&1 &

# 查看进度
tail -f training.log
```

### 场景2: 对比不同算法
```bash
# Minimax vs Minimax
cargo run --release --bin ml_trainer -- --selfplay 20 --algo1 minimax --algo2 minimax

# Monte Carlo vs Monte Carlo
cargo run --release --bin ml_trainer -- --selfplay 20 --algo1 monte_carlo --algo2 monte_carlo

# 混合对战
cargo run --release --bin ml_trainer -- --selfplay 50 --algo1 minimax --algo2 monte_carlo
```

### 场景3: 分析特定算法的表现
```python
# 在 Python 中
import pandas as pd

df = pd.read_csv('data/games.csv')

# 分析 minimax 的胜率
minimax_as_p1 = df[df['player'] == 1]
minimax_wins = len(minimax_as_p1[minimax_as_p1['final_reward'] > 0])
total = len(minimax_as_p1)
print(f"Minimax 先手胜率: {minimax_wins/total:.2%}")
```

## 🔧 自定义开发

### 添加新的 AI 算法
1. 在 `backend/src/` 创建新文件 (如 `my_algo.rs`)
2. 实现 `GomokuSolver` trait
3. 在 `algo.rs` 中注册
4. 在 `ml_trainer` 中添加到算法列表

### 修改评估函数
编辑 `backend/src/board.rs` 中的 `score()` 函数

### 添加新的特征
在 `ml_examples/analyze_data.py` 中的 `extract_features()` 添加新特征

## 🐛 问题排查

### 编译错误
```bash
# 清理并重新构建
cargo clean
cargo build --release
```

### Python 导入错误
```bash
# 确保在虚拟环境中
which python
# 应该显示 venv/bin/python

# 重新安装依赖
pip install -r ml_examples/requirements.txt
```

### 数据文件找不到
```bash
# 确保从正确的目录运行
pwd  # 应该在项目根目录

# 确保 data 目录存在
mkdir -p data
```

## 📖 下一步学习

1. ✅ **完成快速开始** - 生成第一批数据
2. 📊 **数据分析** - 理解数据特征
3. 🤖 **传统 ML** - 使用 scikit-learn 改进评估
4. 🧠 **强化学习** - 实现简单的 Q-Learning
5. 🚀 **深度学习** - 尝试神经网络方法

详细的学习路径请查看 `ML_LEARNING_GUIDE.md`

## 💬 获取帮助

- 查看代码注释
- 阅读 `ML_LEARNING_GUIDE.md`
- 参考 `ml_examples/` 中的示例代码

祝学习愉快！🎉
