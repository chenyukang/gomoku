# 五子棋机器学习学习指南

## 📚 当前项目概况

你的项目使用 Rust 实现，包含：
- **Minimax 算法** (带 Alpha-Beta 剪枝) - 传统 AI 搜索算法
- **蒙特卡洛树搜索 (MCTS)** - 基于随机模拟的算法
- Web 界面 (HTML + WebAssembly)
- 评估函数基于手工设计的特征

## 🚀 机器学习进阶路径

### 第一阶段：数据收集与分析 (1-2周)

#### 1.1 收集对局数据
```bash
# 创建数据收集模块
- 记录每一步的棋盘状态
- 记录最终胜负
- 保存为 JSON/CSV 格式
```

**实现建议：**
- 添加自我对弈功能 (Self-play)
- 记录 MCTS vs Minimax 的对局
- 导出训练数据集

#### 1.2 特征工程
当前的评估函数已经包含一些特征：
- 连子数量 (`count`)
- 空格数 (`space_count`)
- 开放端数 (`open_count`)

**扩展特征：**
- 威胁点数量
- 防守价值
- 位置热力图
- 棋型模式（活三、眠三、活四等）

### 第二阶段：传统机器学习方法 (2-3周)

#### 2.1 监督学习改进评估函数
使用线性回归或决策树学习更好的评估参数

**工具选择：**
- Python: scikit-learn, pandas
- Rust: linfa, smartcore

**步骤：**
1. 从专家对局中学习位置价值
2. 训练一个评估函数
3. 替换手工设计的 `score()` 函数

#### 2.2 强化学习入门 - Q-Learning
实现一个简单的 Q-Learning 算法

**学习目标：**
- 理解状态-动作-奖励
- 实现 Q-table 或线性近似
- 自我对弈训练

### 第三阶段：深度学习方法 (4-8周)

#### 3.1 使用 AlphaGo Zero 思路

**核心组件：**
1. **神经网络** - 策略网络 + 价值网络
   - 输入：棋盘状态 (15x15)
   - 输出：每个位置的概率 + 局面评估值

2. **MCTS + 神经网络**
   - 用神经网络指导 MCTS 搜索
   - 自我对弈生成训练数据

3. **训练流程**
   - 自我对弈 → 生成数据 → 训练网络 → 更新模型 → 循环

**技术栈推荐：**
- PyTorch 或 TensorFlow (Python)
- tch-rs (Rust 的 PyTorch 绑定)
- burn (纯 Rust 深度学习框架)

#### 3.2 具体实现方案

**方案 A：Python + Rust 混合**
```
Python (PyTorch)
  ↓ 训练神经网络
  ↓
Rust (WASM)
  ↓ 推理 + MCTS
  ↓
Web 前端
```

**方案 B：纯 Rust**
```rust
// 使用 tch-rs 或 burn
- 训练：Rust
- 推理：Rust (可编译为 WASM)
- 部署：更简单
```

### 第四阶段：高级优化 (持续)

1. **分布式训练**
   - 多机自我对弈
   - 异步训练

2. **模型压缩**
   - 知识蒸馏
   - 量化加速

3. **在线学习**
   - 从人类对局中学习
   - 持续改进

## 🛠️ 快速开始项目

### 项目1：数据收集系统
**难度：⭐**
- 添加对局记录功能
- 保存为 JSON 格式
- 可视化分析工具

### 项目2：特征学习
**难度：⭐⭐**
- 使用 scikit-learn 学习评估权重
- 与当前评估函数对比
- A/B 测试

### 项目3：神经网络评估函数
**难度：⭐⭐⭐**
- 训练一个简单的 CNN
- 替换 `board.eval_all()` 函数
- 测试棋力提升

### 项目4：AlphaZero-like 系统
**难度：⭐⭐⭐⭐⭐**
- 完整的自我对弈系统
- 策略+价值神经网络
- MCTS 集成

## 📖 推荐学习资源

### 论文
1. **AlphaGo & AlphaGo Zero** - DeepMind
2. **Mastering the Game of Go without Human Knowledge** (Nature 2017)
3. **A Survey of Monte Carlo Tree Search Methods** (IEEE 2012)

### 在线课程
- Coursera: Reinforcement Learning Specialization
- Stanford CS234: Reinforcement Learning
- DeepMind x UCL: Deep Learning Lecture Series

### 开源项目参考
- KataGo (围棋)
- Leela Zero (围棋/国际象棋)
- AlphaZero 复现项目

### 书籍
- 《强化学习》- Sutton & Barto
- 《深度学习》- Goodfellow et al.
- 《Hands-On Machine Learning》- Aurélien Géron

## 💡 具体实践建议

### 从简单开始
1. ✅ 先理解现有的 Minimax 和 MCTS 代码
2. ✅ 实现自我对弈和数据记录
3. ✅ 用传统 ML 改进评估函数
4. ✅ 尝试简单的强化学习
5. ✅ 最后挑战深度强化学习

### 评估进步
- 对比不同版本的 AI 胜率
- 记录训练曲线
- 与在线五子棋 AI 对比

### 社区交流
- GitHub 上分享你的进展
- 加入强化学习社区
- 参考其他棋类 AI 项目

## 🎯 下一步行动

我建议你从以下开始：

1. **理解现有代码** (本周)
   - 运行并测试现有的两个 AI
   - 理解评估函数如何工作
   - 分析 MCTS 的搜索过程

2. **添加数据收集** (下周)
   - 实现自我对弈模式
   - 记录对局到文件
   - 基本数据分析

3. **选择第一个 ML 项目** (第3周开始)
   - 推荐：特征学习 (容易出成果)
   - 或者：简单的 Q-Learning

需要我帮你实现其中任何一个部分吗？
