#!/usr/bin/env python3
"""
简单的 Q-Learning 实现示例

这是一个教学用的简化版本，展示强化学习的基本概念。
在实际应用中需要更复杂的状态表示和网络架构。
"""

import numpy as np
import pickle
from collections import defaultdict

class SimpleQLearning:
    """
    简单的 Q-Learning 算法实现

    状态: 简化的棋盘表示 (可以改进为更复杂的特征)
    动作: 棋盘上的位置 (x, y)
    奖励: 赢 = +1, 输 = -1, 平局 = 0, 中间步骤 = -0.01
    """

    def __init__(self,
                 learning_rate=0.1,
                 discount_factor=0.95,
                 epsilon=0.1):
        """
        参数:
            learning_rate: 学习率 (alpha)
            discount_factor: 折扣因子 (gamma)
            epsilon: 探索率 (epsilon-greedy)
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-table: (state, action) -> Q-value
        # 使用 defaultdict 自动初始化为 0
        self.q_table = defaultdict(float)

        # 统计
        self.total_updates = 0

    def state_to_key(self, board_str):
        """
        将棋盘状态转换为可哈希的 key

        实际应用中可以用更智能的方法:
        - 对称性处理 (旋转、镜像)
        - 局部特征提取
        - 神经网络编码
        """
        return board_str

    def get_q_value(self, state, action):
        """获取 Q 值"""
        key = (self.state_to_key(state), action)
        return self.q_table[key]

    def update_q_value(self, state, action, reward, next_state, next_valid_actions):
        """
        更新 Q 值

        Q(s,a) <- Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.get_q_value(state, action)

        # 计算下一个状态的最大 Q 值
        if next_valid_actions:
            max_next_q = max(self.get_q_value(next_state, a)
                            for a in next_valid_actions)
        else:
            max_next_q = 0  # 终止状态

        # Q-learning 更新公式
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)

        key = (self.state_to_key(state), action)
        self.q_table[key] = new_q
        self.total_updates += 1

    def choose_action(self, state, valid_actions):
        """
        选择动作 (epsilon-greedy 策略)

        以 epsilon 概率随机探索
        以 1-epsilon 概率选择最优动作
        """
        if not valid_actions:
            return None

        # 探索
        if np.random.random() < self.epsilon:
            return valid_actions[np.random.randint(len(valid_actions))]

        # 利用: 选择 Q 值最大的动作
        q_values = [(action, self.get_q_value(state, action))
                    for action in valid_actions]

        # 找出最大 Q 值
        max_q = max(q for _, q in q_values)

        # 如果有多个最大值，随机选一个
        best_actions = [action for action, q in q_values if q == max_q]
        return best_actions[np.random.randint(len(best_actions))]

    def save(self, filename):
        """保存 Q-table"""
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"✅ Q-table 已保存到 {filename}")
        print(f"   总共 {len(self.q_table)} 个状态-动作对")

    def load(self, filename):
        """加载 Q-table"""
        try:
            with open(filename, 'rb') as f:
                self.q_table = defaultdict(float, pickle.load(f))
            print(f"✅ 从 {filename} 加载了 {len(self.q_table)} 个状态-动作对")
        except FileNotFoundError:
            print(f"⚠️  文件不存在: {filename}")

    def get_stats(self):
        """获取统计信息"""
        return {
            'q_table_size': len(self.q_table),
            'total_updates': self.total_updates,
            'avg_q_value': np.mean(list(self.q_table.values())) if self.q_table else 0,
            'max_q_value': max(self.q_table.values()) if self.q_table else 0,
            'min_q_value': min(self.q_table.values()) if self.q_table else 0,
        }


def demo_training():
    """演示训练过程 (简化版)"""
    print("="*60)
    print("🎯 Q-Learning 演示")
    print("="*60)

    # 创建 Q-learning 智能体
    agent = SimpleQLearning(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.2  # 20% 探索
    )

    # 模拟训练过程
    print("\n📚 训练中...")
    print("(这只是一个演示，实际需要与游戏环境交互)\n")

    # 模拟一些状态和动作
    dummy_state = "." * 225  # 空棋盘
    dummy_actions = [(7, 7), (7, 8), (8, 7), (8, 8)]  # 中心区域

    for episode in range(100):
        state = dummy_state
        action = agent.choose_action(state, dummy_actions)

        # 模拟奖励 (实际应该来自游戏结果)
        reward = np.random.choice([1, -1, -0.01], p=[0.3, 0.3, 0.4])

        # 模拟下一个状态
        next_state = dummy_state  # 简化
        next_actions = dummy_actions

        # 更新 Q 值
        agent.update_q_value(state, action, reward, next_state, next_actions)

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/100 完成")

    # 显示统计
    print("\n📊 训练统计:")
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 保存模型
    agent.save('data/q_table.pkl')

    print("\n💡 实际应用步骤:")
    print("1. 修改 Rust 代码,添加 Q-learning 智能体")
    print("2. 让智能体与环境交互(自我对弈)")
    print("3. 收集经验并更新 Q-table")
    print("4. 评估性能并调整超参数")


def integration_guide():
    """集成到 Rust 项目的指南"""
    print("\n" + "="*60)
    print("🔧 与 Rust 项目集成")
    print("="*60)

    guide = """
方案 1: Python 作为训练器
------------------------
1. Rust 提供游戏环境和快速模拟
2. Python 训练 Q-learning / 神经网络
3. 将学到的参数导出为 JSON/二进制
4. Rust 加载参数用于推理

方案 2: PyO3 集成
-----------------
1. 使用 PyO3 在 Rust 中调用 Python
2. 训练和推理都可以在 Rust 中完成
3. 性能较好,部署简单

方案 3: 纯 Rust 实现
--------------------
1. 使用 Rust 机器学习库 (linfa, smartcore)
2. 或使用 tch-rs (PyTorch 绑定)
3. 完全的类型安全和性能

推荐开始方式:
------------
1. 先用 Python 快速原型 (本文件)
2. 验证算法有效后
3. 迁移到 Rust (如果需要性能)
"""
    print(guide)


if __name__ == '__main__':
    demo_training()
    integration_guide()
