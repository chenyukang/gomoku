#!/usr/bin/env python3
"""
五子棋训练数据分析和简单机器学习示例

使用方法:
1. 先运行 Rust 程序生成数据: cargo run --bin gomoku -- --selfplay
2. 运行此脚本分析数据: python ml_examples/analyze_data.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json

class GomokuDataAnalyzer:
    """五子棋数据分析器"""

    def __init__(self, csv_file='data/games.csv', json_file='data/games.json'):
        self.csv_file = csv_file
        self.json_file = json_file
        self.df = None
        self.games = []

    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ 加载了 {len(self.df)} 条数据")
            print(f"📊 数据预览:")
            print(self.df.head())
            print(f"\n📈 数据统计:")
            print(self.df.describe())
        except FileNotFoundError:
            print(f"❌ 找不到文件: {self.csv_file}")
            print("请先运行自我对弈生成数据")
            return False
        return True

    def load_json_games(self):
        """加载 JSON 格式的完整游戏记录"""
        try:
            with open(self.json_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.games.append(json.loads(line))
            print(f"✅ 加载了 {len(self.games)} 局游戏")
        except FileNotFoundError:
            print(f"⚠️  找不到 JSON 文件: {self.json_file}")

    def visualize_basics(self):
        """基础数据可视化"""
        if self.df is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 胜率分布
        winner_counts = self.df['winner'].value_counts()
        axes[0, 0].bar(['平局', 'Player 1', 'Player 2'],
                       [winner_counts.get(0, 0), winner_counts.get(1, 0), winner_counts.get(2, 0)])
        axes[0, 0].set_title('胜负分布')
        axes[0, 0].set_ylabel('次数')

        # 2. 步数分布
        axes[0, 1].hist(self.df['step'], bins=30, edgecolor='black')
        axes[0, 1].set_title('步数分布')
        axes[0, 1].set_xlabel('步数')
        axes[0, 1].set_ylabel('频率')

        # 3. 评估分数分布
        axes[1, 0].hist(self.df['eval_score'], bins=50, edgecolor='black')
        axes[1, 0].set_title('评估分数分布')
        axes[1, 0].set_xlabel('评估分数')
        axes[1, 0].set_ylabel('频率')

        # 4. 落子位置热力图
        heatmap_data = np.zeros((15, 15))
        for _, row in self.df.iterrows():
            x, y = int(row['move_x']), int(row['move_y'])
            if 0 <= x < 15 and 0 <= y < 15:
                heatmap_data[x][y] += 1

        sns.heatmap(heatmap_data, ax=axes[1, 1], cmap='YlOrRd',
                    cbar_kws={'label': '落子次数'})
        axes[1, 1].set_title('落子位置热力图')

        plt.tight_layout()
        plt.savefig('data/analysis_basic.png', dpi=100)
        print("✅ 基础分析图已保存到 data/analysis_basic.png")
        plt.show()

    def extract_features(self, board_str):
        """从棋盘字符串中提取特征"""
        # 简单特征提取
        features = {
            'center_control': 0,  # 中心控制
            'corner_control': 0,  # 角落控制
            'player1_stones': board_str.count('1'),
            'player2_stones': board_str.count('2'),
            'empty_cells': board_str.count('.') + board_str.count('0'),
        }

        # 中心区域 (7,7) 附近
        if len(board_str) == 225:  # 15x15
            center_idx = 7 * 15 + 7
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    idx = center_idx + dx * 15 + dy
                    if 0 <= idx < 225:
                        if board_str[idx] != '.' and board_str[idx] != '0':
                            features['center_control'] += 1

        return features

    def train_simple_model(self):
        """训练一个简单的预测模型 - 预测胜负"""
        if self.df is None or len(self.df) == 0:
            print("❌ 没有数据可用于训练")
            return

        print("\n🤖 训练预测模型...")

        # 准备特征
        X = self.df[['move_x', 'move_y', 'eval_score', 'step', 'player']].values
        y = (self.df['final_reward'] > 0).astype(int)  # 转换为二分类：赢(1) vs 不赢(0)

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n✅ 模型准确率: {accuracy:.2%}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['不赢', '赢']))

        # 特征重要性
        feature_names = ['move_x', 'move_y', 'eval_score', 'step', 'player']
        importances = model.feature_importances_

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances)
        plt.xlabel('重要性')
        plt.title('特征重要性分析')
        plt.tight_layout()
        plt.savefig('data/feature_importance.png', dpi=100)
        print("✅ 特征重要性图已保存到 data/feature_importance.png")
        plt.show()

        return model

    def train_eval_function(self):
        """学习更好的评估函数"""
        if self.df is None or len(self.df) == 0:
            print("❌ 没有数据可用于训练")
            return

        print("\n🎯 训练评估函数...")

        # 使用最终奖励作为目标
        X = self.df[['move_x', 'move_y', 'step', 'player']].values
        y = self.df['final_reward'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        # 评估
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        print(f"训练集 R²: {train_score:.3f}")
        print(f"测试集 R²: {test_score:.3f}")
        print(f"\n学到的权重:")
        print(f"  move_x:  {model.coef_[0]:.4f}")
        print(f"  move_y:  {model.coef_[1]:.4f}")
        print(f"  step:    {model.coef_[2]:.4f}")
        print(f"  player:  {model.coef_[3]:.4f}")
        print(f"  bias:    {model.intercept_:.4f}")

        return model

    def analyze_game_patterns(self):
        """分析游戏模式"""
        if not self.games:
            print("⚠️  没有游戏数据")
            return

        print("\n🔍 游戏模式分析:")

        avg_steps = np.mean([g['total_steps'] for g in self.games])
        print(f"  平均步数: {avg_steps:.1f}")

        win_first_move = sum(1 for g in self.games
                              if g.get('winner') == 1) / len(self.games)
        print(f"  先手胜率: {win_first_move:.1%}")

        # 分析开局位置偏好
        opening_moves = {}
        for game in self.games:
            if len(game['states']) > 0:
                first_move = game['states'][0]
                pos = (first_move['move_x'], first_move['move_y'])
                opening_moves[pos] = opening_moves.get(pos, 0) + 1

        print("\n  最常见的开局位置 (Top 5):")
        sorted_openings = sorted(opening_moves.items(),
                                 key=lambda x: x[1], reverse=True)[:5]
        for (x, y), count in sorted_openings:
            print(f"    ({x}, {y}): {count} 次")

def main():
    """主函数"""
    print("="*60)
    print("🎮 五子棋机器学习数据分析")
    print("="*60)

    analyzer = GomokuDataAnalyzer()

    # 加载数据
    if not analyzer.load_data():
        print("\n💡 提示: 请先运行以下命令生成数据:")
        print("   cd backend")
        print("   cargo run --release --bin gomoku -- --selfplay 10")
        return

    analyzer.load_json_games()

    # 基础可视化
    analyzer.visualize_basics()

    # 分析游戏模式
    if analyzer.games:
        analyzer.analyze_game_patterns()

    # 训练模型
    analyzer.train_simple_model()
    analyzer.train_eval_function()

    print("\n✅ 分析完成！")
    print("\n💡 下一步建议:")
    print("   1. 收集更多数据 (至少 1000+ 局)")
    print("   2. 尝试深度学习方法")
    print("   3. 实现强化学习算法")

if __name__ == '__main__':
    main()
