#!/usr/bin/env python3
"""
使用深度学习改进五子棋评估函数

这个示例展示如何用卷积神经网络 (CNN) 学习棋盘评估。
需要安装: pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class GomokuDataset(Dataset):
    """五子棋数据集"""

    def __init__(self, csv_file):
        """
        Args:
            csv_file: CSV 文件路径
        """
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回:
            board: (2, 15, 15) 的张量 - 两个通道分别表示黑白棋子
            value: 标量 - 这个局面的价值 (基于最终胜负)
        """
        row = self.data.iloc[idx]

        # 解析棋盘
        board_str = row['board']
        board = self.parse_board(board_str, row['player'])

        # 获取价值 (基于最终奖励)
        value = float(row['final_reward'])

        return torch.FloatTensor(board), torch.FloatTensor([value])

    def parse_board(self, board_str, current_player):
        """
        将棋盘字符串转换为 2 通道的张量

        Channel 0: 当前玩家的棋子位置
        Channel 1: 对手的棋子位置
        """
        # 创建 15x15 的棋盘
        board = np.zeros((2, 15, 15), dtype=np.float32)

        opponent = 2 if current_player == 1 else 1

        for i, char in enumerate(board_str[:225]):  # 只取前 225 个字符
            row = i // 15
            col = i % 15

            if char == str(current_player):
                board[0, row, col] = 1.0
            elif char == str(opponent):
                board[1, row, col] = 1.0

        return board


class GomokuCNN(nn.Module):
    """
    卷积神经网络评估棋盘

    架构参考 AlphaGo Zero，但简化版本：
    - 多层卷积提取特征
    - 残差连接
    - 输出棋盘价值
    """

    def __init__(self, num_filters=64, num_blocks=5):
        super(GomokuCNN, self).__init__()

        # 第一层：输入转换
        self.conv_input = nn.Sequential(
            nn.Conv2d(2, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_blocks)
        ])

        # 价值头 (Value Head)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(15 * 15, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # 输出 [-1, 1] 范围的价值
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 2, 15, 15) 棋盘状态
        Returns:
            value: (batch, 1) 局面评估
        """
        x = self.conv_input(x)

        for block in self.residual_blocks:
            x = block(x)

        value = self.value_head(x)

        return value


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = torch.relu(out)
        return out


class GomokuTrainer:
    """训练器"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, train_loader):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0

        for boards, values in train_loader:
            boards = boards.to(self.device)
            values = values.to(self.device)

            # 前向传播
            predictions = self.model(boards)
            loss = self.criterion(predictions, values)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for boards, values in val_loader:
                boards = boards.to(self.device)
                values = values.to(self.device)

                predictions = self.model(boards)
                loss = self.criterion(predictions, values)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, num_epochs=10):
        """完整训练流程"""
        print(f"🚀 开始训练 (设备: {self.device})")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        print("✅ 训练完成!")

    def plot_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.savefig('data/training_history.png', dpi=100)
        print("📊 训练历史已保存到 data/training_history.png")
        plt.show()

    def save_model(self, path='data/gomoku_model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"💾 模型已保存到 {path}")

    def load_model(self, path='data/gomoku_model.pth'):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"✅ 模型已从 {path} 加载")


def main():
    """主函数"""
    print("="*60)
    print("🧠 五子棋深度学习评估函数")
    print("="*60)

    # 检查 CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n📚 加载数据...")
    try:
        dataset = GomokuDataset('data/games.csv')
        print(f"✅ 加载了 {len(dataset)} 条数据")
    except FileNotFoundError:
        print("❌ 找不到 data/games.csv")
        print("请先运行: cargo run --release --bin ml_trainer -- --selfplay 100")
        return

    if len(dataset) < 100:
        print(f"⚠️  数据量较少 ({len(dataset)} 条)，建议至少 1000 条")
        print("运行: cargo run --release --bin ml_trainer -- --selfplay 1000")

    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print(f"训练集: {train_size} 条")
    print(f"验证集: {val_size} 条")

    # 创建模型
    print("\n🏗️  创建模型...")
    model = GomokuCNN(num_filters=64, num_blocks=5)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    trainer = GomokuTrainer(model, device=device)
    trainer.train(train_loader, val_loader, num_epochs=20)

    # 保存
    trainer.save_model()
    trainer.plot_history()

    print("\n💡 下一步:")
    print("1. 增加训练数据量")
    print("2. 调整网络架构和超参数")
    print("3. 将训练好的模型集成到 Rust 代码中")
    print("4. 使用 tch-rs 在 Rust 中加载 PyTorch 模型")


def demo_inference():
    """演示推理过程"""
    print("\n" + "="*60)
    print("🔮 模型推理演示")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = GomokuCNN()
    trainer = GomokuTrainer(model, device=device)

    try:
        trainer.load_model()
    except FileNotFoundError:
        print("❌ 模型文件不存在，请先训练模型")
        return

    # 创建一个测试棋盘
    test_board = np.zeros((2, 15, 15), dtype=np.float32)
    # 在中心放一些棋子
    test_board[0, 7, 7] = 1.0  # 当前玩家
    test_board[1, 7, 8] = 1.0  # 对手
    test_board[0, 8, 7] = 1.0  # 当前玩家

    # 推理
    model.eval()
    with torch.no_grad():
        board_tensor = torch.FloatTensor(test_board).unsqueeze(0).to(device)
        value = model(board_tensor)
        print(f"\n棋盘评估值: {value.item():.4f}")
        print("(正值表示当前玩家优势，负值表示劣势)")


if __name__ == '__main__':
    # 检查 PyTorch 是否安装
    try:
        import torch
        main()
        # demo_inference()  # 取消注释以运行推理演示
    except ImportError:
        print("❌ PyTorch 未安装")
        print("\n安装方法:")
        print("  pip install torch torchvision")
        print("\n或访问: https://pytorch.org/get-started/locally/")
