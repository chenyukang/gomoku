// AlphaZero 神经网络
// 使用 tch-rs (PyTorch for Rust)

#![cfg(feature = "alphazero")]

use tch::{nn, nn::OptimizerConfig, Device, Tensor};

/// 残差块
#[derive(Debug)]
struct ResidualBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
}

impl ResidualBlock {
    fn new(vs: &nn::Path, channels: i64) -> Self {
        let conv_config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };

        Self {
            conv1: nn::conv2d(vs, channels, channels, 3, conv_config),
            bn1: nn::batch_norm2d(vs, channels, Default::default()),
            conv2: nn::conv2d(vs, channels, channels, 3, conv_config),
            bn2: nn::batch_norm2d(vs, channels, Default::default()),
        }
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Tensor {
        let residual = xs.shallow_clone();

        let out = xs
            .apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu()
            .apply(&self.conv2)
            .apply_t(&self.bn2, train);

        (out + residual).relu()
    }
}

/// AlphaZero 双头网络（策略 + 价值）
#[derive(Debug)]
pub struct AlphaZeroNet {
    // 共享特征提取层
    conv_input: nn::Conv2D,
    bn_input: nn::BatchNorm,
    res_blocks: Vec<ResidualBlock>,

    // 策略头（Policy Head）
    policy_conv: nn::Conv2D,
    policy_bn: nn::BatchNorm,
    policy_fc: nn::Linear,

    // 价值头（Value Head）
    value_conv: nn::Conv2D,
    value_bn: nn::BatchNorm,
    value_fc1: nn::Linear,
    value_fc2: nn::Linear,

    num_filters: i64,
    device: Device,
}

impl AlphaZeroNet {
    /// 创建新的 AlphaZero 网络
    ///
    /// # 参数
    /// - num_filters: 卷积层的滤波器数量（默认 256）
    /// - num_res_blocks: 残差块数量（默认 10-20）
    pub fn new(vs: &nn::Path, num_filters: i64, num_res_blocks: i64) -> Self {
        let device = vs.device();

        // 输入层：将棋盘转换为特征
        let conv_config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_input = nn::conv2d(vs / "conv_input", 3, num_filters, 3, conv_config);
        let bn_input = nn::batch_norm2d(vs / "bn_input", num_filters, Default::default());

        // 残差块
        let mut res_blocks = Vec::new();
        for i in 0..num_res_blocks {
            let block = ResidualBlock::new(&(vs / format!("res_block_{}", i)), num_filters);
            res_blocks.push(block);
        }

        // 策略头
        let policy_conv = nn::conv2d(vs / "policy_conv", num_filters, 2, 1, Default::default());
        let policy_bn = nn::batch_norm2d(vs / "policy_bn", 2, Default::default());
        let policy_fc = nn::linear(vs / "policy_fc", 2 * 15 * 15, 15 * 15, Default::default());

        // 价值头
        let value_conv = nn::conv2d(vs / "value_conv", num_filters, 1, 1, Default::default());
        let value_bn = nn::batch_norm2d(vs / "value_bn", 1, Default::default());
        let value_fc1 = nn::linear(vs / "value_fc1", 15 * 15, 256, Default::default());
        let value_fc2 = nn::linear(vs / "value_fc2", 256, 1, Default::default());

        Self {
            conv_input,
            bn_input,
            res_blocks,
            policy_conv,
            policy_bn,
            policy_fc,
            value_conv,
            value_bn,
            value_fc1,
            value_fc2,
            num_filters,
            device,
        }
    }

    /// 获取滤波器数量
    pub fn num_filters(&self) -> i64 {
        self.num_filters
    }

    /// 前向传播
    ///
    /// # 输入
    /// - xs: (batch, 3, 15, 15) 的棋盘状态
    ///   - Channel 0: 当前玩家的棋子
    ///   - Channel 1: 对手的棋子
    ///   - Channel 2: 当前玩家标记（全1或全0）
    ///
    /// # 输出
    /// - policy: (batch, 225) 每个位置的概率
    /// - value: (batch, 1) 局面评估值 [-1, 1]
    pub fn forward(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        // 共享特征提取
        let mut out = xs
            .apply(&self.conv_input)
            .apply_t(&self.bn_input, train)
            .relu();

        // 残差块
        for block in &self.res_blocks {
            out = block.forward(&out, train);
        }

        // 策略头
        let policy = out
            .apply(&self.policy_conv)
            .apply_t(&self.policy_bn, train)
            .relu()
            .flatten(1, -1)
            .apply(&self.policy_fc)
            .log_softmax(-1, tch::Kind::Float);

        // 价值头
        let value = out
            .apply(&self.value_conv)
            .apply_t(&self.value_bn, train)
            .relu()
            .flatten(1, -1)
            .apply(&self.value_fc1)
            .relu()
            .apply(&self.value_fc2)
            .tanh();

        (policy, value)
    }

    /// 预测（不训练模式）
    pub fn predict(&self, board: &Tensor) -> (Tensor, Tensor) {
        tch::no_grad(|| self.forward(board, false))
    }

    /// 保存模型
    pub fn save(
        &mut self,
        vs: &mut nn::VarStore,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        vs.save(path)?;
        println!("✅ Model saved to {}", path);
        Ok(())
    }

    /// 加载模型
    pub fn load(vs: &mut nn::VarStore, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        vs.load(path)?;
        println!("✅ Model loaded from {}", path);
        Ok(())
    }
}

/// AlphaZero 训练器
pub struct AlphaZeroTrainer {
    vs: nn::VarStore,
    net: AlphaZeroNet,
    optimizer: nn::Optimizer,
}

impl AlphaZeroTrainer {
    pub fn new(num_filters: i64, num_res_blocks: i64, learning_rate: f64) -> Self {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let net = AlphaZeroNet::new(&vs.root(), num_filters, num_res_blocks);
        let optimizer = nn::Adam::default().build(&vs, learning_rate).unwrap();

        Self { vs, net, optimizer }
    }

    /// 训练一个批次
    ///
    /// # 输入
    /// - boards: (batch, 3, 15, 15) 棋盘状态
    /// - policy_targets: (batch, 225) 目标策略（来自 MCTS）
    /// - value_targets: (batch, 1) 目标价值（游戏结果）
    pub fn train_batch(
        &mut self,
        boards: &Tensor,
        policy_targets: &Tensor,
        value_targets: &Tensor,
    ) -> (f64, f64, f64) {
        // 前向传播
        let (policy_pred, value_pred) = self.net.forward(boards, true);

        // 策略损失（交叉熵）
        let policy_loss = -(policy_targets * &policy_pred)
            .sum(tch::Kind::Float)
            .mean(tch::Kind::Float);

        // 价值损失（MSE）
        let value_loss = (&value_pred - value_targets)
            .pow_tensor_scalar(2)
            .mean(tch::Kind::Float);

        // 总损失
        let total_loss = &policy_loss + &value_loss;

        // 反向传播
        self.optimizer.backward_step(&total_loss);

        (
            total_loss.double_value(&[]),
            policy_loss.double_value(&[]),
            value_loss.double_value(&[]),
        )
    }

    /// 获取网络引用
    pub fn net(&self) -> &AlphaZeroNet {
        &self.net
    }

    /// 保存模型
    pub fn save(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.net.save(&mut self.vs, path)
    }

    /// 加载模型
    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        AlphaZeroNet::load(&mut self.vs, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "alphazero")]
    fn test_network_creation() {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = AlphaZeroNet::new(&vs.root(), 64, 5);

        // 创建一个随机输入
        let input = Tensor::randn(&[1, 3, 15, 15], (tch::Kind::Float, Device::Cpu));
        let (policy, value) = net.predict(&input);

        assert_eq!(policy.size(), vec![1, 225]);
        assert_eq!(value.size(), vec![1, 1]);

        println!("✅ Network test passed");
    }
}
