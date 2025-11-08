// AlphaZero 神经网络 - 专门为 Connect4 设计的简化版本
// 输入: (3, 6, 7) - 3个通道，6行7列
// 输出: policy (7,) 和 value (1,)

#![cfg(feature = "alphazero")]

use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

pub struct Connect4Net {
    // 共享卷积层
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    conv3: nn::Conv2D,
    bn3: nn::BatchNorm,

    // 策略头 (policy head)
    policy_conv: nn::Conv2D,
    policy_bn: nn::BatchNorm,
    policy_fc: nn::Linear,

    // 价值头 (value head)
    value_conv: nn::Conv2D,
    value_bn: nn::BatchNorm,
    value_fc1: nn::Linear,
    value_fc2: nn::Linear,

    device: Device,
}

impl Connect4Net {
    pub fn new(vs: &nn::Path, num_filters: i64) -> Self {
        let device = vs.device();

        let conv_cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };

        // 共享卷积层
        let conv1 = nn::conv2d(vs / "conv1", 3, num_filters, 3, conv_cfg);
        let bn1 = nn::batch_norm2d(vs / "bn1", num_filters, Default::default());
        let conv2 = nn::conv2d(vs / "conv2", num_filters, num_filters, 3, conv_cfg);
        let bn2 = nn::batch_norm2d(vs / "bn2", num_filters, Default::default());
        let conv3 = nn::conv2d(vs / "conv3", num_filters, num_filters, 3, conv_cfg);
        let bn3 = nn::batch_norm2d(vs / "bn3", num_filters, Default::default());

        // 策略头
        let policy_conv = nn::conv2d(vs / "policy_conv", num_filters, 2, 1, Default::default());
        let policy_bn = nn::batch_norm2d(vs / "policy_bn", 2, Default::default());
        let policy_fc = nn::linear(vs / "policy_fc", 2 * 6 * 7, 7, Default::default());

        // 价值头
        let value_conv = nn::conv2d(vs / "value_conv", num_filters, 1, 1, Default::default());
        let value_bn = nn::batch_norm2d(vs / "value_bn", 1, Default::default());
        let value_fc1 = nn::linear(vs / "value_fc1", 6 * 7, 64, Default::default());
        let value_fc2 = nn::linear(vs / "value_fc2", 64, 1, Default::default());

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            policy_conv,
            policy_bn,
            policy_fc,
            value_conv,
            value_bn,
            value_fc1,
            value_fc2,
            device,
        }
    }

    /// 前向传播
    ///
    /// 输入: (batch, 3, 6, 7) 的棋盘状态
    /// 输出: (policy, value)
    ///   - policy: (batch, 7) log概率
    ///   - value: (batch, 1) 值在 [-1, 1] 之间
    pub fn forward(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        // 共享层
        let mut x = xs.apply(&self.conv1).apply_t(&self.bn1, train).relu();

        x = x.apply(&self.conv2).apply_t(&self.bn2, train).relu();

        x = x.apply(&self.conv3).apply_t(&self.bn3, train).relu();

        // 策略头
        let policy = x
            .apply(&self.policy_conv)
            .apply_t(&self.policy_bn, train)
            .relu()
            .flatten(1, -1)
            .apply(&self.policy_fc)
            .log_softmax(-1, Kind::Float);

        // 价值头
        let value = x
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

    /// 预测（推理模式）
    pub fn predict(&self, board_tensor: &Tensor) -> (Tensor, Tensor) {
        tch::no_grad(|| {
            let input = board_tensor.to_device(self.device);
            self.forward(&input, false)
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

/// AlphaZero 训练器
pub struct Connect4Trainer {
    vs: nn::VarStore,
    pub net: Connect4Net,
    optimizer: nn::Optimizer,
}

fn best_device() -> Device {
    // 优先级: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    if std::panic::catch_unwind(|| {
        let _ = Tensor::zeros(&[1], (tch::Kind::Float, Device::Mps));
    })
    .is_ok()
    {
        Device::Mps
    } else if Device::cuda_if_available() != Device::Cpu {
        Device::cuda_if_available()
    } else {
        Device::Cpu
    }
}

impl Connect4Trainer {
    pub fn new(num_filters: i64, learning_rate: f64) -> Self {
        let device = best_device();
        println!("使用设备: {:?}", device);

        let vs = nn::VarStore::new(device);
        let net = Connect4Net::new(&vs.root(), num_filters);
        let optimizer = nn::Adam::default().build(&vs, learning_rate).unwrap();

        Self { vs, net, optimizer }
    }

    /// 训练一个批次
    ///
    /// boards: (batch, 3, 6, 7)
    /// target_policies: (batch, 7)
    /// target_values: (batch, 1)
    pub fn train_batch(
        &mut self,
        boards: &Tensor,
        target_policies: &Tensor,
        target_values: &Tensor,
    ) -> (f64, f64, f64) {
        let boards = boards.to_device(self.vs.device());
        let target_policies = target_policies.to_device(self.vs.device());
        let target_values = target_values.to_device(self.vs.device());

        // 前向传播
        let (pred_policy, pred_value) = self.net.forward(&boards, true);

        // 策略损失：交叉熵
        let policy_loss = -(target_policies * pred_policy)
            .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
            .mean(Kind::Float);

        // 价值损失：MSE
        let value_loss = (pred_value - target_values)
            .pow_tensor_scalar(2)
            .mean(Kind::Float);

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

    /// 保存模型
    pub fn save(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.vs.save(path)?;
        println!("✅ 模型已保存到: {}", path);
        Ok(())
    }

    /// 加载模型
    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.vs.load(path)?;
        println!("✅ 模型已从 {} 加载", path);
        Ok(())
    }

    pub fn device(&self) -> Device {
        self.vs.device()
    }
}
