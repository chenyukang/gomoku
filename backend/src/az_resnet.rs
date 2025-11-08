use std::convert::TryFrom;
use tch::{nn, nn::Module, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};

// ResNet残差块
#[derive(Debug)]
struct ResidualBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
}

impl ResidualBlock {
    fn new(vs: &nn::Path, num_filters: i64) -> Self {
        let config = nn::ConvConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };

        let conv1 = nn::conv2d(vs / "conv1", num_filters, num_filters, 3, config);
        let bn1 = nn::batch_norm2d(vs / "bn1", num_filters, Default::default());
        let conv2 = nn::conv2d(vs / "conv2", num_filters, num_filters, 3, config);
        let bn2 = nn::batch_norm2d(vs / "bn2", num_filters, Default::default());

        ResidualBlock {
            conv1,
            bn1,
            conv2,
            bn2,
        }
    }
}

impl ModuleT for ResidualBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let identity = xs.shallow_clone();

        // First conv block
        let mut x = xs.apply(&self.conv1);
        x = self.bn1.forward_t(&x, train);
        x = x.relu();

        // Second conv block
        x = x.apply(&self.conv2);
        x = self.bn2.forward_t(&x, train);

        // Residual connection
        (x + identity).relu()
    }
}

// Connect4 ResNet
#[derive(Debug)]
pub struct Connect4ResNet {
    // Initial convolution
    conv_init: nn::Conv2D,
    bn_init: nn::BatchNorm,

    // Residual blocks
    residual_blocks: Vec<ResidualBlock>,

    // Policy head
    policy_conv: nn::Conv2D,
    policy_bn: nn::BatchNorm,
    policy_fc: nn::Linear,

    // Value head
    value_conv: nn::Conv2D,
    value_bn: nn::BatchNorm,
    value_fc1: nn::Linear,
    value_fc2: nn::Linear,

    device: Device,
}

impl Connect4ResNet {
    pub fn new(vs: &nn::Path, num_filters: i64, num_residual_blocks: i64) -> Self {
        let device = vs.device();

        // Initial convolution (3 channels -> num_filters)
        let config = nn::ConvConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let conv_init = nn::conv2d(vs / "conv_init", 3, num_filters, 3, config);
        let bn_init = nn::batch_norm2d(vs / "bn_init", num_filters, Default::default());

        // Residual tower
        let mut residual_blocks = Vec::new();
        for i in 0..num_residual_blocks {
            residual_blocks.push(ResidualBlock::new(
                &(vs / format!("res_{}", i)),
                num_filters,
            ));
        }

        // Policy head
        let policy_conv = nn::conv2d(vs / "policy_conv", num_filters, 2, 1, Default::default());
        let policy_bn = nn::batch_norm2d(vs / "policy_bn", 2, Default::default());
        let policy_fc = nn::linear(vs / "policy_fc", 2 * 6 * 7, 7, Default::default());

        // Value head
        let value_conv = nn::conv2d(vs / "value_conv", num_filters, 1, 1, Default::default());
        let value_bn = nn::batch_norm2d(vs / "value_bn", 1, Default::default());
        let value_fc1 = nn::linear(vs / "value_fc1", 6 * 7, 256, Default::default());
        let value_fc2 = nn::linear(vs / "value_fc2", 256, 1, Default::default());

        Connect4ResNet {
            conv_init,
            bn_init,
            residual_blocks,
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

    pub fn predict(&self, board: &Tensor) -> (Tensor, Tensor) {
        self.forward_train(board, false)
    }

    pub fn forward_train(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        // Initial convolution
        let mut x = xs.apply(&self.conv_init);
        x = self.bn_init.forward_t(&x, train);
        x = x.relu();

        // Residual tower
        for block in &self.residual_blocks {
            x = block.forward_t(&x, train);
        }

        // Policy head
        let mut policy = x.apply(&self.policy_conv);
        policy = self.policy_bn.forward_t(&policy, train);
        policy = policy.relu().flat_view().apply(&self.policy_fc);

        // Value head
        let mut value = x.apply(&self.value_conv);
        value = self.value_bn.forward_t(&value, train);
        value = value
            .relu()
            .flat_view()
            .apply(&self.value_fc1)
            .relu()
            .apply(&self.value_fc2)
            .tanh();

        (policy, value)
    }
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

// Trainer for Connect4 ResNet
pub struct Connect4ResNetTrainer {
    vs: nn::VarStore,
    pub net: Connect4ResNet,
    pub opt: nn::Optimizer,
}

impl Connect4ResNetTrainer {
    pub fn new(num_filters: i64, num_residual_blocks: i64, learning_rate: f64) -> Self {
        let device = best_device();
        println!("使用设备: {:?}", device);
        println!(
            "网络架构: ResNet-{} with {} filters",
            num_residual_blocks, num_filters
        );

        let vs = nn::VarStore::new(device);
        let net = Connect4ResNet::new(&vs.root(), num_filters, num_residual_blocks);
        let opt = nn::Adam::default().build(&vs, learning_rate).unwrap();

        Self { vs, net, opt }
    }

    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // 尝试更宽容的加载方式
        self.vs.freeze();
        match self.vs.load(path) {
            Ok(_) => {
                println!("✅ 模型加载成功");
                Ok(())
            },
            Err(e) => {
                eprintln!("❌ 模型加载失败: {}", e);
                Err(Box::new(e))
            }
        }
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.vs.save(path)?;
        Ok(())
    }

    pub fn device(&self) -> Device {
        self.vs.device()
    }

    pub fn train_batch(
        &mut self,
        boards: &Tensor,
        target_policies: &Tensor,
        target_values: &Tensor,
    ) -> (f64, f64, f64) {
        let (pred_policy, pred_value) = self.net.forward_train(boards, true);

        // Policy loss (交叉熵)
        let policy_loss = pred_policy.log_softmax(-1, tch::Kind::Float) * target_policies;
        let policy_loss = -policy_loss.sum(tch::Kind::Float).mean(tch::Kind::Float);

        // Value loss (MSE)
        let value_loss = (&pred_value - target_values)
            .pow_tensor_scalar(2)
            .mean(tch::Kind::Float);

        // Total loss
        let total_loss = &policy_loss + &value_loss;

        self.opt.zero_grad();
        total_loss.backward();
        self.opt.step();

        (
            f64::try_from(policy_loss).unwrap(),
            f64::try_from(value_loss).unwrap(),
            f64::try_from(total_loss).unwrap(),
        )
    }
}
