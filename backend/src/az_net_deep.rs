// 更深的神经网络 - 5层CNN
#![cfg(feature = "alphazero")]

use std::convert::TryFrom;
use tch::{
    nn,
    nn::{Module, ModuleT, OptimizerConfig},
    Device, Kind, Tensor,
};

pub struct Connect4NetDeep {
    // 5个共享卷积层
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    conv3: nn::Conv2D,
    bn3: nn::BatchNorm,
    conv4: nn::Conv2D,
    bn4: nn::BatchNorm,
    conv5: nn::Conv2D,
    bn5: nn::BatchNorm,

    // 策略头
    policy_conv: nn::Conv2D,
    policy_bn: nn::BatchNorm,
    policy_fc: nn::Linear,

    // 价值头
    value_conv: nn::Conv2D,
    value_bn: nn::BatchNorm,
    value_fc1: nn::Linear,
    value_fc2: nn::Linear,

    device: Device,
}

impl Connect4NetDeep {
    pub fn new(vs: &nn::Path, num_filters: i64) -> Self {
        let device = vs.device();

        let conv_cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };

        // 5个共享卷积层
        let conv1 = nn::conv2d(vs / "conv1", 3, num_filters, 3, conv_cfg);
        let bn1 = nn::batch_norm2d(vs / "bn1", num_filters, Default::default());
        let conv2 = nn::conv2d(vs / "conv2", num_filters, num_filters, 3, conv_cfg);
        let bn2 = nn::batch_norm2d(vs / "bn2", num_filters, Default::default());
        let conv3 = nn::conv2d(vs / "conv3", num_filters, num_filters, 3, conv_cfg);
        let bn3 = nn::batch_norm2d(vs / "bn3", num_filters, Default::default());
        let conv4 = nn::conv2d(vs / "conv4", num_filters, num_filters, 3, conv_cfg);
        let bn4 = nn::batch_norm2d(vs / "bn4", num_filters, Default::default());
        let conv5 = nn::conv2d(vs / "conv5", num_filters, num_filters, 3, conv_cfg);
        let bn5 = nn::batch_norm2d(vs / "bn5", num_filters, Default::default());

        // 策略头
        let policy_conv = nn::conv2d(vs / "policy_conv", num_filters, 2, 1, Default::default());
        let policy_bn = nn::batch_norm2d(vs / "policy_bn", 2, Default::default());
        let policy_fc = nn::linear(vs / "policy_fc", 2 * 6 * 7, 7, Default::default());

        // 价值头
        let value_conv = nn::conv2d(vs / "value_conv", num_filters, 1, 1, Default::default());
        let value_bn = nn::batch_norm2d(vs / "value_bn", 1, Default::default());
        let value_fc1 = nn::linear(vs / "value_fc1", 6 * 7, 128, Default::default());
        let value_fc2 = nn::linear(vs / "value_fc2", 128, 1, Default::default());

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            conv4,
            bn4,
            conv5,
            bn5,
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

    pub fn forward(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        // 5层共享卷积
        let mut x = xs.to_device(self.device).to_kind(Kind::Float);

        x = self.conv1.forward(&x);
        x = self.bn1.forward_t(&x, train).relu();

        x = self.conv2.forward(&x);
        x = self.bn2.forward_t(&x, train).relu();

        x = self.conv3.forward(&x);
        x = self.bn3.forward_t(&x, train).relu();

        x = self.conv4.forward(&x);
        x = self.bn4.forward_t(&x, train).relu();

        x = self.conv5.forward(&x);
        x = self.bn5.forward_t(&x, train).relu();

        // 策略头
        let policy = self.policy_conv.forward(&x);
        let policy = self.policy_bn.forward_t(&policy, train).relu();
        let policy = policy.view([-1, 2 * 6 * 7]);
        let policy = self.policy_fc.forward(&policy); // log_softmax在外部

        // 价值头
        let value = self.value_conv.forward(&x);
        let value = self.value_bn.forward_t(&value, train).relu();
        let value = value.view([-1, 6 * 7]);
        let value = self.value_fc1.forward(&value).relu();
        let value = self.value_fc2.forward(&value).tanh();

        (policy, value)
    }

    pub fn predict(&self, xs: &Tensor) -> (Tensor, Tensor) {
        tch::no_grad(|| self.forward(xs, false))
    }
}

pub struct Connect4TrainerDeep {
    pub vs: nn::VarStore,
    pub net: Connect4NetDeep,
    pub opt: nn::Optimizer,
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

impl Connect4TrainerDeep {
    pub fn new(num_filters: i64, learning_rate: f64) -> Self {
        let device = best_device();
        println!("使用设备: {:?}", device);

        let vs = nn::VarStore::new(device);
        let net = Connect4NetDeep::new(&vs.root(), num_filters);
        let opt = nn::Adam::default().build(&vs, learning_rate).unwrap();

        Self { vs, net, opt }
    }

    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.vs.load(path)?;
        Ok(())
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.vs.save(path)?;
        Ok(())
    }

    pub fn train_batch(
        &mut self,
        boards: &Tensor,
        target_policies: &Tensor,
        target_values: &Tensor,
    ) -> (f64, f64, f64) {
        let (policy_logits, value_preds) = self.net.forward(boards, true);

        // Policy loss: 交叉熵
        let policy_loss = policy_logits
            .log_softmax(-1, Kind::Float)
            .multiply(target_policies)
            .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
            .neg()
            .mean(Kind::Float);

        // Value loss: MSE
        let value_loss = (value_preds - target_values).square().mean(Kind::Float);

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
