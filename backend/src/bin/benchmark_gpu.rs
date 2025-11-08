use std::time::Instant;
use tch::{nn, Device, Kind, Tensor};

fn benchmark_device(device: Device, name: &str) {
    println!("\n🔥 测试 {} 性能:", name);

    // 创建一个简单的3层CNN (Connect4: 6x7 board)
    let vs = nn::VarStore::new(device);
    let root = vs.root();

    let config = nn::ConvConfig {
        stride: 1,
        padding: 1,
        ..Default::default()
    };
    let conv1 = nn::conv2d(&root, 3, 64, 3, config);
    let conv2 = nn::conv2d(&root, 64, 64, 3, config);
    let conv3 = nn::conv2d(&root, 64, 128, 3, config);

    // 模拟Connect4训练批次
    let batch_size = 64;
    let iterations = 100;

    println!("  批次大小: {}", batch_size);
    println!("  迭代次数: {}", iterations);

    let start = Instant::now();

    for _ in 0..iterations {
        // 前向传播
        let input = Tensor::randn(&[batch_size, 3, 6, 7], (Kind::Float, device));
        let x = input.apply(&conv1).relu();
        let x = x.apply(&conv2).relu();
        let x = x.apply(&conv3).relu();

        // 模拟loss计算和反向传播
        let loss = x.sum(Kind::Float);
        let _ = loss.backward();
    }

    let elapsed = start.elapsed();
    let per_iter = elapsed.as_millis() as f64 / iterations as f64;

    println!("  总耗时: {:.2}s", elapsed.as_secs_f64());
    println!("  每次迭代: {:.2}ms", per_iter);
    println!("  吞吐量: {:.1} batches/sec", 1000.0 / per_iter);
}

fn main() {
    println!("🚀 GPU vs CPU 性能对比测试\n");
    println!("════════════════════════════════════════");

    // 测试MPS
    if std::panic::catch_unwind(|| {
        let _ = Tensor::zeros(&[1], (Kind::Float, Device::Mps));
    })
    .is_ok()
    {
        benchmark_device(Device::Mps, "Apple M2 Pro GPU (MPS)");
    }

    // 测试CPU
    benchmark_device(Device::Cpu, "CPU");

    println!("\n════════════════════════════════════════");
    println!("💡 结论:");
    println!("  GPU训练可以显著加速神经网络训练！");
    println!("  在AlphaZero训练中，GPU可以：");
    println!("    • 更快的网络前向/反向传播");
    println!("    • 支持更深的网络(10-20层)");
    println!("    • 支持更大的batch size");
    println!("    • 更快的迭代速度 → 更多训练数据");
}
