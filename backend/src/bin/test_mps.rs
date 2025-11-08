use tch::{Device, Tensor};

fn main() {
    println!("🔍 检测可用设备:");

    // 检查CUDA
    let cuda_available = Device::cuda_if_available() != Device::Cpu;
    println!("  CUDA available: {}", cuda_available);

    // 尝试MPS
    println!("\n🧪 测试 MPS (Apple GPU):");
    match std::panic::catch_unwind(|| {
        let device = Device::Mps;
        let t = Tensor::randn(&[100, 100], (tch::Kind::Float, device));
        let result = t.matmul(&t);
        println!("  矩阵运算测试: {:?}", result.size());
        println!("  ✅ MPS 工作正常！");
        true
    }) {
        Ok(true) => {
            println!("\n💡 推荐使用: Device::Mps (Apple Silicon GPU加速)");
        }
        Ok(false) | Err(_) => {
            println!("  ❌ MPS 不可用或失败");

            if cuda_available {
                println!("\n💡 推荐使用: Device::Cuda(0)");
            } else {
                println!("\n💡 只能使用: Device::Cpu");
            }
        }
    }

    // 列出Device枚举的所有变体
    println!("\n📋 Device 枚举变体:");
    println!("  - Device::Cpu");
    println!("  - Device::Cuda(0)");
    println!("  - Device::Mps (Apple Silicon)");
}
