// 简单的 AlphaZero 训练程序
#![cfg(feature = "alphazero")]

use gomoku::alphazero_trainer::{AlphaZeroConfig, AlphaZeroPipeline};

fn main() {
    let args: std::vec::Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!(
            "Usage: {} <output_model_path> [num_games] [num_iterations]",
            args[0]
        );
        println!("Example: {} ../data/test_model.pt 3 50", args[0]);
        return;
    }

    let model_path = &args[1];
    let num_games = if args.len() > 2 {
        args[2].parse().unwrap_or(3)
    } else {
        3
    };
    let num_iterations = if args.len() > 3 {
        args[3].parse().unwrap_or(50)
    } else {
        50
    };

    println!("🚀 Training AlphaZero");
    println!("   Games: {}", num_games);
    println!("   Training iterations: {}", num_iterations);
    println!("   Output: {}\n", model_path);

    // 创建小规模配置用于快速训练
    let config = AlphaZeroConfig {
        num_filters: 32,
        num_res_blocks: 2,
        learning_rate: 0.001,
        batch_size: 32,
        num_self_play_games: num_games,
        num_training_iterations: num_iterations,
        replay_buffer_size: 10000,
        num_mcts_simulations: 25, // 减少MCTS模拟次数以加速训练
        temperature: 1.0,
    };

    let mut pipeline = AlphaZeroPipeline::new(config);

    // 生成自对弈数据
    pipeline.generate_self_play_data(num_games);

    // 训练
    pipeline.train(num_iterations);

    // 保存模型
    match pipeline.save_model(model_path) {
        Ok(_) => {
            println!("\n✅ Training complete! Model saved to {}", model_path);
            println!("\n⚠️  Important: To use this model with play_match, convert it first:");
            println!(
                "   python3 convert_model.py {} {}_converted.pt",
                model_path,
                model_path.trim_end_matches(".pt")
            );
        }
        Err(e) => eprintln!("\n❌ Failed to save model: {}", e),
    }
}
