// 简单的 AlphaZero 训练程序
#![cfg(feature = "alphazero")]

use gomoku::alphazero_trainer::{AlphaZeroConfig, AlphaZeroPipeline};

fn main() {
    let args: std::vec::Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!(
            "Usage: {} <output_model_path> [num_games_per_iter] [train_iters] [num_epochs]",
            args[0]
        );
        println!("Example: {} ../data/model.pt 100 500 10", args[0]);
        println!();
        println!("Arguments:");
        println!("  output_model_path    - 模型保存路径");
        println!("  num_games_per_iter   - 每轮自我对弈游戏数 (default: 100)");
        println!("  train_iters          - 每轮训练迭代数 (default: 500)");
        println!("  num_epochs           - 迭代训练轮数 (default: 10)");
        return;
    }

    let model_path = &args[1];
    let num_games = if args.len() > 2 {
        args[2].parse().unwrap_or(100)
    } else {
        100
    };
    let num_iterations = if args.len() > 3 {
        args[3].parse().unwrap_or(500)
    } else {
        500
    };
    let num_epochs = if args.len() > 4 {
        args[4].parse().unwrap_or(10)
    } else {
        10
    };

    println!("🚀 AlphaZero Iterative Training");
    println!("   Games per epoch: {}", num_games);
    println!("   Training iterations per epoch: {}", num_iterations);
    println!("   Number of epochs: {}", num_epochs);
    println!("   Output: {}\n", model_path);

    // 创建配置
    let config = AlphaZeroConfig {
        // Use stronger defaults more suitable for Connect4 training
        num_filters: 128,
        num_res_blocks: 6,
        learning_rate: 0.001,
        batch_size: 64,
        num_self_play_games: num_games,
        num_training_iterations: num_iterations,
        replay_buffer_size: 100000, // 增大缓冲区
        num_mcts_simulations: 200,
        temperature: 1.0,
    };

    let mut pipeline = AlphaZeroPipeline::new(config);

    // 如果模型文件已存在，加载它（用于继续训练）
    if std::path::Path::new(model_path).exists() {
        println!("📂 Loading existing model from {}...", model_path);
        match pipeline.load_model(model_path) {
            Ok(_) => println!("✅ Model loaded successfully! Continuing training...\n"),
            Err(e) => {
                eprintln!(
                    "⚠️  Warning: Failed to load model ({}). Starting fresh training...\n",
                    e
                );
            }
        }
    } else {
        println!("📝 No existing model found. Starting fresh training...\n");
    }

    // 使用改进的迭代训练循环
    pipeline.train_loop(num_epochs);

    // 保存最终模型
    match pipeline.save_model(model_path) {
        Ok(_) => {
            println!("\n✅ Training complete! Model saved to {}", model_path);
            println!("\n💡 Next steps:");
            println!(
                "   1. Convert model: python3 convert_model.py {} {}_converted.pt",
                model_path,
                model_path.trim_end_matches(".pt")
            );
            println!(
                "   2. Test model: ./play_match.sh {}_converted.pt 10 500",
                model_path.trim_end_matches(".pt")
            );
        }
        Err(e) => eprintln!("\n❌ Failed to save model: {}", e),
    }
}
