// AlphaZero 训练 CLI
#![cfg(feature = "alphazero")]

use gomoku::alphazero_solver::AlphaZeroSolver;
use gomoku::alphazero_trainer::{AlphaZeroConfig, AlphaZeroPipeline};
use gomoku::board::Board;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "train" => train_alphazero(&args[2..]),
        "test" => test_model(&args[2..]),
        "benchmark" => benchmark_model(&args[2..]),
        _ => {
            println!("Unknown command: {}", args[1]);
            print_usage();
        }
    }
}

fn print_usage() {
    println!("AlphaZero Training CLI\n");
    println!("Usage:");
    println!("  train [OPTIONS]              - Train AlphaZero model");
    println!("  test MODEL_PATH              - Test trained model");
    println!("  benchmark MODEL_PATH         - Benchmark against other algorithms\n");
    println!("Train Options:");
    println!("  --filters N                  - Number of filters (default: 128)");
    println!("  --blocks N                   - Number of residual blocks (default: 10)");
    println!("  --lr RATE                    - Learning rate (default: 0.001)");
    println!("  --batch-size N               - Batch size (default: 32)");
    println!("  --buffer-size N              - Replay buffer size (default: 10000)");
    println!("  --games N                    - Self-play games per iteration (default: 100)");
    println!("  --iterations N               - Training iterations (default: 1000)");
    println!("  --simulations N              - MCTS simulations (default: 400)");
    println!("  --epochs N                   - Number of training epochs (default: 10)");
    println!(
        "  --output PATH                - Output model path (default: data/alphazero_final.pt)\n"
    );
    println!("Examples:");
    println!("  cargo run --bin alphazero_cli --features alphazero -- train --epochs 5");
    println!(
        "  cargo run --bin alphazero_cli --features alphazero -- test data/alphazero_final.pt"
    );
}

fn train_alphazero(args: &[String]) {
    let mut config = AlphaZeroConfig::default();
    let mut output_path = "data/alphazero_final.pt".to_string();
    let mut num_epochs = 10;

    // 解析参数
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--filters" => {
                if i + 1 < args.len() {
                    config.num_filters = args[i + 1].parse().unwrap_or(128);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--blocks" => {
                if i + 1 < args.len() {
                    config.num_res_blocks = args[i + 1].parse().unwrap_or(10);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    config.learning_rate = args[i + 1].parse().unwrap_or(0.001);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--batch-size" => {
                if i + 1 < args.len() {
                    config.batch_size = args[i + 1].parse().unwrap_or(32);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--buffer-size" => {
                if i + 1 < args.len() {
                    config.replay_buffer_size = args[i + 1].parse().unwrap_or(10000);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--games" => {
                if i + 1 < args.len() {
                    config.num_self_play_games = args[i + 1].parse().unwrap_or(100);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--iterations" => {
                if i + 1 < args.len() {
                    config.num_training_iterations = args[i + 1].parse().unwrap_or(1000);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--simulations" => {
                if i + 1 < args.len() {
                    config.num_mcts_simulations = args[i + 1].parse().unwrap_or(400);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--epochs" => {
                if i + 1 < args.len() {
                    num_epochs = args[i + 1].parse().unwrap_or(10);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    println!("🚀 AlphaZero Training Configuration:");
    println!("  Filters: {}", config.num_filters);
    println!("  Residual Blocks: {}", config.num_res_blocks);
    println!("  Learning Rate: {}", config.learning_rate);
    println!("  Batch Size: {}", config.batch_size);
    println!("  Replay Buffer: {}", config.replay_buffer_size);
    println!("  Self-Play Games: {}", config.num_self_play_games);
    println!("  Training Iterations: {}", config.num_training_iterations);
    println!("  MCTS Simulations: {}", config.num_mcts_simulations);
    println!("  Training Epochs: {}", num_epochs);
    println!("  Output: {}\n", output_path);

    // 创建训练管道
    let mut pipeline = AlphaZeroPipeline::new(config);

    // 开始训练
    let start_time = Instant::now();
    pipeline.train_loop(num_epochs);
    let elapsed = start_time.elapsed();

    // 保存最终模型
    if let Err(e) = pipeline.save_model(&output_path) {
        eprintln!("❌ Failed to save model: {}", e);
    } else {
        println!("\n✅ Model saved to: {}", output_path);
    }

    println!("\n⏱️  Total training time: {:.2}s", elapsed.as_secs_f64());
}

fn test_model(args: &[String]) {
    if args.is_empty() {
        println!("❌ Error: Please specify model path");
        return;
    }

    let model_path = &args[0];
    println!("🧪 Testing model: {}", model_path);

    // 加载模型
    let solver = match AlphaZeroSolver::from_file(model_path, 128, 10, 400) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            return;
        }
    };

    // 创建测试棋盘
    let mut board = Board::new_default();
    board.place(7, 7, 1);
    board.place(7, 8, 2);
    board.place(8, 7, 1);

    println!("\nTest Board:");
    println!("{}", board.to_string());

    // 测试推理
    let start = Instant::now();
    if let Some((x, y)) = solver.solve(&board, 2) {
        let elapsed = start.elapsed();
        println!("\n✅ AlphaZero suggests: ({}, {})", x, y);
        println!("⏱️  Inference time: {:.3}s", elapsed.as_secs_f64());
    } else {
        println!("❌ No move found");
    }
}

fn benchmark_model(args: &[String]) {
    if args.is_empty() {
        println!("❌ Error: Please specify model path");
        return;
    }

    let model_path = &args[0];
    println!("📊 Benchmarking model: {}\n", model_path);

    // 加载 AlphaZero 模型
    let alphazero = match AlphaZeroSolver::from_file(model_path, 128, 10, 100) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            return;
        }
    };

    println!("🎮 AlphaZero vs Random (10 games)");
    let az_wins = play_vs_random(&alphazero, 10);
    println!("Results: AlphaZero wins {}/10", az_wins);
}

fn play_vs_random(alphazero: &AlphaZeroSolver, num_games: usize) -> usize {
    let mut wins = 0;

    for i in 0..num_games {
        print!("\rGame {}/{}...", i + 1, num_games);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let mut board = Board::new_default();
        let mut current_player = 1u8;

        for _ in 0..300 {
            let next_move = if current_player == 1 {
                // AlphaZero
                alphazero.solve(&board, current_player)
            } else {
                // Random
                let moves = board.gen_ordered_moves(current_player);
                if !moves.is_empty() {
                    let mv = &moves[0];
                    Some((mv.x as i32, mv.y as i32))
                } else {
                    None
                }
            };

            if let Some((x, y)) = next_move {
                board.place(x as usize, y as usize, current_player);

                if let Some(winner) = board.any_winner() {
                    if winner == 1 {
                        wins += 1;
                    }
                    break;
                }

                current_player = if current_player == 1 { 2 } else { 1 };
            } else {
                break;
            }
        }
    }

    println!();
    wins
}
