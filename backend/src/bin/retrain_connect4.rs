// 重新训练 Connect4 AlphaZero - 优化版本

use gomoku::az_eval::{evaluate, Player};
use gomoku::az_trainer::AlphaZeroTrainer;
use std::convert::TryFrom;

fn main() {
    println!("🚀 AlphaZero Connect4 重新训练\n");
    println!("目标：修复policy bias问题，训练一个真正有用的模型\n");

    // 优化的超参数
    let num_filters = 128;       // 使用128滤波器（与之前一致）
    let learning_rate = 0.002;   // 稍微提高学习率
    let replay_buffer_size = 5000; // 增大回放缓冲区
    let num_mcts_simulations = 200; // 增加MCTS模拟次数以获得更好的训练数据

    let num_iterations = 30;     // 减少迭代次数，但每次质量更高
    let games_per_iteration = 30; // 每次迭代生成更多数据
    let train_batches = 100;     // 增加训练批次
    let batch_size = 64;         // 批次大小
    let temperature = 1.0;       // 温度参数

    println!("📋 训练配置:");
    println!("  网络: ResNet-10 with {} filters", num_filters);
    println!("  MCTS模拟次数: {}", num_mcts_simulations);
    println!("  学习率: {}", learning_rate);
    println!("  回放缓冲区: {}", replay_buffer_size);
    println!("  训练迭代: {}", num_iterations);
    println!("  每轮自对弈: {} 局", games_per_iteration);
    println!("  每轮训练批次: {}", train_batches);
    println!("  批次大小: {}\n", batch_size);

    let mut trainer = AlphaZeroTrainer::new(
        num_filters,
        learning_rate,
        replay_buffer_size,
        num_mcts_simulations,
    );

    // 评估初始模型
    println!("╔════════════════════════════════════════════╗");
    println!("║  初始评估（随机初始化的网络）              ║");
    println!("╚════════════════════════════════════════════╝");
    let initial_win_rate = quick_evaluate(&trainer, "初始");

    let mut best_win_rate = initial_win_rate;
    let mut best_model_iteration = 0;

    // 训练循环
    for iteration in 0..num_iterations {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📊 迭代 {}/{}", iteration + 1, num_iterations);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // 1. 生成自对弈数据
        println!("\n🎮 阶段1: 生成自对弈数据");
        trainer.generate_self_play_data(games_per_iteration, temperature);

        // 2. 训练网络
        println!("\n🧠 阶段2: 训练神经网络");
        trainer.train(batch_size, train_batches);

        // 3. 保存检查点
        let checkpoint_path = format!("connect4_resnet_iter_{}.pt", iteration + 1);
        if let Err(e) = trainer.save_model(&checkpoint_path) {
            eprintln!("⚠️  保存模型失败: {}", e);
        } else {
            println!("💾 模型已保存到: {}", checkpoint_path);
        }

        // 4. 每5次迭代进行详细评估
        if (iteration + 1) % 5 == 0 {
            println!("\n╔════════════════════════════════════════════╗");
            println!("║  详细评估 - 迭代 {}                        ║", iteration + 1);
            println!("╚════════════════════════════════════════════╝");
            
            let win_rate = detailed_evaluate(&trainer, iteration + 1);
            
            // 保存最佳模型
            if win_rate > best_win_rate {
                best_win_rate = win_rate;
                best_model_iteration = iteration + 1;
                let best_path = "connect4_resnet_best.pt";
                if let Ok(_) = trainer.save_model(best_path) {
                    println!("🏆 保存最佳模型到: {} (胜率: {:.1}%)", best_path, best_win_rate * 100.0);
                }
            }
        } else {
            // 快速评估
            quick_evaluate(&trainer, &format!("迭代{}", iteration + 1));
        }
    }

    // 最终评估
    println!("\n╔════════════════════════════════════════════╗");
    println!("║  最终评估                                  ║");
    println!("╚════════════════════════════════════════════╝");
    detailed_evaluate(&trainer, num_iterations);

    // 保存最终模型
    let final_path = "connect4_resnet_final.pt";
    if let Ok(_) = trainer.save_model(final_path) {
        println!("\n💾 最终模型已保存到: {}", final_path);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎉 训练完成！");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("最佳模型: 迭代 {} (胜率 vs 随机: {:.1}%)", best_model_iteration, best_win_rate * 100.0);
    println!("模型文件: connect4_resnet_best.pt");
}

/// 快速评估（少量对局）
fn quick_evaluate(trainer: &AlphaZeroTrainer, label: &str) -> f32 {
    let alphazero = Player::AlphaZero {
        net: &trainer.trainer.net,
        simulations: 50,
    };

    let random_player = Player::Random;

    print!("\n📊 {} vs 随机玩家 (5局快速测试)...", label);
    let stats = evaluate(&alphazero, &random_player, 5, false);
    let win_rate = stats.player1_winrate();
    println!(" 胜率: {:.1}%", win_rate * 100.0);
    
    win_rate
}

/// 详细评估（更多对局）
fn detailed_evaluate(trainer: &AlphaZeroTrainer, iteration: usize) -> f32 {
    let alphazero = Player::AlphaZero {
        net: &trainer.trainer.net,
        simulations: 50,
    };

    let random_player = Player::Random;
    let pure_mcts = Player::PureMCTS { simulations: 50 };

    println!("\n📊 迭代{} vs 随机玩家 (20局)", iteration);
    let stats1 = evaluate(&alphazero, &random_player, 20, false);
    let win_rate_random = stats1.player1_winrate();
    println!("  胜率: {:.1}%", win_rate_random * 100.0);
    println!("  详情: {} 胜, {} 负, {} 平", 
             stats1.player1_wins, stats1.player2_wins, stats1.draws);

    println!("\n📊 迭代{} vs 纯MCTS(50模拟) (20局)", iteration);
    let stats2 = evaluate(&alphazero, &pure_mcts, 20, false);
    let win_rate_mcts = stats2.player1_winrate();
    println!("  胜率: {:.1}%", win_rate_mcts * 100.0);
    println!("  详情: {} 胜, {} 负, {} 平", 
             stats2.player1_wins, stats2.player2_wins, stats2.draws);

    // 检查policy bias
    check_policy_bias(trainer);

    win_rate_random
}

/// 检查policy偏差
fn check_policy_bias(trainer: &AlphaZeroTrainer) {
    use gomoku::connect4::Connect4;
    use tch::Tensor;

    println!("\n🔍 检查空棋盘的policy输出:");
    
    let game = Connect4::new();
    let board_tensor = game.to_tensor();
    
    // 转换为Tensor [1, 3, 6, 7]
    let device = trainer.trainer.net.device();
    let tensor = Tensor::from_slice(&board_tensor)
        .view([1, 3, 6, 7])
        .to(device);
    
    let (policy_logits, value) = tch::no_grad(|| {
        trainer.trainer.net.predict(&tensor)
    });
    
    let policy_probs = policy_logits.softmax(-1, tch::Kind::Float);
    let probs: Vec<f32> = Vec::try_from(policy_probs.squeeze()).unwrap();
    let value_f: f32 = value.double_value(&[]) as f32;
    
    println!("  Value预测: {:.4}", value_f);
    println!("  Policy概率分布:");
    for (i, prob) in probs.iter().enumerate() {
        let bar = "█".repeat((prob * 50.0) as usize);
        println!("    Col {}: {:.4} {}", i, prob, bar);
    }
    
    // 检查是否有明显偏差
    let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
    let avg_prob = probs.iter().sum::<f32>() / probs.len() as f32;
    
    if max_prob > 0.3 {
        println!("  ⚠️  检测到偏差：column {} 的概率过高 ({:.1}%)", 
                 probs.iter().position(|&p| p == max_prob).unwrap(), 
                 max_prob * 100.0);
    } else if (max_prob - avg_prob).abs() < 0.05 {
        println!("  ✅ Policy分布相对均匀（好现象）");
    }
}
