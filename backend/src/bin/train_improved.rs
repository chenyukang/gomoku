// 改进的训练程序 - 解决训练不收敛问题

use gomoku::az_eval::{evaluate, Player};
use gomoku::az_trainer::AlphaZeroTrainer;

fn main() {
    println!("🚀 AlphaZero Connect4 改进训练\n");

    // 关键改进：
    // 1. 增加MCTS模拟次数（50 -> 200）
    // 2. 使用更大的网络（32 -> 64 filters）
    // 3. 降低学习率避免震荡
    // 4. 增加每轮游戏数量

    let num_filters = 64; // 更大的网络容量
    let learning_rate = 0.0003; // 更小的学习率（原0.001）
    let replay_buffer_size = 10000; // 🎯 2倍buffer - 保留更多历史数据！
    let num_mcts_simulations = 400; // 🎯 适中的模拟次数（不要太高导致过慢）

    let num_iterations = 50; // 🎯 更多迭代机会（有早停保护）
    let games_per_iteration = 50; // 🎯 更多游戏 = 更多样数据
    let train_batches = 30; // 🎯 降低！防止过拟合（从100降到30）
    let batch_size = 64; // 🎯 降回64（更稳定）

    println!("📋 防过拟合优化配置:");
    println!("  🎯 Buffer: {} (2倍↑ 保留更多历史)", replay_buffer_size);
    println!("  🎯 每轮游戏: {}局 (↑ 提升多样性)", games_per_iteration);
    println!("  🎯 训练批次: {} (↓ 防止过拟合)", train_batches);
    println!("  🎯 MCTS模拟: {}", num_mcts_simulations);
    println!("\n  关键策略: 更大buffer + 更高温度 + 更少训练 = 防止过拟合");
    println!("           早停机制 = 检测到胜率下降立即停止\n");

    let mut trainer = AlphaZeroTrainer::new(
        num_filters,
        learning_rate,
        replay_buffer_size,
        num_mcts_simulations,
    );

    // 初始评估
    println!("╔══════════════════════════════════════╗");
    println!("║  初始评估（随机初始化）              ║");
    println!("╚══════════════════════════════════════╝");

    let mut best_random_winrate = evaluate_model(&trainer, "初始");
    let mut no_improve_count = 0;
    let mut best_iteration = 0;

    // 训练循环
    for iteration in 0..num_iterations {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📊 迭代 {}/{}", iteration + 1, num_iterations);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // 🎯 关键改进：保持更高温度，避免数据单一化
        let temperature = if iteration < 10 {
            1.0 // 前10轮高探索（之前只有5轮）
        } else if iteration < 30 {
            0.7 // 🔥 中期保持较高（之前0.5）
        } else {
            0.5 // 🔥 后期也保持探索（之前0.1太低）
        };

        println!("🌡️  当前温度: {:.1} (保持探索性)", temperature);

        trainer.generate_self_play_data(games_per_iteration, temperature);
        trainer.train(batch_size, train_batches);

        // 每3轮评估一次
        if (iteration + 1) % 3 == 0 || iteration == 0 {
            println!("\n╔══════════════════════════════════════╗");
            println!("║  评估模型（迭代 {}）                  ║", iteration + 1);
            println!("╚══════════════════════════════════════╝");

            let current_winrate = evaluate_model(&trainer, &format!("迭代{}", iteration + 1));

            // 🎯 早停机制：检测性能下降
            if current_winrate > best_random_winrate {
                best_random_winrate = current_winrate;
                best_iteration = iteration + 1;
                no_improve_count = 0;

                // 保存最佳模型
                let filename = "connect4_best.pt";
                trainer.save_model(filename).ok();
                println!("\n  🏆 新最佳模型！胜率: {:.1}%", current_winrate * 100.0);
                println!("  💾 保存为: {}", filename);
            } else {
                no_improve_count += 1;
                println!("\n  📉 未改进 ({}/3次)", no_improve_count);
                println!(
                    "  当前: {:.1}% vs 最佳: {:.1}%",
                    current_winrate * 100.0,
                    best_random_winrate * 100.0
                );

                // 连续3次评估都没提升 = 过拟合
                if no_improve_count >= 3 {
                    println!("\n  🛑 检测到过拟合趋势，提前停止训练！");
                    println!(
                        "  💡 最佳模型在迭代{}，胜率{:.1}%",
                        best_iteration,
                        best_random_winrate * 100.0
                    );
                    break;
                }
            }

            // 定期保存检查点
            if (iteration + 1) % 6 == 0 {
                let filename = format!("connect4_model_iter_{}.pt", iteration + 1);
                trainer.save_model(&filename).ok();
                println!("  💾 检查点: {}", filename);
            }
        }
    }

    println!("\n🎉 训练完成！");

    // 最终完整评估
    println!("\n╔══════════════════════════════════════╗");
    println!("║  最终评估（完整版）                  ║");
    println!("╚══════════════════════════════════════╝");
    final_evaluation(&trainer);
}

fn evaluate_model(trainer: &AlphaZeroTrainer, label: &str) -> f64 {
    // 评估时使用rollout版本（因为网络还在训练中）
    let alphazero = Player::AlphaZeroRollout {
        net: &trainer.trainer.net,
        simulations: 200,
    };

    let random_player = Player::Random;
    let pure_mcts = Player::PureMCTS { simulations: 100 }; // MCTS也加强

    println!("\n📊 {} vs 随机玩家 (20局)", label);
    let stats1 = evaluate(&alphazero, &random_player, 20, false);
    let win_rate_random = stats1.player1_winrate() * 100.0;
    println!("  胜率: {:.1}%", win_rate_random);

    println!("\n📊 {} vs 纯MCTS (20局)", label);
    let stats2 = evaluate(&alphazero, &pure_mcts, 20, false);
    let win_rate_mcts = stats2.player1_winrate() * 100.0;
    println!("  胜率: {:.1}%", win_rate_mcts);

    println!(
        "\n📈 综合评分: 随机{:.0}%  MCTS{:.0}%",
        win_rate_random, win_rate_mcts
    );

    // 棋力等级判断
    if win_rate_random >= 90.0 && win_rate_mcts >= 50.0 {
        println!("🏆 棋力等级: 优秀");
    } else if win_rate_random >= 70.0 && win_rate_mcts >= 30.0 {
        println!("✅ 棋力等级: 良好");
    } else if win_rate_random >= 50.0 {
        println!("📊 棋力等级: 及格");
    } else {
        println!("⚠️  棋力等级: 需要改进");
    }

    // 返回vs随机的胜率作为主要指标
    stats1.player1_winrate() as f64
}

fn final_evaluation(trainer: &AlphaZeroTrainer) {
    let alphazero = Player::AlphaZeroRollout {
        net: &trainer.trainer.net,
        simulations: 200,
    };

    let random_player = Player::Random;
    let pure_mcts_weak = Player::PureMCTS { simulations: 50 };
    let pure_mcts_strong = Player::PureMCTS { simulations: 100 };

    println!("\n📊 vs 随机玩家 (50局)");
    let stats1 = evaluate(&alphazero, &random_player, 50, false);
    println!("  胜率: {:.1}%", stats1.player1_winrate() * 100.0);

    println!("\n📊 vs 纯MCTS(50模拟) (30局)");
    let stats2 = evaluate(&alphazero, &pure_mcts_weak, 30, false);
    println!("  胜率: {:.1}%", stats2.player1_winrate() * 100.0);

    println!("\n📊 vs 纯MCTS(100模拟) (30局)");
    let stats3 = evaluate(&alphazero, &pure_mcts_strong, 30, false);
    println!("  胜率: {:.1}%", stats3.player1_winrate() * 100.0);
}
