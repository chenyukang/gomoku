// 带评估功能的训练程序

use gomoku::az_eval::{evaluate, Player};
use gomoku::az_trainer::AlphaZeroTrainer;

fn main() {
    println!("🚀 AlphaZero Connect4 训练+评估\n");

    let num_filters = 32;
    let learning_rate = 0.001;
    let replay_buffer_size = 2000;
    let num_mcts_simulations = 50;

    let num_iterations = 50;
    let games_per_iteration = 20;
    let train_batches = 30;
    let batch_size = 32;
    let temperature = 1.0;

    println!("📋 配置:");
    println!(
        "  滤波器: {}, MCTS模拟: {}",
        num_filters, num_mcts_simulations
    );
    println!(
        "  迭代: {}, 每轮自对弈: {}局\n",
        num_iterations, games_per_iteration
    );

    let mut trainer = AlphaZeroTrainer::new(
        num_filters,
        learning_rate,
        replay_buffer_size,
        num_mcts_simulations,
    );

    // 评估初始模型（未训练）
    println!("╔══════════════════════════════════════╗");
    println!("║  初始评估（随机初始化）              ║");
    println!("╚══════════════════════════════════════╝");
    evaluate_model(&trainer, "初始");

    // 训练循环
    for iteration in 0..num_iterations {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📊 迭代 {}/{}", iteration + 1, num_iterations);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        trainer.generate_self_play_data(games_per_iteration, temperature);
        trainer.train(batch_size, train_batches);

        // 每次迭代后评估
        println!("\n╔══════════════════════════════════════╗");
        println!("║  评估模型（迭代 {}）                  ║", iteration + 1);
        println!("╚══════════════════════════════════════╝");
        evaluate_model(&trainer, &format!("迭代{}", iteration + 1));
    }

    println!("\n🎉 训练完成！");
}

fn evaluate_model(trainer: &AlphaZeroTrainer, label: &str) {
    let alphazero = Player::AlphaZero {
        net: &trainer.trainer.net,
        simulations: 50,
    };

    let random_player = Player::Random;
    let pure_mcts = Player::PureMCTS { simulations: 50 };

    println!("\n📊 {} vs 随机玩家 (10局)", label);
    let stats1 = evaluate(&alphazero, &random_player, 10, false);
    let win_rate_random = stats1.player1_winrate() * 100.0;
    println!("  胜率: {:.1}%", win_rate_random);

    println!("\n📊 {} vs 纯MCTS (10局)", label);
    let stats2 = evaluate(&alphazero, &pure_mcts, 10, false);
    let win_rate_mcts = stats2.player1_winrate() * 100.0;
    println!("  胜率: {:.1}%", win_rate_mcts);

    println!(
        "\n📈 综合评分: 随机{:.0}%  MCTS{:.0}%",
        win_rate_random, win_rate_mcts
    );
}
