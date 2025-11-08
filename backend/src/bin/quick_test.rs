// 快速测试改进版本（3轮验证）

use gomoku::az_eval::{evaluate, Player};
use gomoku::az_trainer::AlphaZeroTrainer;

fn main() {
    println!("🧪 快速验证改进效果（3轮）\n");

    let num_filters = 64;
    let learning_rate = 0.0003;
    let replay_buffer_size = 5000;
    let num_mcts_simulations = 200; // 关键改进

    let num_iterations = 3; // 只跑3轮
    let games_per_iteration = 20;
    let train_batches = 40;
    let batch_size = 64;

    println!("📋 配置:");
    println!("  ⚡ MCTS模拟: {} (关键改进)", num_mcts_simulations);
    println!("  ⚡ 网络滤波器: {}", num_filters);
    println!("  ⚡ 学习率: {}", learning_rate);
    println!();

    let mut trainer = AlphaZeroTrainer::new(
        num_filters,
        learning_rate,
        replay_buffer_size,
        num_mcts_simulations,
    );

    // 初始评估
    println!("━━━ 初始状态 ━━━");
    let (init_random, init_mcts) = quick_eval(&trainer);
    println!("vs随机: {:.0}%, vs MCTS: {:.0}%\n", init_random, init_mcts);

    // 训练3轮
    for iteration in 0..num_iterations {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📊 迭代 {}/{}", iteration + 1, num_iterations);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let temperature = if iteration == 0 { 1.0 } else { 0.5 };
        println!("🌡️  温度: {:.1}", temperature);

        trainer.generate_self_play_data(games_per_iteration, temperature);
        trainer.train(batch_size, train_batches);

        let (rate_random, rate_mcts) = quick_eval(&trainer);
        println!("vs随机: {:.0}%, vs MCTS: {:.0}%", rate_random, rate_mcts);

        // 判断趋势
        if iteration == 2 {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("📊 3轮训练效果总结");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!(
                "初始: vs随机 {:.0}%, vs MCTS {:.0}%",
                init_random, init_mcts
            );
            println!(
                "迭代3: vs随机 {:.0}%, vs MCTS {:.0}%",
                rate_random, rate_mcts
            );

            let random_improve = rate_random - init_random;
            let mcts_improve = rate_mcts - init_mcts;

            println!("\n改进幅度:");
            println!("  vs随机: {:+.0}%", random_improve);
            println!("  vs MCTS: {:+.0}%", mcts_improve);

            if rate_random > 40.0 && random_improve > 0.0 {
                println!("\n✅ 效果良好！可以进行完整训练");
                println!("运行: cargo run --features alphazero --bin train_improved");
            } else if rate_random > init_random {
                println!("\n📊 有小幅改进，但可能需要更多训练轮次");
            } else {
                println!("\n⚠️  效果不明显，可能需要调整超参数:");
                println!("  - 进一步增加MCTS模拟次数 (200->400)");
                println!("  - 降低学习率 (0.0003->0.0001)");
                println!("  - 增加每轮游戏数 (20->40)");
            }
        }
        println!();
    }
}

fn quick_eval(trainer: &AlphaZeroTrainer) -> (f32, f32) {
    let alphazero = Player::AlphaZeroRollout {
        net: &trainer.trainer.net,
        simulations: 200,
    };

    let random_player = Player::Random;
    let pure_mcts = Player::PureMCTS { simulations: 100 };

    let stats1 = evaluate(&alphazero, &random_player, 10, false);
    let stats2 = evaluate(&alphazero, &pure_mcts, 10, false);

    (
        stats1.player1_winrate() * 100.0,
        stats2.player1_winrate() * 100.0,
    )
}
