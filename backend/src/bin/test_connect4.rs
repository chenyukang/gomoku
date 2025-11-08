// AlphaZero Connect4 快速测试版本

use gomoku::az_trainer::AlphaZeroTrainer;

fn main() {
    println!("🚀 AlphaZero Connect4 快速测试！\n");

    // 较小的超参数用于快速测试
    let num_filters = 32; // 减少滤波器
    let learning_rate = 0.001;
    let replay_buffer_size = 1000; // 减少缓冲区
    let num_mcts_simulations = 50; // 减少MCTS模拟

    let num_iterations = 3; // 只运行3轮
    let games_per_iteration = 10; // 每轮10局
    let train_batches = 20; // 每轮20批次训练
    let batch_size = 32; // 更小的批次
    let temperature = 1.0;

    println!("📋 测试配置:");
    println!("  神经网络滤波器: {}", num_filters);
    println!("  MCTS模拟次数: {}", num_mcts_simulations);
    println!("  训练迭代: {}", num_iterations);
    println!("  每轮自对弈: {} 局", games_per_iteration);
    println!("  每轮训练: {} 批次", train_batches);
    println!();

    let mut trainer = AlphaZeroTrainer::new(
        num_filters,
        learning_rate,
        replay_buffer_size,
        num_mcts_simulations,
    );

    for iteration in 0..num_iterations {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📊 迭代 {}/{}", iteration + 1, num_iterations);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        trainer.generate_self_play_data(games_per_iteration, temperature);
        trainer.train(batch_size, train_batches);

        let checkpoint_path = format!("test_model_iter_{}.pt", iteration + 1);
        if let Err(e) = trainer.save_model(&checkpoint_path) {
            eprintln!("⚠️  保存模型失败: {}", e);
        } else {
            println!("💾 模型已保存到: {}", checkpoint_path);
        }
        println!();
    }

    println!("🎉 测试完成！");
}
