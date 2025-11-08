// AlphaZero Connect4 训练主程序

use gomoku::az_trainer::AlphaZeroTrainer;

fn main() {
    println!("🚀 AlphaZero Connect4 训练开始！\n");

    // 超参数
    let num_filters = 64; // 卷积滤波器数量
    let learning_rate = 0.001; // 学习率
    let replay_buffer_size = 10000; // 回放缓冲区大小
    let num_mcts_simulations = 100; // MCTS模拟次数

    let num_iterations = 10; // 训练迭代次数
    let games_per_iteration = 50; // 每次迭代自对弈局数
    let train_batches = 50; // 每次迭代训练批次数
    let batch_size = 64; // 训练批次大小
    let temperature = 1.0; // MCTS温度参数

    println!("📋 训练配置:");
    println!("  神经网络滤波器: {}", num_filters);
    println!("  学习率: {}", learning_rate);
    println!("  MCTS模拟次数: {}", num_mcts_simulations);
    println!("  回放缓冲区: {}", replay_buffer_size);
    println!("  训练迭代: {}", num_iterations);
    println!("  每轮自对弈: {} 局", games_per_iteration);
    println!("  每轮训练: {} 批次", train_batches);
    println!("  批次大小: {}", batch_size);
    println!();

    // 创建训练器
    let mut trainer = AlphaZeroTrainer::new(
        num_filters,
        learning_rate,
        replay_buffer_size,
        num_mcts_simulations,
    );

    // 训练循环
    for iteration in 0..num_iterations {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📊 迭代 {}/{}", iteration + 1, num_iterations);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // 1. 生成自对弈数据
        trainer.generate_self_play_data(games_per_iteration, temperature);

        // 2. 训练网络
        trainer.train(batch_size, train_batches);

        // 3. 保存检查点
        let checkpoint_path = format!("connect4_model_iter_{}.pt", iteration + 1);
        if let Err(e) = trainer.save_model(&checkpoint_path) {
            eprintln!("⚠️  保存模型失败: {}", e);
        } else {
            println!("💾 模型已保存到: {}", checkpoint_path);
        }

        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎉 训练完成！");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 保存最终模型
    if let Err(e) = trainer.save_model("connect4_model_final.pt") {
        eprintln!("⚠️  保存最终模型失败: {}", e);
    } else {
        println!("💾 最终模型已保存到: connect4_model_final.pt");
    }
}
