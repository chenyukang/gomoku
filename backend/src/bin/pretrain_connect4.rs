use gomoku::connect4::Connect4;
use gomoku::az_mcts_rollout::MCTSWithRollout;
use gomoku::az_trainer::{AlphaZeroTrainer, TrainingSample};
use gomoku::az_net::Connect4Net;
use std::convert::TryFrom;
use tch::{nn, Device};

fn main() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎓 Connect4 监督学习预训练");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 配置参数
    let num_filters = 128;
    let mcts_simulations = 200; // MCTS模拟次数 (降低以加快速度)
    let num_games = 100;        // 生成100局游戏数据 (降低以加快速度)
    let batch_size = 64;
    let train_epochs = 20;      // 训练20轮 (降低以加快测试)
    let learning_rate = 0.01;   // 监督学习可以用更大的学习率

    println!("📋 配置:");
    println!("  网络: {} filters", num_filters);
    println!("  数据生成: {} 局游戏，MCTS {} 模拟（使用rollout）", num_games, mcts_simulations);
    println!("  训练: {} epochs, batch_size={}, lr={}", train_epochs, batch_size, learning_rate);
    println!();

    // 1. 生成训练数据
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 阶段1: 使用MCTS+Rollout生成训练数据");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let training_data = generate_expert_data(num_games, mcts_simulations);
    
    println!("\n✅ 数据生成完成:");
    println!("  总样本数: {}", training_data.len());
    println!("  平均每局: {:.1} 步", training_data.len() as f32 / num_games as f32);

    // 2. 创建并训练网络
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🧠 阶段2: 监督学习训练神经网络");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut trainer = AlphaZeroTrainer::new(
        num_filters,
        learning_rate,
        5000, // replay_buffer_size
        mcts_simulations as u32,
    );

    // 将数据加入训练器
    println!("📥 加载训练数据到replay buffer...");
    for sample in training_data {
        trainer.add_sample(sample);
    }

    // 监督学习训练
    println!("🎯 开始监督学习训练...\n");
    
    for epoch in 0..train_epochs {
        let num_batches = trainer.replay_buffer_size().max(batch_size) / batch_size;
        
        // 训练一个epoch
        trainer.train(batch_size, num_batches);
        
        // 每5个epoch评估一次
        if (epoch + 1) % 5 == 0 {
            println!("\n📊 Epoch {}/{} 评估:", epoch + 1, train_epochs);
            
            // 快速评估
            let win_rate = evaluate_model(&trainer, 20);
            println!("  vs 随机: {:.1}%", win_rate * 100.0);
            
            // 检查policy偏差
            check_policy_bias(&trainer);
            
            // 保存检查点
            let checkpoint_path = format!("connect4_pretrain_epoch_{}.pt", epoch + 1);
            if let Err(e) = trainer.save_model(&checkpoint_path) {
                eprintln!("  ⚠️  保存失败: {}", e);
            } else {
                println!("  💾 已保存: {}", checkpoint_path);
            }
            println!();
        }
    }

    // 3. 最终评估
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 最终评估");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("测试 vs 随机玩家 (100局):");
    let final_win_rate = evaluate_model(&trainer, 100);
    println!("  胜率: {:.1}%\n", final_win_rate * 100.0);

    // 检查policy分布
    check_policy_bias(&trainer);

    // 保存最终模型
    let final_path = "connect4_pretrained.pt";
    if let Err(e) = trainer.save_model(final_path) {
        eprintln!("⚠️  保存最终模型失败: {}", e);
    } else {
        println!("\n🎉 预训练完成！模型已保存到: {}", final_path);
        println!("\n💡 下一步: 使用此模型作为起点进行强化学习");
        println!("   1. 转换模型: python3 convert_model_v2.py {} connect4_pretrained_converted.pt", final_path);
        println!("   2. 测试模型: 复制到client目录并在网页中测试");
    }
}

/// 使用MCTS+Rollout生成专家级训练数据（不依赖神经网络）
fn generate_expert_data(num_games: usize, mcts_sims: usize) -> Vec<TrainingSample> {
    let mut all_samples = Vec::new();
    
    // 创建一个dummy网络（因为MCTSWithRollout signature需要，但use_rollout=true时不会真正使用）
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let dummy_net = Connect4Net::new(&vs.root(), 64); // 小网络，不会被用到
    
    for game_idx in 0..num_games {
        if (game_idx + 1) % 20 == 0 {
            println!("  完成 {}/{} 局", game_idx + 1, num_games);
        }
        
        let mut game = Connect4::new();
        let mut history = Vec::new();
        
        // 使用MCTS+Rollout自对弈（use_rollout=true表示用rollout，不用网络）
        while !game.is_game_over() {
            let mut mcts = MCTSWithRollout::new(mcts_sims as u32, true); // true = 使用rollout
            
            // 执行MCTS搜索（虽然传入网络，但use_rollout=true时不会使用）
            let policy = mcts.search(&game, &dummy_net);
            
            // 保存状态和MCTS的policy
            history.push((game.to_tensor(), policy.clone(), game.current_player()));
            
            // 选择最佳动作
            let action = mcts.select_action(0.5); // 适度的温度保持探索
            
            game.play(action).expect("非法动作");
        }
        
        // 根据游戏结果创建训练样本
        let winner = game.winner();
        for (board, policy, player) in history {
            let outcome = match winner {
                Some(0) => 0.0,
                Some(w) if w == player => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };
            
            all_samples.push(TrainingSample {
                board,
                policy,
                outcome,
            });
        }
    }
    
    all_samples
}

/// 评估模型对随机玩家的胜率
fn evaluate_model(trainer: &AlphaZeroTrainer, num_games: usize) -> f32 {
    let mut wins = 0;
    
    // 注意：由于MPS的float64限制，评估时简单地测试模型
    // 实际上我们只需要快速检查模型是否在学习
    for _ in 0..num_games {
        let mut game = Connect4::new();
        
        while !game.is_game_over() {
            if game.current_player() == 1 {
                // AI回合：使用模型预测最佳动作（不用MCTS，避免MPS问题）
                let (policy, _) = trainer.predict(&game);
                
                // 直接从policy中选择概率最高的动作
                // 硬编码size=7避免调用policy.size()触发MPS转换
                let legal_moves = game.legal_moves();
                let policy_vec: Vec<f32> = unsafe {
                    let data_ptr = policy.data_ptr() as *const f32;
                    std::slice::from_raw_parts(data_ptr, 7).to_vec()  // Connect4固定7列
                };
                
                let mut best_action = legal_moves[0];
                let mut best_prob = -1.0f32;
                for &action in &legal_moves {
                    if policy_vec[action] > best_prob {
                        best_prob = policy_vec[action];
                        best_action = action;
                    }
                }
                
                game.play(best_action).ok();
            } else {
                // 随机玩家
                let valid_moves = game.legal_moves();
                if !valid_moves.is_empty() {
                    use rand::Rng;
                    let random_move = valid_moves[rand::thread_rng().gen_range(0..valid_moves.len())];
                    game.play(random_move).ok();
                }
            }
        }
        
        if game.winner() == Some(1) {
            wins += 1;
        }
    }
    
    wins as f32 / num_games as f32
}

/// 检查空棋盘的policy是否有偏差
fn check_policy_bias(trainer: &AlphaZeroTrainer) {
    let game = Connect4::new();
    let (policy, value) = trainer.predict(&game);
    
    println!("🔍 空棋盘policy检查:");
    println!("  Value预测: {:.4}", value);
    println!("  Policy分布:");
    
    // 直接从policy tensor读取数据，避免MPS float64问题
    // 硬编码size=7避免调用policy.size()
    let policy_vec: Vec<f32> = unsafe {
        let data_ptr = policy.data_ptr() as *const f32;
        std::slice::from_raw_parts(data_ptr, 7).to_vec()  // Connect4固定7列
    };
    
    let max_prob = policy_vec.iter().cloned().fold(0.0f32, f32::max);
    
    for (col, &prob) in policy_vec.iter().enumerate() {
        let bar_length = (prob / max_prob * 50.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("    Col {}: {:.4} {}", col, prob, bar);
    }
    
    // 检查是否有某一列过于突出
    let max_col = policy_vec.iter()
        .enumerate()
        .max_by(|(_, a), (_, b): &(usize, &f32)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    
    if max_prob > 0.3 {
        println!("  ⚠️  检测到偏差：column {} 的概率过高 ({:.1}%)", max_col, max_prob * 100.0);
    } else {
        println!("  ✅ 分布较为均匀");
    }
}
