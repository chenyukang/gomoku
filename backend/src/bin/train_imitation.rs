// 向纯MCTS学习 - 模仿学习（Imitation Learning）
use gomoku::az_mcts_rollout::MCTSWithRollout;
use gomoku::az_net::Connect4Trainer; // 用原来的网络，只是训练方式不同
use gomoku::connect4::Connect4;
use rand::Rng;
use tch::Tensor;

fn main() {
    println!("🎓 向纯MCTS学习训练\n");
    println!("策略：让神经网络模仿强手（纯MCTS）的走法");
    println!("优势：直接学习好策略，不需要自我探索\n");

    let mut trainer = Connect4Trainer::new(64, 0.001);

    let num_iterations = 50;  // 减少迭代次数（数据质量更重要）
    let games_per_iter = 50;  // 增加每轮游戏数
    let mcts_simulations = 200; // 增加MCTS模拟次数 - 更强的老师！
    let batch_size = 64;
    let train_epochs = 30;  // 增加训练轮数

    println!("📋 配置:");
    println!("  网络: 5层CNN + 64 filters");
    println!("  老师: 纯MCTS ({}次模拟，无网络)", mcts_simulations);
    println!("  数据: 每轮{}局游戏", games_per_iter);
    println!("  训练: {}epochs per iteration\n", train_epochs);

    let mut all_data: Vec<(Vec<f32>, Vec<f32>, f32)> = Vec::new();
    let mut mcts = MCTSWithRollout::new(mcts_simulations, true);

    for iteration in 1..=num_iterations {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📊 迭代 {}/{}", iteration, num_iterations);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // 生成训练数据：观察MCTS怎么下
        print!("  🎮 收集MCTS示范数据... ");
        for _ in 0..games_per_iter {
            let game_data = collect_mcts_game(&mut mcts, &trainer);
            all_data.extend(game_data);
        }
        println!("✅ 数据总量: {}", all_data.len());

        // 训练网络去模仿MCTS
        if all_data.len() >= batch_size {
            print!("  🎯 训练网络模仿MCTS... ");
            let mut total_loss = 0.0;
            let mut count = 0;

            for _ in 0..train_epochs {
                // 随机采样一个batch
                let mut rng = rand::thread_rng();
                let mut batch_boards: Vec<f32> = Vec::new();
                let mut batch_policies: Vec<f32> = Vec::new();
                let mut batch_values: Vec<f32> = Vec::new();

                for _ in 0..batch_size.min(all_data.len()) {
                    let idx = rng.gen_range(0..all_data.len());
                    let (board, policy, value) = &all_data[idx];
                    batch_boards.extend(board);
                    batch_policies.extend(policy);
                    batch_values.push(*value);
                }

                let boards_t = Tensor::f_from_slice(&batch_boards).unwrap().reshape(&[
                    batch_size as i64,
                    3,
                    6,
                    7,
                ]);
                let policies_t = Tensor::f_from_slice(&batch_policies)
                    .unwrap()
                    .reshape(&[batch_size as i64, 7]);
                let values_t = Tensor::f_from_slice(&batch_values)
                    .unwrap()
                    .reshape(&[batch_size as i64, 1]);

                let (_p_loss, _v_loss, t_loss) =
                    trainer.train_batch(&boards_t, &policies_t, &values_t);

                total_loss += t_loss;
                count += 1;
            }

            let avg_loss = total_loss / count as f64;
            println!("✅ 平均loss: {:.4}", avg_loss);
        }

        // 保持最近5000条数据
        if all_data.len() > 5000 {
            all_data.drain(0..all_data.len() - 5000);
        }

        // 每10轮评估一次
        if iteration % 10 == 0 {
            println!("\n  📊 评估学习效果:");
            evaluate_student(&trainer, iteration);

            let filename = format!("connect4_imitation_iter_{}.pt", iteration);
            trainer.save(&filename).ok();
            println!("  💾 保存: {}\n", filename);
        }
    }

    println!("\n🎉 训练完成！");
    println!("\n📊 最终评估:");
    evaluate_student(&trainer, num_iterations);

    trainer.save("connect4_imitation_final.pt").ok();
    println!("\n💾 最终模型: connect4_imitation_final.pt");
}

// 收集一局MCTS的示范数据
fn collect_mcts_game(
    mcts: &mut MCTSWithRollout,
    trainer: &Connect4Trainer,
) -> Vec<(Vec<f32>, Vec<f32>, f32)> {
    mcts.reset(); // 重置MCTS树，避免跨游戏污染
    let mut game = Connect4::new();
    let mut history = Vec::new();

    while !game.is_game_over() {
        // 让MCTS决策
        let policy = mcts.search(&game, &trainer.net);

        // 选择最佳动作（不用温度，直接选最好的）
        let action = policy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        history.push((game.to_tensor(), policy));

        if game.play(action).is_err() {
            break;
        }
    }

    // 标注结果
    let final_value = if game.is_game_over() && game.winner().is_some() {
        1.0
    } else {
        0.0
    };

    // 生成训练数据
    let mut training_data = Vec::new();
    for (i, (board, policy)) in history.iter().enumerate() {
        // 从当前玩家视角
        let value = if i % 2 == (history.len() - 1) % 2 {
            final_value
        } else {
            -final_value
        };

        training_data.push((board.clone(), policy.clone(), value));
    }

    training_data
}

// 评估学生网络
fn evaluate_student(trainer: &Connect4Trainer, _iteration: usize) {
    let mut rng = rand::thread_rng();

    // 测试vs随机
    let mut wins = 0;
    let num_games = 20;

    for _ in 0..num_games {
        let mut game = Connect4::new();
        let student_first = rng.gen_bool(0.5);

        while !game.is_game_over() {
            let is_student_turn = (game.current_player() == 1) == student_first;

            if is_student_turn {
                // 学生网络决策
                let board_t = Tensor::f_from_slice(&game.to_tensor())
                    .unwrap()
                    .reshape(&[1, 3, 6, 7]);
                let (policy, _) = trainer.net.predict(&board_t);

                let mut policy_vec = vec![0.0f32; 7];
                policy.view([7i64]).copy_data(&mut policy_vec, 7);

                // Softmax
                let max_val = policy_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = policy_vec.iter().map(|&x| (x - max_val).exp()).collect();
                let sum: f32 = exp_vals.iter().sum();
                let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

                let action = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();

                if game.play(action).is_err() {
                    break;
                }
            } else {
                // 随机对手
                let valid = game.legal_moves();
                if valid.is_empty() {
                    break;
                }
                let action = valid[rng.gen_range(0..valid.len())];
                game.play(action).unwrap();
            }
        }

        if game.winner().is_some() {
            let winner = game.winner().unwrap();
            if (winner == 1) == student_first {
                wins += 1;
            }
        }
    }

    println!(
        "    vs 随机: {}/{}  ({:.0}%)",
        wins,
        num_games,
        wins as f64 / num_games as f64 * 100.0
    );

    // 测试vs弱MCTS
    let mut wins_mcts = 0;
    let mut mcts_weak = MCTSWithRollout::new(30, true);

    for _ in 0..10 {
        mcts_weak.reset(); // 重置MCTS树
        let mut game = Connect4::new();
        let student_first = rng.gen_bool(0.5);

        while !game.is_game_over() {
            let is_student_turn = (game.current_player() == 1) == student_first;

            if is_student_turn {
                // 学生网络
                let board_t = Tensor::f_from_slice(&game.to_tensor())
                    .unwrap()
                    .reshape(&[1, 3, 6, 7]);
                let (policy, _) = trainer.net.predict(&board_t);

                let mut policy_vec = vec![0.0f32; 7];
                policy.view([7i64]).copy_data(&mut policy_vec, 7);

                let max_val = policy_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = policy_vec.iter().map(|&x| (x - max_val).exp()).collect();
                let sum: f32 = exp_vals.iter().sum();
                let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

                let action = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();

                if game.play(action).is_err() {
                    break;
                }
            } else {
                // 弱MCTS (30次模拟)
                let policy = mcts_weak.search(&game, &trainer.net);
                let action = policy
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();

                if game.play(action).is_err() {
                    break;
                }
            }
        }

        if game.winner().is_some() {
            let winner = game.winner().unwrap();
            if (winner == 1) == student_first {
                wins_mcts += 1;
            }
        }
    }

    println!(
        "    vs 弱MCTS(30模拟): {}/10  ({:.0}%)",
        wins_mcts,
        wins_mcts as f64 * 10.0
    );
}
