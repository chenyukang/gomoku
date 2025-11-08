use gomoku::az_mcts_rollout::MCTSWithRollout;
use gomoku::az_net::Connect4Net;
use gomoku::az_resnet::Connect4ResNetTrainer;
use gomoku::connect4::Connect4;
use rand::Rng;
use tch::{nn, Device, Tensor};

// 创建一个临时的Connect4Net来兼容MCTS接口（实际MCTS用rollout不需要网络）
fn create_dummy_net(device: Device) -> Connect4Net {
    let vs = nn::VarStore::new(device);
    Connect4Net::new(&vs.root(), 64)
}

fn main() {
    println!("🚀 终极优化训练 - ResNet + GPU + 模仿学习\n");
    println!("═══════════════════════════════════════════════════");

    // 超参数 - 基于所有经验调优
    let num_filters = 128; // 增加到128 (vs 64)
    let num_residual_blocks = 10; // 10层残差网络
    let learning_rate = 0.001;
    let mcts_simulations = 200; // 更强的MCTS老师
    let games_per_iter = 50; // 更多游戏数据
    let batch_size = 128; // GPU支持更大batch
    let train_epochs = 30; // 更多训练轮次
    let num_iterations = 50; // 50次迭代
    let buffer_size = 20000; // 更大的缓冲区

    println!("📋 超参数配置:");
    println!(
        "  网络: ResNet-{} with {} filters",
        num_residual_blocks, num_filters
    );
    println!("  学习率: {}", learning_rate);
    println!("  MCTS模拟: {}次 (纯rollout，不用网络)", mcts_simulations);
    println!("  每轮游戏: {}局", games_per_iter);
    println!("  批次大小: {}", batch_size);
    println!("  训练轮次: {} epochs/iter", train_epochs);
    println!("  数据缓冲: {} 条", buffer_size);
    println!("═══════════════════════════════════════════════════\n");

    // 创建trainer (自动使用GPU)
    let mut trainer = Connect4ResNetTrainer::new(num_filters, num_residual_blocks, learning_rate);

    // 创建一个dummy net用于MCTS (实际上MCTS用rollout，不需要网络)
    let dummy_net = create_dummy_net(trainer.device());

    // 数据缓冲区
    let mut all_data: Vec<(Vec<f32>, Vec<f32>, f32)> = Vec::new();
    let mut mcts = MCTSWithRollout::new(mcts_simulations, true);

    // 最佳模型追踪
    let mut best_win_rate = 0.0;
    let mut patience = 0;
    let max_patience = 10;

    for iteration in 1..=num_iterations {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📊 迭代 {}/{}", iteration, num_iterations);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // 1. 收集MCTS示范数据
        print!("  🎮 收集MCTS示范数据... ");
        for _ in 0..games_per_iter {
            let game_data = collect_mcts_game(&mut mcts, &trainer, &dummy_net);
            all_data.extend(game_data);
        }

        // 保持缓冲区大小
        if all_data.len() > buffer_size {
            all_data.drain(0..all_data.len() - buffer_size);
        }
        println!("✅ 数据总量: {}", all_data.len());

        // 2. 训练网络
        if all_data.len() >= batch_size {
            print!("  🎯 训练网络... ");
            let mut total_loss = 0.0;
            let mut count = 0;

            for _ in 0..train_epochs {
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

                let device = trainer.device();
                let boards_t = Tensor::f_from_slice(&batch_boards)
                    .unwrap()
                    .reshape(&[batch_size as i64, 3, 6, 7])
                    .to_device(device);
                let policies_t = Tensor::f_from_slice(&batch_policies)
                    .unwrap()
                    .reshape(&[batch_size as i64, 7])
                    .to_device(device);
                let values_t = Tensor::f_from_slice(&batch_values)
                    .unwrap()
                    .reshape(&[batch_size as i64, 1])
                    .to_device(device);

                let (_p_loss, _v_loss, t_loss) =
                    trainer.train_batch(&boards_t, &policies_t, &values_t);

                total_loss += t_loss;
                count += 1;
            }

            let avg_loss = total_loss / count as f64;
            println!("✅ 平均loss: {:.4}", avg_loss);
        }

        // 3. 每5轮评估一次
        if iteration % 5 == 0 {
            println!("\n  📊 评估学习效果:");

            let (win_vs_random, win_vs_mcts) = evaluate_student(&trainer, iteration);

            println!("    vs 随机: {:.0}%", win_vs_random * 100.0);
            println!("    vs 弱MCTS(50模拟): {:.0}%", win_vs_mcts * 100.0);

            // 保存模型
            let model_path = format!("connect4_resnet_iter_{}.pt", iteration);
            trainer.save(&model_path).unwrap();
            println!("  💾 保存: {}", model_path);

            // 检查是否有改进
            if win_vs_random > best_win_rate {
                best_win_rate = win_vs_random;
                patience = 0;
                trainer.save("connect4_resnet_best.pt").unwrap();
                println!("  🌟 新的最佳模型！Win rate: {:.0}%", best_win_rate * 100.0);
            } else {
                patience += 1;
                println!("  ⏳ 无改进 ({}/{})", patience, max_patience);
            }

            // Early stopping
            if patience >= max_patience {
                println!("\n⚠️  {} 轮无改进，触发早停", max_patience);
                break;
            }

            println!();
        }
    }

    println!("\n🎉 训练完成！");
    println!("\n📊 最终评估:");
    let (win_vs_random, win_vs_mcts) = evaluate_student(&trainer, 999);
    println!("    vs 随机: {:.0}%", win_vs_random * 100.0);
    println!("    vs 弱MCTS(50模拟): {:.0}%", win_vs_mcts * 100.0);

    trainer.save("connect4_resnet_final.pt").unwrap();
    println!("\n💾 最终模型: connect4_resnet_final.pt");
    println!(
        "💾 最佳模型: connect4_resnet_best.pt (Win rate: {:.0}%)",
        best_win_rate * 100.0
    );
}

fn collect_mcts_game(
    mcts: &mut MCTSWithRollout,
    _trainer: &Connect4ResNetTrainer,
    dummy_net: &Connect4Net,
) -> Vec<(Vec<f32>, Vec<f32>, f32)> {
    mcts.reset();
    let mut game = Connect4::new();
    let mut history = Vec::new();

    while !game.is_game_over() {
        // 使用纯rollout MCTS（不需要网络评估）
        let policy = mcts.search(&game, dummy_net);

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

    let final_value = if game.is_game_over() && game.winner().is_some() {
        1.0
    } else {
        0.0
    };

    let mut training_data = Vec::new();
    for (i, (board, policy)) in history.iter().enumerate() {
        let value = if i % 2 == (history.len() - 1) % 2 {
            final_value
        } else {
            -final_value
        };

        training_data.push((board.clone(), policy.clone(), value));
    }

    training_data
}

fn evaluate_student(trainer: &Connect4ResNetTrainer, _iteration: usize) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    let device = trainer.device();

    // 创建dummy net用于MCTS
    let dummy_net = create_dummy_net(device);

    // 测试 vs 随机
    let mut wins = 0;
    let num_games = 30;

    for _ in 0..num_games {
        let mut game = Connect4::new();
        let student_first = rng.gen_bool(0.5);

        while !game.is_game_over() {
            let is_student_turn = (game.current_player() == 1) == student_first;

            if is_student_turn {
                let board_t = Tensor::f_from_slice(&game.to_tensor())
                    .unwrap()
                    .reshape(&[1, 3, 6, 7])
                    .to_device(device);
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

    let win_rate_random = wins as f64 / num_games as f64;

    // 测试 vs 弱MCTS
    let mut wins_mcts = 0;
    let mut mcts_weak = MCTSWithRollout::new(50, true);

    for _ in 0..20 {
        mcts_weak.reset();
        let mut game = Connect4::new();
        let student_first = rng.gen_bool(0.5);

        while !game.is_game_over() {
            let is_student_turn = (game.current_player() == 1) == student_first;

            if is_student_turn {
                let board_t = Tensor::f_from_slice(&game.to_tensor())
                    .unwrap()
                    .reshape(&[1, 3, 6, 7])
                    .to_device(device);
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
                let policy = mcts_weak.search(&game, &dummy_net);
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

    let win_rate_mcts = wins_mcts as f64 / 20.0;

    (win_rate_random, win_rate_mcts)
}
