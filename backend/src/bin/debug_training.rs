// 调试训练数据质量

use gomoku::az_mcts::MCTS;
use gomoku::az_trainer::AlphaZeroTrainer;
use gomoku::connect4::Connect4;

fn main() {
    println!("🔍 调试训练数据质量\n");

    let mut trainer = AlphaZeroTrainer::new(64, 0.0003, 5000, 200);

    println!("━━━ 测试1: 检查自对弈游戏结果分布 ━━━\n");
    test_self_play_distribution(&mut trainer);

    println!("\n━━━ 测试2: 检查MCTS质量 ━━━\n");
    test_mcts_quality(&trainer);

    println!("\n━━━ 测试3: 对比MCTS策略 vs 随机策略 ━━━\n");
    test_mcts_vs_random(&trainer);
}

fn test_self_play_distribution(trainer: &mut AlphaZeroTrainer) {
    let mut player1_wins = 0;
    let mut player2_wins = 0;
    let mut draws = 0;
    let num_games = 20;

    println!("进行 {} 局自对弈（温度=0，纯MCTS，无随机性）...", num_games);

    for _ in 0..num_games {
        let mut game = Connect4::new();

        while !game.is_game_over() {
            let mut mcts = MCTS::new(200);
            mcts.search(&game, &trainer.trainer.net);
            let action = mcts.select_action(0.0); // 温度0，完全确定性
            game.play(action).ok();
        }

        match game.winner() {
            Some(1) => player1_wins += 1,
            Some(2) => player2_wins += 1,
            Some(0) => draws += 1,
            _ => {}
        }
    }

    println!("\n结果分布:");
    println!(
        "  玩家1胜: {} ({:.0}%)",
        player1_wins,
        player1_wins as f32 / num_games as f32 * 100.0
    );
    println!(
        "  玩家2胜: {} ({:.0}%)",
        player2_wins,
        player2_wins as f32 / num_games as f32 * 100.0
    );
    println!(
        "  平局: {} ({:.0}%)",
        draws,
        draws as f32 / num_games as f32 * 100.0
    );

    println!("\n📊 分析:");
    if ((player1_wins as i32) - (player2_wins as i32)).abs() > 5 {
        println!("⚠️  严重不平衡！说明自对弈策略有严重偏差");
        println!("   可能原因: 先手优势太大 或 网络输出有偏");
    } else if draws > num_games / 2 {
        println!("⚠️  平局太多！说明双方都不会进攻，过于保守");
    } else {
        println!("✅ 分布相对合理");
    }
}

fn test_mcts_quality(trainer: &AlphaZeroTrainer) {
    println!("测试MCTS搜索质量（从初始局面）...\n");

    let game = Connect4::new();
    let mut mcts = MCTS::new(200);
    let policy = mcts.search(&game, &trainer.trainer.net);

    println!("MCTS策略分布 (200次模拟):");
    for (col, prob) in policy.iter().enumerate() {
        if *prob > 0.0 {
            println!("  列{}: {:.3} ({:.0}%)", col, prob, prob * 100.0);
        }
    }

    let max_prob = policy.iter().cloned().fold(0.0f32, f32::max);
    let entropy = -policy
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f32>();

    println!("\n统计:");
    println!("  最大概率: {:.1}%", max_prob * 100.0);
    println!("  熵: {:.2} (越高越分散)", entropy);

    if max_prob < 0.15 {
        println!("\n⚠️  概率过于均匀！MCTS没有找到明显的好走法");
        println!("   说明网络评估很差，或搜索次数不够");
    } else if max_prob > 0.8 {
        println!("\n✅ 有明确偏好，MCTS质量好");
    } else {
        println!("\n📊 中等置信度");
    }
}

fn test_mcts_vs_random(trainer: &AlphaZeroTrainer) {
    println!("MCTS(200模拟) vs 随机策略，10局...\n");

    let mut mcts_wins = 0;

    for _game_num in 0..10 {
        let mut game = Connect4::new();

        while !game.is_game_over() {
            let action = if game.current_player() == 1 {
                // 玩家1: MCTS
                let mut mcts = MCTS::new(200);
                mcts.search(&game, &trainer.trainer.net);
                mcts.select_action(0.0)
            } else {
                // 玩家2: 随机
                use rand::Rng;
                let legal = game.legal_moves();
                legal[rand::thread_rng().gen_range(0..legal.len())]
            };

            game.play(action).ok();
        }

        if game.winner() == Some(1) {
            mcts_wins += 1;
            print!("✓");
        } else {
            print!("✗");
        }
    }

    println!(
        "\n\nMCTS胜率: {}/10 ({:.0}%)",
        mcts_wins,
        mcts_wins as f32 * 10.0
    );

    if mcts_wins < 5 {
        println!("\n🔥 严重问题！MCTS应该能100%战胜随机");
        println!("   说明: 网络引导让MCTS变弱了！");
        println!("   解决: 可能需要 (1)增加模拟次数 或 (2)不使用网络价值");
    } else if mcts_wins < 10 {
        println!("\n⚠️  MCTS质量不够强");
    } else {
        println!("\n✅ MCTS质量正常");
    }
}
