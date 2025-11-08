// 分析网络学到的policy质量
use gomoku::az_mcts_rollout::MCTSWithRollout;
use gomoku::az_net::Connect4Trainer;
use gomoku::connect4::Connect4;
use tch::Tensor;

fn main() {
    println!("🔍 分析Policy质量\n");

    let mut trainer = Connect4Trainer::new(64, 0.0003);
    if let Err(_) = trainer.load("connect4_model_iter_30.pt") {
        println!("⚠️  使用随机初始化模型");
    } else {
        println!("✅ 加载了训练后的模型");
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("测试1: 初始局面 - 网络Policy vs 纯MCTS Policy");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let game = Connect4::new();

    // 网络的policy
    let board_tensor = Tensor::f_from_slice(&game.to_tensor())
        .unwrap()
        .reshape(&[1, 3, 6, 7]);
    let (policy, _value) = trainer.net.predict(&board_tensor);
    let mut policy_vec = vec![0.0f32; 7];
    policy.view([7]).copy_data(&mut policy_vec, 7);

    // 转换为概率
    let policy_probs: Vec<f32> = policy_vec.iter().map(|&x| x.exp()).collect();
    let sum: f32 = policy_probs.iter().sum();
    let policy_probs: Vec<f32> = policy_probs.iter().map(|&x| x / sum).collect();

    println!("网络Policy (初始局面):");
    for (i, &p) in policy_probs.iter().enumerate() {
        println!("  列{}: {:.1}%", i, p * 100.0);
    }

    // 纯MCTS的policy (作为baseline)
    println!("\n纯MCTS Policy (50模拟):");
    let mut pure_mcts = MCTSWithRollout::new(50, true);
    let mcts_probs = pure_mcts.search(&game, &trainer.net);
    for (i, &p) in mcts_probs.iter().enumerate() {
        println!("  列{}: {:.1}%", i, p * 100.0);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("测试2: 简单威胁局面");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 创建一个简单的威胁：X有3个连在一起
    let mut game2 = Connect4::new();
    game2.play(3).unwrap(); // X
    game2.play(3).unwrap(); // O
    game2.play(2).unwrap(); // X
    game2.play(2).unwrap(); // O
    game2.play(4).unwrap(); // X
    game2.play(4).unwrap(); // O
                            // 现在X如果下列1或列5就能赢

    println!("当前局面:");
    game2.print();
    println!("X应该下列1或列5来连4赢");

    let board_tensor = Tensor::f_from_slice(&game2.to_tensor())
        .unwrap()
        .reshape(&[1, 3, 6, 7]);
    let (policy, _value) = trainer.net.predict(&board_tensor);
    let mut policy_vec = vec![0.0f32; 7];
    policy.view([7]).copy_data(&mut policy_vec, 7);

    let policy_probs: Vec<f32> = policy_vec.iter().map(|&x| x.exp()).collect();
    let sum: f32 = policy_probs.iter().sum();
    let policy_probs: Vec<f32> = policy_probs.iter().map(|&x| x / sum).collect();

    println!("\n网络Policy:");
    for (i, &p) in policy_probs.iter().enumerate() {
        let marker = if i == 1 || i == 5 { " ← 正确" } else { "" };
        println!("  列{}: {:.1}%{}", i, p * 100.0, marker);
    }

    let best_col = policy_probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    if best_col == 1 || best_col == 5 {
        println!("\n✅ 网络选择了正确的列{}", best_col);
    } else {
        println!("\n❌ 网络选择了错误的列{} (应该是1或5)", best_col);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("测试3: 必须防守的局面");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut game3 = Connect4::new();
    game3.play(3).unwrap(); // X
    game3.play(2).unwrap(); // O
    game3.play(3).unwrap(); // X
    game3.play(2).unwrap(); // O
    game3.play(3).unwrap(); // X
    game3.play(2).unwrap(); // O
                            // 现在O如果不在列3防守，X下一步列3就赢了

    println!("当前局面:");
    game3.print();
    println!("O必须在列3防守！");

    let board_tensor = Tensor::f_from_slice(&game3.to_tensor())
        .unwrap()
        .reshape(&[1, 3, 6, 7]);
    let (policy, _value) = trainer.net.predict(&board_tensor);
    let mut policy_vec = vec![0.0f32; 7];
    policy.view([7]).copy_data(&mut policy_vec, 7);

    let policy_probs: Vec<f32> = policy_vec.iter().map(|&x| x.exp()).collect();
    let sum: f32 = policy_probs.iter().sum();
    let policy_probs: Vec<f32> = policy_probs.iter().map(|&x| x / sum).collect();

    println!("\n网络Policy:");
    for (i, &p) in policy_probs.iter().enumerate() {
        let marker = if i == 3 { " ← 正确(防守)" } else { "" };
        println!("  列{}: {:.1}%{}", i, p * 100.0, marker);
    }

    let best_col = policy_probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    if best_col == 3 {
        println!("\n✅ 网络正确防守");
    } else {
        println!("\n❌ 网络没有防守列3，选择了列{}", best_col);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 结论");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("如果网络在简单局面都无法识别明显的赢法/防守，");
    println!("说明policy学习有问题。可能的原因:");
    println!("  1. 训练数据太少（只有900局）");
    println!("  2. 网络容量太小（只有3层）");
    println!("  3. MCTS模拟太少导致训练数据质量差");
}
