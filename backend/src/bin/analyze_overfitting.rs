// 分析为什么会过拟合
use gomoku::az_net::Connect4Trainer;
use gomoku::connect4::Connect4;
use rand::Rng;

fn main() {
    println!("🔍 过拟合原因分析\n");

    println!("对比不同迭代的模型:\n");

    // 加载迭代9（最好的）
    let mut trainer9 = Connect4Trainer::new(64, 0.0003);
    if trainer9.load("connect4_highq_iter_9.pt").is_ok() {
        println!("✅ 加载迭代9模型（峰值70%）");
    }

    // 加载迭代18（过拟合后）
    let mut trainer18 = Connect4Trainer::new(64, 0.0003);
    if trainer18.load("connect4_highq_iter_18.pt").is_ok() {
        println!("✅ 加载迭代18模型（下降到50%）");
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("测试：Policy的多样性");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 测试10个随机局面
    let mut rng = rand::thread_rng();
    let mut diversity9 = Vec::new();
    let mut diversity18 = Vec::new();

    for test_num in 0..10 {
        let mut game = Connect4::new();

        // 随机走几步
        let steps = rng.gen_range(3..8);
        for _ in 0..steps {
            if game.is_game_over() {
                break;
            }
            let moves = (0..7)
                .filter(|&col| game.play(col).is_ok())
                .collect::<Vec<_>>();
            if moves.is_empty() {
                break;
            }
            let m = moves[rng.gen_range(0..moves.len())];
            let _ = game.play(m);
        }

        if game.is_game_over() {
            continue;
        }

        // 获取两个模型的policy
        let board = game.to_tensor();
        let board_tensor = tch::Tensor::f_from_slice(&board)
            .unwrap()
            .reshape(&[1, 3, 6, 7]);

        let (policy9, _) = trainer9.net.predict(&board_tensor);
        let (policy18, _) = trainer18.net.predict(&board_tensor);

        let mut p9 = vec![0.0f32; 7];
        let mut p18 = vec![0.0f32; 7];
        policy9.view([7]).copy_data(&mut p9, 7);
        policy18.view([7]).copy_data(&mut p18, 7);

        // 计算熵（多样性指标）
        let entropy9 = calculate_entropy(&p9);
        let entropy18 = calculate_entropy(&p18);

        diversity9.push(entropy9);
        diversity18.push(entropy18);

        if test_num < 3 {
            println!("局面{}:", test_num + 1);
            println!(
                "  迭代9  熵={:.3}  最大概率={:.1}%",
                entropy9,
                p9.iter()
                    .map(|x| x.exp())
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
                    * 100.0
            );
            println!(
                "  迭代18 熵={:.3}  最大概率={:.1}%",
                entropy18,
                p18.iter()
                    .map(|x| x.exp())
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
                    * 100.0
            );
            println!();
        }
    }

    let avg9 = diversity9.iter().sum::<f32>() / diversity9.len() as f32;
    let avg18 = diversity18.iter().sum::<f32>() / diversity18.len() as f32;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 平均Policy熵（多样性）:");
    println!("  迭代9:  {:.3} (更多样)", avg9);
    println!("  迭代18: {:.3} (更确定/过拟合)", avg18);

    if avg18 < avg9 {
        println!("\n⚠️  迭代18的policy更确定（熵更低）");
        println!("   这说明模型过度自信，失去了探索能力");
        println!("   对于没见过的局面（如随机玩家的走法）表现变差");
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("测试：对意外走法的应对");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 测试：对手走了"愚蠢"的一步后，能否抓住机会
    test_unexpected_move(&trainer9, &trainer18);

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("💡 结论");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("过拟合的原因:");
    println!("  1. 只和自己下棋 → 数据分布越来越窄");
    println!("  2. 温度太低(0.1) → 后期几乎不探索新走法");
    println!("  3. 训练次数太多(100批) → 过度拟合训练数据");
    println!("  4. Buffer满后覆盖 → 丢失早期多样数据");
    println!("\n解决方案:");
    println!("  ✅ 增大buffer (5000→10000)");
    println!("  ✅ 保持更高温度 (0.5-0.7而非0.1)");
    println!("  ✅ 减少训练批次 (100→50)");
    println!("  ✅ 早停机制 (检测到不再提升就停止)");
}

fn calculate_entropy(logits: &[f32]) -> f32 {
    // 转为概率
    let probs: Vec<f32> = logits.iter().map(|x| x.exp()).collect();
    let sum: f32 = probs.iter().sum();
    let probs: Vec<f32> = probs.iter().map(|x| x / sum).collect();

    // 计算熵 H = -Σ p*log(p)
    let mut entropy = 0.0;
    for &p in &probs {
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

fn test_unexpected_move(trainer9: &Connect4Trainer, trainer18: &Connect4Trainer) {
    // 创建一个局面：对手走了边缘（不太好的）
    let mut game = Connect4::new();
    game.play(3).unwrap(); // X中间（好）
    game.play(0).unwrap(); // O左边（不太好）
    game.play(3).unwrap(); // X继续中间

    println!("局面：对手刚走了不太好的边缘位置\n");
    game.print();

    let board = game.to_tensor();
    let board_tensor = tch::Tensor::f_from_slice(&board)
        .unwrap()
        .reshape(&[1, 3, 6, 7]);

    let (policy9, value9) = trainer9.net.predict(&board_tensor);
    let (policy18, value18) = trainer18.net.predict(&board_tensor);

    let mut p9 = vec![0.0f32; 7];
    let mut p18 = vec![0.0f32; 7];
    let mut v9 = vec![0.0f32; 1];
    let mut v18 = vec![0.0f32; 1];

    policy9.view([7]).copy_data(&mut p9, 7);
    policy18.view([7]).copy_data(&mut p18, 7);
    value9.copy_data(&mut v9, 1);
    value18.copy_data(&mut v18, 1);

    // 转概率
    let probs9: Vec<f32> = p9.iter().map(|x| x.exp()).collect();
    let sum9: f32 = probs9.iter().sum();
    let probs9: Vec<f32> = probs9.iter().map(|x| x / sum9).collect();

    let probs18: Vec<f32> = p18.iter().map(|x| x.exp()).collect();
    let sum18: f32 = probs18.iter().sum();
    let probs18: Vec<f32> = probs18.iter().map(|x| x / sum18).collect();

    println!("\n迭代9的判断:");
    println!("  Value: {:.3} (评估局面)", v9[0]);
    println!(
        "  Policy: 列3={:.1}% 列4={:.1}% 列2={:.1}%",
        probs9[3] * 100.0,
        probs9[4] * 100.0,
        probs9[2] * 100.0
    );

    println!("\n迭代18的判断:");
    println!("  Value: {:.3} (评估局面)", v18[0]);
    println!(
        "  Policy: 列3={:.1}% 列4={:.1}% 列2={:.1}%",
        probs18[3] * 100.0,
        probs18[4] * 100.0,
        probs18[2] * 100.0
    );

    println!("\n观察：迭代18是否能灵活应对意外局面？");
}
