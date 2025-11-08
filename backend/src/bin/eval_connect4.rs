// AlphaZero 模型评估程序

use gomoku::az_eval::{evaluate_symmetric, Player};
use gomoku::az_net::Connect4Trainer;

fn main() {
    println!("🎯 AlphaZero Connect4 模型评估\n");

    // 1. 尝试加载不同的模型
    println!("📂 尝试加载模型...");

    let model_configs = vec![
        ("connect4_model_final.pt", 64),
        ("test_model_iter_3.pt", 32),
        ("test_model_iter_2.pt", 32),
        ("test_model_iter_1.pt", 32),
    ];

    let mut trainer = None;
    for (path, filters) in model_configs {
        println!("  尝试: {} (filters={})", path, filters);
        let mut t = Connect4Trainer::new(filters, 0.001);
        if t.load(path).is_ok() {
            println!("✅ 成功加载: {}\n", path);
            trainer = Some(t);
            break;
        }
    }

    let trainer = match trainer {
        Some(t) => t,
        None => {
            eprintln!("❌ 无法加载任何模型文件");
            eprintln!("请先运行训练程序:");
            eprintln!("  cargo run --features alphazero --bin test_connect4");
            return;
        }
    };

    // 2. 创建不同类型的玩家
    let alphazero_strong = Player::AlphaZero {
        net: &trainer.net,
        simulations: 100, // 高质量搜索
    };

    let alphazero_fast = Player::AlphaZero {
        net: &trainer.net,
        simulations: 50, // 快速搜索
    };

    let pure_mcts = Player::PureMCTS { simulations: 50 };

    let random_player = Player::Random;

    // 3. 评估：AlphaZero vs Random
    println!("\n╔══════════════════════════════════════╗");
    println!("║  测试1: AlphaZero vs 随机玩家        ║");
    println!("╚══════════════════════════════════════╝");
    evaluate_symmetric(&alphazero_fast, &random_player, 25, false);

    // 4. 评估：AlphaZero vs Pure MCTS
    println!("\n╔══════════════════════════════════════╗");
    println!("║  测试2: AlphaZero vs 纯MCTS          ║");
    println!("╚══════════════════════════════════════╝");
    evaluate_symmetric(&alphazero_strong, &pure_mcts, 20, false);

    // 5. 评估：强弱 AlphaZero 对比
    println!("\n╔══════════════════════════════════════╗");
    println!("║  测试3: AlphaZero(100) vs AlphaZero(50) ║");
    println!("╚══════════════════════════════════════╝");
    evaluate_symmetric(&alphazero_strong, &alphazero_fast, 20, false);

    // 6. 观看一局详细对局
    println!("\n╔══════════════════════════════════════╗");
    println!("║  演示对局: AlphaZero vs 纯MCTS       ║");
    println!("╚══════════════════════════════════════╝");
    gomoku::az_eval::play_game(&alphazero_strong, &pure_mcts, true);

    println!("\n🎉 评估完成！");
}
