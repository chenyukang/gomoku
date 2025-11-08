// 简单快速评估

use gomoku::az_eval::{evaluate, Player};
use gomoku::az_net::Connect4Trainer;

fn main() {
    println!("🎯 快速评估测试\n");

    println!("📂 加载模型...");
    let mut trainer = Connect4Trainer::new(32, 0.001);

    if let Err(e) = trainer.load("test_model_iter_3.pt") {
        eprintln!("❌ 加载失败: {}", e);
        return;
    }
    println!("✅ 模型加载成功\n");

    let alphazero = Player::AlphaZero {
        net: &trainer.net,
        simulations: 50,
    };

    let random_player = Player::Random;

    println!("测试: AlphaZero(50次模拟) vs 随机玩家");
    println!("进行 20 局对战...\n");

    let stats = evaluate(&alphazero, &random_player, 20, false);

    println!("\n✅ 评估完成！");
    println!("AlphaZero 胜率: {:.1}%", stats.player1_winrate() * 100.0);
}
