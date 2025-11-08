// 评估纯网络（不用MCTS）的表现
use gomoku::az_eval::{evaluate, Player};
use gomoku::az_net::Connect4Trainer;

fn main() {
    println!("🔍 评估纯网络（无MCTS）的棋力\n");

    // 加载最新的模型
    let mut trainer = Connect4Trainer::new(64, 0.0003);
    if let Err(e) = trainer.load("connect4_model_iter_30.pt") {
        println!("⚠️  无法加载模型: {}, 使用随机初始化", e);
    }

    println!("📊 测试1: 网络Value vs 随机（MCTS 50模拟）\n");
    let alphazero_with_value = Player::AlphaZero {
        net: &trainer.net,
        simulations: 50,
    };
    let random = Player::Random;
    let stats1 = evaluate(&alphazero_with_value, &random, 20, true);
    println!("胜率: {:.1}%\n", stats1.player1_winrate() * 100.0);

    println!("📊 测试2: 网络+Rollout vs 随机（MCTS 50模拟）\n");
    let alphazero_with_rollout = Player::AlphaZeroRollout {
        net: &trainer.net,
        simulations: 50,
    };
    let stats2 = evaluate(&alphazero_with_rollout, &random, 20, true);
    println!("胜率: {:.1}%\n", stats2.player1_winrate() * 100.0);

    println!("📊 测试3: 纯MCTS vs 随机（50模拟，无网络）\n");
    let pure_mcts = Player::PureMCTS { simulations: 50 };
    let stats3 = evaluate(&pure_mcts, &random, 20, true);
    println!("胜率: {:.1}%\n", stats3.player1_winrate() * 100.0);

    println!("\n🔍 分析:");
    println!("  - 如果测试1很差 → 网络value有问题");
    println!("  - 如果测试2比测试1好很多 → rollout在补偿网络的错误");
    println!("  - 如果测试3=100% → baseline MCTS很强");
}
