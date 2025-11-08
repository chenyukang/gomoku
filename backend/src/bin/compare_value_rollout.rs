// 对比测试: 网络value vs Rollout

use gomoku::az_mcts::MCTS;
use gomoku::az_mcts_rollout::MCTSWithRollout;
use gomoku::az_net::Connect4Trainer;
use gomoku::connect4::Connect4;

fn main() {
    println!("🔬 对比测试: 网络value vs Rollout\n");

    let trainer = Connect4Trainer::new(64, 0.0003);

    println!("━━━ 测试1: MCTS+网络value vs 随机 ━━━\n");
    test_vs_random(&trainer, false);

    println!("\n━━━ 测试2: MCTS+Rollout vs 随机 ━━━\n");
    test_vs_random_rollout(&trainer, true);

    println!("\n━━━ 测试3: 纯MCTS(无网络) vs 随机 ━━━\n");
    test_pure_mcts_vs_random();
}

fn test_vs_random(trainer: &Connect4Trainer, _use_value: bool) {
    let mut wins = 0;

    for _ in 0..10 {
        let mut game = Connect4::new();

        while !game.is_game_over() {
            let action = if game.current_player() == 1 {
                let mut mcts = MCTS::new(200);
                mcts.search(&game, &trainer.net);
                mcts.select_action(0.0)
            } else {
                use rand::Rng;
                let legal = game.legal_moves();
                legal[rand::thread_rng().gen_range(0..legal.len())]
            };
            game.play(action).ok();
        }

        if game.winner() == Some(1) {
            wins += 1;
            print!("✓");
        } else {
            print!("✗");
        }
    }

    println!("\n胜率: {}/10 ({:.0}%)", wins, wins as f32 * 10.0);
}

fn test_vs_random_rollout(trainer: &Connect4Trainer, use_rollout: bool) {
    let mut wins = 0;

    for _ in 0..10 {
        let mut game = Connect4::new();

        while !game.is_game_over() {
            let action = if game.current_player() == 1 {
                let mut mcts = MCTSWithRollout::new(200, use_rollout);
                mcts.search(&game, &trainer.net);
                mcts.select_action(0.0)
            } else {
                use rand::Rng;
                let legal = game.legal_moves();
                legal[rand::thread_rng().gen_range(0..legal.len())]
            };
            game.play(action).ok();
        }

        if game.winner() == Some(1) {
            wins += 1;
            print!("✓");
        } else {
            print!("✗");
        }
    }

    println!("\n胜率: {}/10 ({:.0}%)", wins, wins as f32 * 10.0);
}

fn test_pure_mcts_vs_random() {
    use gomoku::az_eval::Player;

    let pure_mcts = Player::PureMCTS { simulations: 200 };
    let random = Player::Random;

    let mut wins = 0;
    for _ in 0..10 {
        let mut game = Connect4::new();

        while !game.is_game_over() {
            let action = if game.current_player() == 1 {
                pure_mcts.select_move(&game)
            } else {
                random.select_move(&game)
            };
            game.play(action).ok();
        }

        if game.winner() == Some(1) {
            wins += 1;
            print!("✓");
        } else {
            print!("✗");
        }
    }

    println!("\n胜率: {}/10 ({:.0}%)", wins, wins as f32 * 10.0);
}
