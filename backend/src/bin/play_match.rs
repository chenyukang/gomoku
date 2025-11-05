// AlphaZero vs Monte Carlo 对弈程序
#![cfg(feature = "alphazero")]

use gomoku::alphazero_solver::AlphaZeroSolver;
use gomoku::board::Board;
use gomoku::monte::MonteCarlo;
use std::env;
use yansi::Paint;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!(
            "Usage: {} <model_path> [num_games] [mcts_simulations]",
            args[0]
        );
        println!("Example: {} data/test_model.pt 10 1000", args[0]);
        return;
    }

    let model_path = &args[1];
    let num_games = if args.len() > 2 {
        args[2].parse().unwrap_or(10)
    } else {
        10
    };
    let mc_simulations = if args.len() > 3 {
        args[3].parse().unwrap_or(1000)
    } else {
        1000
    };

    println!("🎮 AlphaZero vs Monte Carlo Tournament\n");
    println!("Loading AlphaZero model: {}", model_path);

    // 加载 AlphaZero 模型
    // Match the training architecture defaults (128 filters, 6 blocks) and stronger MCTS sims
    let alphazero = match AlphaZeroSolver::from_file(model_path, 128, 6, 200) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            return;
        }
    };

    println!("✅ Model loaded");
    println!("Monte Carlo simulations: {}", mc_simulations);
    println!("Number of games: {}\n", num_games);

    let mut az_wins_as_first = 0;
    let mut az_wins_as_second = 0;
    let mut mc_wins_as_first = 0;
    let mut mc_wins_as_second = 0;
    let mut draws = 0;

    println!("{}", "=".repeat(60));
    println!("Starting tournament...\n");

    for game in 0..num_games {
        let az_is_first = game % 2 == 0;

        println!(
            "Game {}/{}: {} vs {}",
            game + 1,
            num_games,
            if az_is_first {
                "AlphaZero"
            } else {
                "MonteCarlo"
            },
            if az_is_first {
                "MonteCarlo"
            } else {
                "AlphaZero"
            }
        );

        let mut board = Board::new_default();
        let mut current_player = 1u8;
        let mut move_count = 0;

        loop {
            move_count += 1;

            // 决定使用哪个算法
            let next_move =
                if (current_player == 1 && az_is_first) || (current_player == 2 && !az_is_first) {
                    // AlphaZero
                    alphazero.solve(&board, current_player)
                } else {
                    // Monte Carlo - 需要为每次走法创建新的 MonteCarlo 实例
                    let mut mc = MonteCarlo::new(board.clone(), current_player, mc_simulations);
                    let mv = mc.search_move();
                    Some((mv.x as i32, mv.y as i32))
                };

            if let Some((x, y)) = next_move {
                let player_name = if (current_player == 1 && az_is_first)
                    || (current_player == 2 && !az_is_first)
                {
                    "AlphaZero"
                } else {
                    "MonteCarlo"
                };

                board.place(x as usize, y as usize, current_player);

                println!(
                    "  Move {}: {} (Player {}) -> ({}, {}) {}",
                    move_count,
                    player_name,
                    current_player,
                    x,
                    y,
                    Paint::red("←")
                );

                // 使用 Board 的 print 方法，它会高亮最后一步（大写字母）
                board.print();
                println!();

                // 检查是否获胜
                if let Some(winner) = board.any_winner() {
                    let winner_name =
                        if (winner == 1 && az_is_first) || (winner == 2 && !az_is_first) {
                            "AlphaZero"
                        } else {
                            "MonteCarlo"
                        };

                    println!(
                        "  🏆 Winner: {} (Player {}) in {} moves",
                        winner_name, winner, move_count
                    );

                    // 统计胜利
                    if winner_name == "AlphaZero" {
                        if az_is_first {
                            az_wins_as_first += 1;
                        } else {
                            az_wins_as_second += 1;
                        }
                    } else {
                        if !az_is_first {
                            mc_wins_as_first += 1;
                        } else {
                            mc_wins_as_second += 1;
                        }
                    }
                    break;
                }

                // 切换玩家
                current_player = if current_player == 1 { 2 } else { 1 };

                // 防止无限循环
                if move_count > 225 {
                    println!("  ⚖️  Draw (board full)");
                    draws += 1;
                    break;
                }
            } else {
                println!("  ⚖️  Draw (no valid moves)");
                draws += 1;
                break;
            }
        }
    }

    // 打印统计结果
    println!("\n{}", "=".repeat(60));
    println!("📊 Tournament Results\n");

    let total_az_wins = az_wins_as_first + az_wins_as_second;
    let total_mc_wins = mc_wins_as_first + mc_wins_as_second;
    let total = num_games;

    println!("AlphaZero:");
    println!(
        "  Total wins: {}/{} ({:.1}%)",
        total_az_wins,
        total,
        total_az_wins as f64 / total as f64 * 100.0
    );
    println!("  Wins as first player: {}", az_wins_as_first);
    println!("  Wins as second player: {}", az_wins_as_second);

    println!("\nMonte Carlo:");
    println!(
        "  Total wins: {}/{} ({:.1}%)",
        total_mc_wins,
        total,
        total_mc_wins as f64 / total as f64 * 100.0
    );
    println!("  Wins as first player: {}", mc_wins_as_first);
    println!("  Wins as second player: {}", mc_wins_as_second);

    println!("\nDraws: {}", draws);

    println!("\n{}", "=".repeat(60));

    if total_az_wins > total_mc_wins {
        println!("🎉 AlphaZero wins the tournament!");
    } else if total_mc_wins > total_az_wins {
        println!("🎉 Monte Carlo wins the tournament!");
    } else {
        println!("🤝 Tournament tied!");
    }
}
