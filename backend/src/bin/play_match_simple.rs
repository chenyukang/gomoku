// 使用新训练的 AlphaZero 直接对弈（不从文件加载）
#![cfg(feature = "alphazero")]

use gomoku::alphazero_solver::AlphaZeroSolver;
use gomoku::board::Board;
use gomoku::monte::MonteCarlo;

fn main() {
    let num_games = 6;
    let mc_simulations = 500;
    let az_simulations = 100;

    println!("🎮 AlphaZero vs Monte Carlo Tournament\n");
    println!("Creating new AlphaZero model (untrained)...");

    // 创建新的 AlphaZero（未训练）
    let alphazero = AlphaZeroSolver::new(32, 2, az_simulations);

    println!(
        "✅ AlphaZero created (filters=32, blocks=2, sims={})",
        az_simulations
    );
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
                    // Monte Carlo
                    let mut mc = MonteCarlo::new(board.clone(), current_player, mc_simulations);
                    let mv = mc.search_move();
                    Some((mv.x as i32, mv.y as i32))
                };

            if let Some((x, y)) = next_move {
                board.place(x as usize, y as usize, current_player);

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

    println!("AlphaZero (untrained):");
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
    println!("Note: AlphaZero is untrained, so Monte Carlo should win most games.");
    println!("To improve AlphaZero, train it with more epochs and data!");

    if total_az_wins > total_mc_wins {
        println!("🎉 AlphaZero wins the tournament!");
    } else if total_mc_wins > total_az_wins {
        println!("🎉 Monte Carlo wins the tournament!");
    } else {
        println!("🤝 Tournament tied!");
    }
}
