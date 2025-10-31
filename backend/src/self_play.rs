// 自我对弈模块 - 用于生成训练数据
use super::algo::gomoku_solve;
use super::board::{Board, Move};
use super::game_record::{GameRecord, GameState};

#[cfg(feature = "random")]
use rand::Rng;

pub struct SelfPlay {
    max_steps: usize,
    verbose: bool,
    random_opening_steps: usize,
}

impl SelfPlay {
    pub fn new(max_steps: usize, verbose: bool) -> Self {
        Self {
            max_steps,
            verbose,
            random_opening_steps: 0,
        }
    }

    /// 创建带随机开局的自我对弈（增加多样性）
    pub fn new_with_random_opening(max_steps: usize, verbose: bool, opening_steps: usize) -> Self {
        Self {
            max_steps,
            verbose,
            random_opening_steps: opening_steps.min(3), // 最多3步
        }
    }

    /// 随机生成开局棋形（不指定玩家，只是位置）
    /// 返回格子位置列表，调用者决定谁下哪个子
    fn generate_random_opening_positions(&self) -> Vec<(usize, usize)> {
        let mut positions = Vec::new();

        cfg_if::cfg_if! {
            if #[cfg(feature = "random")] {
                if self.random_opening_steps == 0 {
                    return positions;
                }

                let mut rng = rand::thread_rng();
                let center: usize = 7; // 15x15 棋盘的中心

                // 第一步：在中心区域随机选择
                let offset: usize = 2; // 中心 ±2 范围
                let x1 = center.saturating_sub(offset) + rng.gen_range(0..=offset * 2);
                let y1 = center.saturating_sub(offset) + rng.gen_range(0..=offset * 2);

                positions.push((x1, y1));

                if self.random_opening_steps >= 2 {
                    // 第二步：在第一步附近
                    let nearby_range = 2;
                    for _ in 0..10 { // 最多尝试10次
                        let x2 = x1.saturating_sub(nearby_range) + rng.gen_range(0..=nearby_range * 2);
                        let y2 = y1.saturating_sub(nearby_range) + rng.gen_range(0..=nearby_range * 2);

                        // 确保不重复且在棋盘内
                        if x2 < 15 && y2 < 15 && (x2, y2) != (x1, y1) {
                            positions.push((x2, y2));
                            break;
                        }
                    }
                }

                if self.random_opening_steps >= 3 && positions.len() >= 2 {
                    // 第三步：在前两步附近
                    let (last_x, last_y) = positions[positions.len() - 1];
                    let nearby_range = 2;

                    for _ in 0..10 { // 最多尝试10次
                        let x3 = last_x.saturating_sub(nearby_range) + rng.gen_range(0..=nearby_range * 2);
                        let y3 = last_y.saturating_sub(nearby_range) + rng.gen_range(0..=nearby_range * 2);

                        if x3 < 15 && y3 < 15 && !positions.contains(&(x3, y3)) {
                            positions.push((x3, y3));
                            break;
                        }
                    }
                }
            }
        }

        positions
    }

    /// 运行一局游戏: algo1 vs algo2
    /// game_index 用于决定：
    /// - 偶数：algo1 是 Player 1（先手）
    /// - 奇数：algo2 是 Player 1（先手）
    pub fn play_game(&self, algo1: &str, algo2: &str, game_index: usize) -> GameRecord {
        let mut board = Board::new_default();

        // 决定谁是 Player 1（先手）
        let (player1_algo, player2_algo) = if game_index % 2 == 0 {
            (algo1, algo2)
        } else {
            (algo2, algo1)
        };

        let mut record = GameRecord::new(player1_algo.to_string(), player2_algo.to_string());

        // 生成随机开局位置（不指定玩家）
        let opening_positions = self.generate_random_opening_positions();

        if self.verbose && !opening_positions.is_empty() {
            println!("🎲 Random opening positions: {:?}", opening_positions);
        }

        // 将位置分配给 Player 1 和 Player 2
        for (i, (x, y)) in opening_positions.iter().enumerate() {
            let player = if i % 2 == 0 { 1u8 } else { 2u8 };
            board.place(*x, *y, player);

            let state = GameState {
                board: board.to_string(),
                player,
                move_x: *x,
                move_y: *y,
                eval_score: 0,
                step: i + 1,
            };
            record.add_state(state);

            if self.verbose {
                let algo = if player == 1 {
                    player1_algo
                } else {
                    player2_algo
                };
                println!(
                    "  Step {}: Player {} ({}) -> ({}, {})",
                    i + 1,
                    player,
                    algo,
                    x,
                    y
                );
            }
        }

        // 确定开局后的当前玩家
        let mut current_player = if opening_positions.len() % 2 == 0 {
            1u8
        } else {
            2u8
        };

        let start_step = opening_positions.len();
        for step in start_step..self.max_steps {
            if self.verbose {
                println!("\n=== Step {} ===", step + 1);
                println!("Player: {}", current_player);
            }

            // 选择算法（根据当前是 Player 1 还是 Player 2）
            let algo = if current_player == 1 {
                player1_algo
            } else {
                player2_algo
            };

            // 获取最佳落子
            let board_str = board.to_string();
            // 获取最佳落子
            let best_move = gomoku_solve(&board_str, algo);

            if self.verbose {
                println!(
                    "Best move: ({}, {}), score: {}",
                    best_move.x, best_move.y, best_move.score
                );
            }

            if best_move.x == 0 && best_move.y == 0 && best_move.score == 0 {
                // 无法继续，平局
                if self.verbose {
                    println!("No valid moves, game ends in draw");
                }
                break;
            }

            // 记录当前状态
            let state = GameState {
                board: board.to_string(),
                player: current_player,
                move_x: best_move.x,
                move_y: best_move.y,
                eval_score: best_move.score,
                step: step + 1,
            };
            record.add_state(state);

            // 执行落子
            board.place(best_move.x, best_move.y, current_player);

            if self.verbose {
                println!(
                    "Move: ({}, {}), Score: {}",
                    best_move.x, best_move.y, best_move.score
                );
                self.print_board(&board);
            }

            // 检查是否有赢家
            if let Some(winner) = board.any_winner() {
                record.set_winner(Some(winner));
                if self.verbose {
                    println!("\n🎉 Player {} wins!", winner);
                }
                break;
            }

            // 切换玩家
            current_player = if current_player == 1 { 2 } else { 1 };
        }

        if self.verbose {
            println!("\n{}", record.get_stats());
        }

        record
    }

    /// 批量自我对弈
    /// 策略：每个随机开局会被双方各玩一遍（一次先手，一次后手）
    pub fn play_multiple_games(
        &self,
        num_games: usize,
        algo1: &str,
        algo2: &str,
    ) -> Vec<GameRecord> {
        let mut records = Vec::new();

        println!("🎮 Starting {} games: {} vs {}", num_games, algo1, algo2);
        if self.random_opening_steps > 0 {
            println!(
                "   (Random opening: {} steps, each position played by both sides)",
                self.random_opening_steps
            );
        } else {
            println!("   (Alternating first player for fair evaluation)");
        }

        for i in 0..num_games {
            if !self.verbose {
                print!("\rProgress: {}/{}", i + 1, num_games);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            } else {
                println!("\n{}", "=".repeat(60));
                println!("Game {}/{}", i + 1, num_games);
                println!("{}", "=".repeat(60));
            }

            // 轮流先手：偶数局 algo1 是 Player 1，奇数局 algo2 是 Player 1
            let (player1, player2) = if i % 2 == 0 {
                (algo1, algo2)
            } else {
                (algo2, algo1)
            };

            if self.verbose {
                println!("Player 1 (first): {}", player1);
                println!("Player 2 (second): {}", player2);
            }

            let record = self.play_game(algo1, algo2, i);
            records.push(record);
        }

        if !self.verbose {
            println!(); // 换行
        }

        println!("✅ Completed {} games", num_games);
        records
    }

    /// 打印棋盘 (简化版)
    fn print_board(&self, board: &Board) {
        println!("\n   0 1 2 3 4 5 6 7 8 9 A B C D E");
        for i in 0..board.height {
            print!("{:2} ", i);
            for j in 0..board.width {
                let c = match board.get(i as i32, j as i32) {
                    Some(0) => '.',
                    Some(1) => 'X',
                    Some(2) => 'O',
                    _ => '?',
                };
                print!("{} ", c);
            }
            println!();
        }
    }
}

/// 锦标赛模式 - 让多个算法互相对战
pub struct Tournament {
    algorithms: Vec<String>,
    games_per_pair: usize,
}

impl Tournament {
    pub fn new(algorithms: Vec<String>, games_per_pair: usize) -> Self {
        Self {
            algorithms,
            games_per_pair,
        }
    }

    pub fn run(&self) -> Vec<GameRecord> {
        let mut all_records = Vec::new();
        let self_play = SelfPlay::new(300, false);

        println!("\n🏆 Tournament Mode");
        println!("Algorithms: {:?}", self.algorithms);
        println!("Games per pair: {}\n", self.games_per_pair);

        for i in 0..self.algorithms.len() {
            for j in 0..self.algorithms.len() {
                if i == j {
                    continue; // 跳过自己对自己
                }

                let algo1 = &self.algorithms[i];
                let algo2 = &self.algorithms[j];

                println!("\n📊 {} vs {}", algo1, algo2);
                let records = self_play.play_multiple_games(self.games_per_pair, algo1, algo2);
                all_records.extend(records);
            }
        }

        self.print_tournament_stats(&all_records);
        all_records
    }

    fn print_tournament_stats(&self, records: &[GameRecord]) {
        println!("\n{}", "=".repeat(60));
        println!("🏆 Tournament Results");
        println!("{}", "=".repeat(60));

        for algo in &self.algorithms {
            let wins = records
                .iter()
                .filter(|r| {
                    (r.algo_player1 == *algo && r.winner == Some(1))
                        || (r.algo_player2 == *algo && r.winner == Some(2))
                })
                .count();

            let losses = records
                .iter()
                .filter(|r| {
                    (r.algo_player1 == *algo && r.winner == Some(2))
                        || (r.algo_player2 == *algo && r.winner == Some(1))
                })
                .count();

            let total = records
                .iter()
                .filter(|r| r.algo_player1 == *algo || r.algo_player2 == *algo)
                .count();

            let win_rate = if total > 0 {
                wins as f32 / total as f32 * 100.0
            } else {
                0.0
            };

            println!(
                "{:15} - Wins: {:3}, Losses: {:3}, Win Rate: {:.1}%",
                algo, wins, losses, win_rate
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_play() {
        let self_play = SelfPlay::new(100, false);
        let record = self_play.play_game("minimax", "minimax", 0);
        assert!(record.total_steps > 0);
    }
}
