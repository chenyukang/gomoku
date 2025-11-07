// 游戏记录模块 - 用于收集训练数据
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub board: String,   // 棋盘状态 (15x15 = 225 chars)
    pub player: u8,      // 当前玩家 (1 或 2)
    pub move_x: usize,   // 下的位置 x
    pub move_y: usize,   // 下的位置 y
    pub eval_score: i32, // 评估分数
    pub step: usize,     // 第几步
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameRecord {
    pub states: Vec<GameState>,
    pub winner: Option<u8>, // 胜者 (1, 2, 或 None 表示平局)
    pub total_steps: usize,
    pub algo_player1: String, // Player 1 使用的算法
    pub algo_player2: String, // Player 2 使用的算法
    pub timestamp: String,
}

impl GameRecord {
    pub fn new(algo1: String, algo2: String) -> Self {
        Self {
            states: Vec::new(),
            winner: None,
            total_steps: 0,
            algo_player1: algo1,
            algo_player2: algo2,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn add_state(&mut self, state: GameState) {
        self.states.push(state);
        self.total_steps += 1;
    }

    pub fn set_winner(&mut self, winner: Option<u8>) {
        self.winner = winner;
    }

    /// 保存游戏记录到 JSON 文件
    pub fn save_to_file(&self, filename: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)?;
        writeln!(file, "{}", json)?;
        Ok(())
    }

    /// 保存为 CSV 格式 (适合机器学习)
    pub fn save_to_csv(&self, filename: &str) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)?;

        // 如果是新文件，写入表头
        if file.metadata()?.len() == 0 {
            writeln!(
                file,
                "board,player,move_x,move_y,eval_score,step,winner,final_reward"
            )?;
        }

        // 计算每一步的奖励
        for state in &self.states {
            let reward = self.calculate_reward(state.player);
            writeln!(
                file,
                "{},{},{},{},{},{},{},{}",
                state.board,
                state.player,
                state.move_x,
                state.move_y,
                state.eval_score,
                state.step,
                self.winner.unwrap_or(0),
                reward
            )?;
        }
        Ok(())
    }

    /// 计算奖励：赢 = 1.0, 输 = -1.0, 平局 = 0.0
    fn calculate_reward(&self, player: u8) -> f32 {
        match self.winner {
            Some(w) if w == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> String {
        format!(
            "Game Stats:\n\
             - Algorithm: {} vs {}\n\
             - Total steps: {}\n\
             - Winner: {:?}\n\
             - Timestamp: {}",
            self.algo_player1, self.algo_player2, self.total_steps, self.winner, self.timestamp
        )
    }
}

/// 数据集管理器
pub struct DatasetManager {
    games: Vec<GameRecord>,
}

impl DatasetManager {
    pub fn new() -> Self {
        Self { games: Vec::new() }
    }

    pub fn add_game(&mut self, game: GameRecord) {
        self.games.push(game);
    }

    /// 保存整个数据集
    pub fn save_dataset(&self, json_file: &str, csv_file: &str) -> std::io::Result<()> {
        for game in &self.games {
            game.save_to_file(json_file)?;
            game.save_to_csv(csv_file)?;
        }
        Ok(())
    }

    /// 统计信息
    pub fn print_stats(&self) {
        let total = self.games.len();
        if total == 0 {
            println!("No games recorded.");
            return;
        }

        let player1_wins = self.games.iter().filter(|g| g.winner == Some(1)).count();
        let player2_wins = self.games.iter().filter(|g| g.winner == Some(2)).count();
        let draws = self.games.iter().filter(|g| g.winner.is_none()).count();

        println!("Dataset Statistics:");
        println!("  Total games: {}", total);
        println!(
            "  Player 1 wins: {} ({:.1}%)",
            player1_wins,
            player1_wins as f32 / total as f32 * 100.0
        );
        println!(
            "  Player 2 wins: {} ({:.1}%)",
            player2_wins,
            player2_wins as f32 / total as f32 * 100.0
        );
        println!(
            "  Draws: {} ({:.1}%)",
            draws,
            draws as f32 / total as f32 * 100.0
        );

        // 按算法统计
        self.print_algorithm_stats();
    }

    /// 按算法统计胜率
    fn print_algorithm_stats(&self) {
        use std::collections::HashMap;

        let mut algo_stats: HashMap<String, (usize, usize, usize)> = HashMap::new();

        for game in &self.games {
            let algo1 = &game.algo_player1;
            let algo2 = &game.algo_player2;

            // 更新 algo1 的统计
            let stats1 = algo_stats.entry(algo1.clone()).or_insert((0, 0, 0));
            match game.winner {
                Some(1) => stats1.0 += 1, // 胜
                Some(2) => stats1.1 += 1, // 负
                None => stats1.2 += 1,    // 平
                _ => {}
            }

            // 更新 algo2 的统计
            let stats2 = algo_stats.entry(algo2.clone()).or_insert((0, 0, 0));
            match game.winner {
                Some(2) => stats2.0 += 1, // 胜
                Some(1) => stats2.1 += 1, // 负
                None => stats2.2 += 1,    // 平
                _ => {}
            }
        }

        println!("\nAlgorithm Performance:");
        for (algo, (wins, losses, draws)) in algo_stats.iter() {
            let total = wins + losses + draws;
            let win_rate = if total > 0 {
                *wins as f32 / total as f32 * 100.0
            } else {
                0.0
            };
            println!(
                "  {}: {} wins, {} losses, {} draws (win rate: {:.1}%)",
                algo, wins, losses, draws, win_rate
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_record() {
        let mut record = GameRecord::new("minimax".to_string(), "monte_carlo".to_string());

        let state = GameState {
            board: ".".repeat(225),
            player: 1,
            move_x: 7,
            move_y: 7,
            eval_score: 100,
            step: 1,
        };

        record.add_state(state);
        record.set_winner(Some(1));

        assert_eq!(record.total_steps, 1);
        assert_eq!(record.winner, Some(1));
    }

    #[test]
    fn test_calculate_reward() {
        let mut record = GameRecord::new("test".to_string(), "test".to_string());
        record.set_winner(Some(1));

        assert_eq!(record.calculate_reward(1), 1.0);
        assert_eq!(record.calculate_reward(2), -1.0);
    }
}
