// Connect4 游戏逻辑
// 7列 x 6行，连续4个棋子获胜

#![cfg(feature = "alphazero")]

#[derive(Clone, Debug, PartialEq)]
pub struct Connect4 {
    board: Vec<Vec<u8>>, // 0=空, 1=玩家1, 2=玩家2
    current_player: u8,
    winner: Option<u8>,
}

impl Connect4 {
    pub fn new() -> Self {
        Self {
            board: vec![vec![0; 7]; 6], // 6行7列
            current_player: 1,
            winner: None,
        }
    }

    /// 在指定列落子
    pub fn play(&mut self, col: usize) -> Result<(), String> {
        if col >= 7 {
            return Err("列号超出范围".to_string());
        }

        if self.winner.is_some() {
            return Err("游戏已结束".to_string());
        }

        // 从底部往上找第一个空位
        for row in (0..6).rev() {
            if self.board[row][col] == 0 {
                self.board[row][col] = self.current_player;
                self.check_winner(row, col);
                self.current_player = if self.current_player == 1 { 2 } else { 1 };
                return Ok(());
            }
        }

        Err("该列已满".to_string())
    }

    /// 获取所有合法的落子位置
    pub fn legal_moves(&self) -> Vec<usize> {
        if self.winner.is_some() {
            return vec![];
        }
        (0..7).filter(|&col| self.board[0][col] == 0).collect()
    }

    /// 检查是否有玩家获胜
    fn check_winner(&mut self, row: usize, col: usize) {
        let player = self.board[row][col];

        // 检查水平方向
        if self.check_line(row, col, 0, 1, player) {
            self.winner = Some(player);
            return;
        }

        // 检查垂直方向
        if self.check_line(row, col, 1, 0, player) {
            self.winner = Some(player);
            return;
        }

        // 检查两个对角线方向
        if self.check_line(row, col, 1, 1, player) || self.check_line(row, col, 1, -1, player) {
            self.winner = Some(player);
            return;
        }

        // 检查是否平局
        if self.legal_moves().is_empty() {
            self.winner = Some(0); // 0表示平局
        }
    }

    /// 检查从指定位置开始，在给定方向上是否有4连
    fn check_line(&self, row: usize, col: usize, dr: isize, dc: isize, player: u8) -> bool {
        let mut count = 1; // 当前位置已经算1个

        // 正方向计数
        let mut r = row as isize + dr;
        let mut c = col as isize + dc;
        while r >= 0 && r < 6 && c >= 0 && c < 7 && self.board[r as usize][c as usize] == player {
            count += 1;
            r += dr;
            c += dc;
        }

        // 反方向计数
        let mut r = row as isize - dr;
        let mut c = col as isize - dc;
        while r >= 0 && r < 6 && c >= 0 && c < 7 && self.board[r as usize][c as usize] == player {
            count += 1;
            r -= dr;
            c -= dc;
        }

        count >= 4
    }

    /// 获取当前玩家
    pub fn current_player(&self) -> u8 {
        self.current_player
    }

    /// 获取获胜者（None=游戏未结束, Some(0)=平局, Some(1/2)=玩家获胜）
    pub fn winner(&self) -> Option<u8> {
        self.winner
    }

    /// 获取棋盘状态
    pub fn board(&self) -> &Vec<Vec<u8>> {
        &self.board
    }

    /// 判断游戏是否结束
    pub fn is_game_over(&self) -> bool {
        self.winner.is_some()
    }

    /// 将棋盘转换为神经网络输入 (3, 6, 7)
    /// Channel 0: 当前玩家的棋子
    /// Channel 1: 对手的棋子
    /// Channel 2: 当前玩家标记（全1或全0）
    pub fn to_tensor(&self) -> Vec<f32> {
        let mut tensor = Vec::with_capacity(3 * 6 * 7);
        let opponent = if self.current_player == 1 { 2 } else { 1 };

        // Channel 0: 当前玩家
        for row in &self.board {
            for &cell in row {
                tensor.push(if cell == self.current_player {
                    1.0
                } else {
                    0.0
                });
            }
        }

        // Channel 1: 对手
        for row in &self.board {
            for &cell in row {
                tensor.push(if cell == opponent { 1.0 } else { 0.0 });
            }
        }

        // Channel 2: 当前玩家标记
        for _ in 0..42 {
            tensor.push(if self.current_player == 1 { 1.0 } else { 0.0 });
        }

        tensor
    }

    /// 打印棋盘（用于调试）
    pub fn print(&self) {
        println!("\n  0 1 2 3 4 5 6");
        for row in &self.board {
            print!(" |");
            for &cell in row {
                let ch = match cell {
                    0 => " ",
                    1 => "X",
                    2 => "O",
                    _ => "?",
                };
                print!("{}|", ch);
            }
            println!();
        }
        println!("  =============");
        if let Some(w) = self.winner {
            match w {
                0 => println!("平局!"),
                1 => println!("玩家1 (X) 获胜!"),
                2 => println!("玩家2 (O) 获胜!"),
                _ => {}
            }
        } else {
            println!(
                "当前玩家: {}",
                if self.current_player == 1 { "X" } else { "O" }
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let game = Connect4::new();
        assert_eq!(game.current_player(), 1);
        assert_eq!(game.winner(), None);
        assert_eq!(game.legal_moves().len(), 7);
    }

    #[test]
    fn test_play() {
        let mut game = Connect4::new();
        assert!(game.play(3).is_ok());
        assert_eq!(game.current_player(), 2);
        assert_eq!(game.board()[5][3], 1);
    }

    #[test]
    fn test_horizontal_win() {
        let mut game = Connect4::new();
        // 玩家1连下4个子
        for col in 0..4 {
            game.play(col).unwrap();
            if col < 3 {
                game.play(col).unwrap(); // 玩家2也下一个
            }
        }
        assert_eq!(game.winner(), Some(1));
    }

    #[test]
    fn test_vertical_win() {
        let mut game = Connect4::new();
        // 玩家1在列0连下4个
        game.play(0).unwrap(); // 玩家1
        game.play(1).unwrap(); // 玩家2
        game.play(0).unwrap(); // 玩家1
        game.play(1).unwrap(); // 玩家2
        game.play(0).unwrap(); // 玩家1
        game.play(1).unwrap(); // 玩家2
        game.play(0).unwrap(); // 玩家1 - 应该获胜
        assert_eq!(game.winner(), Some(1));
    }
}
