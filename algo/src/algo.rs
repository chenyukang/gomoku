#![allow(dead_code)]

use super::board::*;
use super::utils;

pub struct Runner {
    player: u8,
    depth: i32,
    gen_move_count: u32,
    eval_node: u32,
}

impl Runner {
    pub fn new(player: u8, depth: i32) -> Self {
        Self {
            player: player,
            depth: depth,
            gen_move_count: 0,
            eval_node: 0,
        }
    }
    pub fn gen_move(&mut self, board: &mut Board, player: u8, depth: i32) -> (i32, usize, usize) {
        self.eval_node += 1;
        if depth <= 0 {
            return (board.eval(player), 0, 0);
        }
        let mut max_score = std::i32::MIN;
        let mut move_x = 0;
        let mut move_y = 0;
        //println!("gen_move: {} {}", player, depth);
        for i in 0..board.height {
            for j in 0..board.width {
                if board.get(i as i32, j as i32) != Some(0) || board.is_remote_cell(i, j) {
                    continue;
                }
                board.place(i, j, player);
                let score1 = board.eval(player);
                let (score2, _, _) = self.gen_move(board, utils::opponent(player), depth - 1);
                if score1 - score2 > max_score {
                    max_score = score1 - score2;
                    move_x = i;
                    move_y = j;
                }
                board.place(i, j, 0);
            }
        }
        (max_score, move_x, move_y)
    }

    pub fn gen_move_negamax(
        &mut self,
        board: &mut Board,
        player: u8,
        depth: i32,
    ) -> (i32, usize, usize) {
        self.eval_node += 1;
        if depth <= 0 {
            let flag = if player == self.player { 1 } else { -1 };
            return (flag * board.eval(player), 0, 0);
        }
        let mut max_score = std::i32::MIN;
        let mut move_x = 0;
        let mut move_y = 0;
        //println!("gen_move: {} {}", player, depth);
        for i in 0..board.height {
            for j in 0..board.width {
                if board.get(i as i32, j as i32) != Some(0) || board.is_remote_cell(i, j) {
                    continue;
                }
                board.place(i, j, player);
                let (score, _, _) =
                    self.gen_move_negamax(board, utils::opponent(player), depth - 1);
                let score = -1 * score;
                if score > max_score {
                    max_score = score;
                    move_x = i;
                    move_y = j;
                }
                board.place(i, j, 0);
            }
        }
        (max_score, move_x, move_y)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn make_empty_board() -> Board {
        let mut res = String::from("");
        for _ in 0..(10 * 10) {
            res = res + "0";
        }
        Board::from(res)
    }

    #[allow(unused_assignments)]
    #[test]
    fn test_algo() {
        let mut board = make_empty_board();
        let mut winner = 0;
        loop {
            let mut runner1 = Runner::new(1, 2);
            let mut runner2 = Runner::new(2, 2);
            let (_, mv_x1, mv_y1) = runner1.gen_move(&mut board, 1, 2);
            board.place(mv_x1, mv_y1, 1);
            if let Some(w) = board.any_winner() {
                println!("winner1: {}", w);
                winner = w;
                break;
            }
            let (_, mv_x2, mv_y2) = runner2.gen_move_negamax(&mut board, 2, 3);
            if let Some(w) = board.any_winner() {
                println!("winner2: {}", w);
                winner = w;
                break;
            }
            board.place(mv_x2, mv_y2, 2);
        }
        println!(
            "winner: {}, empty_cells: {}",
            winner,
            board.empty_cells_count()
        );
        assert_eq!(winner, 1);
    }
}
