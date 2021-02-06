#![allow(dead_code)]

use super::board::*;
use super::utils;
use rand::prelude::*;

#[derive(Debug, Copy, Clone)]
pub struct Move {
    x: usize,
    y: usize,
    score: i32,
}

impl Move {
    pub fn new(x: usize, y: usize, score: i32) -> Self {
        Self {
            x: x,
            y: y,
            score: score,
        }
    }
}

pub struct Runner {
    player: u8,
    depth: i32,
    pub gen_move_count: u32,
    pub eval_node: u32,
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

    // The naive minimax algorithm
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

    // 3 -->              -
    //     2 -->          +
    //          1 -->     -
    //              0 --> +
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
        let mut max_score = std::i32::MIN / 2;
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
                let score = -score;
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

    pub fn gen_ordered_moves(&mut self, board: &mut Board, player: u8) -> Vec<Move> {
        let mut moves = vec![];
        for i in 0..board.height {
            for j in 0..board.width {
                if board.get(i as i32, j as i32) != Some(0) || board.is_remote_cell(i, j) {
                    continue;
                }
                board.place(i, j, player);
                let score = board.eval(player);
                board.place(i, j, 0);
                moves.push(Move::new(i, j, score));
            }
        }
        moves.sort_by(|a, b| b.score.cmp(&a.score));
        moves
    }

    pub fn run_heuristic(
        &mut self,
        board: &mut Board,
        player: u8,
        depth: i32,
    ) -> (i32, usize, usize) {
        self.gen_move_heuristic(board, player, depth, std::i32::MIN / 2, std::i32::MAX / 2)
    }
    /* The minimax algorithm with alpha-beta tunning
     */
    fn gen_move_heuristic(
        &mut self,
        board: &mut Board,
        player: u8,
        depth: i32,
        alpha: i32,
        beta: i32,
    ) -> (i32, usize, usize) {
        self.eval_node += 1;
        if depth <= 0 {
            return (board.eval(player), 0, 0);
        }
        let mut max_score = std::i32::MIN;
        let mut move_x = 0;
        let mut move_y = 0;
        let mut cur_alpha = alpha;
        let dead_score = 100000;
        let threatening_score = 3500;
        let mut block_move = None;
        let mut best_moves: Vec<Move> = vec![];
        let candidates = self.gen_ordered_moves(board, player);
        if candidates.len() == 1 || candidates[0].score >= dead_score {
            return (candidates[0].score, candidates[0].x, candidates[0].y);
        }

        let opponent_candidates = self.gen_ordered_moves(board, utils::opponent(player));
        // If there are more than 2 threatening choices for opponent, we must lose the game
        // Anyway, try to block the first threatening choice
        if opponent_candidates.len() >= 1 && opponent_candidates[0].score >= threatening_score {
            block_move = Some(opponent_candidates[0]);
        }

        for i in 0..candidates.len() {
            let mut mv = candidates[i];
            board.place(mv.x, mv.y, player);
            let mut opponent_score = 0;
            if depth >= 1 {
                let (s, _, _) = self.gen_move_heuristic(
                    board,
                    utils::opponent(player),
                    -beta,
                    -cur_alpha + mv.score,
                    depth - 1,
                );
                opponent_score = s;
            }
            board.place(mv.x, mv.y, 0);
            mv.score -= opponent_score;
            if mv.score > max_score {
                println!("opponent_score: {}", opponent_score);
                max_score = mv.score;
                move_x = mv.x;
                move_y = mv.y;
                best_moves.clear();
                best_moves.push(mv);
            } else if mv.score == max_score {
                best_moves.push(mv);
            }
            cur_alpha = std::cmp::max(cur_alpha, max_score);
            if cur_alpha > beta {
                break;
            }
        }
        if block_move.is_some() && max_score <= block_move.unwrap().score {
            move_x = block_move.unwrap().x;
            move_y = block_move.unwrap().y;
        } else if best_moves.len() > 1 {
            //choose a random step
            let mut rng = thread_rng();
            let idx: usize = rng.gen_range(0, best_moves.len());
            let mv = best_moves[idx];
            move_x = mv.x;
            move_y = mv.y;
            println!("random choose one: {} from {}", idx, best_moves.len());
        }
        println!("block move: {:?}   max_score: {}", block_move, max_score);
        (max_score, move_x, move_y)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn make_empty_board() -> Board {
        let mut res = String::from("");
        for _ in 0..(7 * 7) {
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
