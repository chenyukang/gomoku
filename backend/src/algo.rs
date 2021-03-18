#![allow(dead_code)]
use super::board::*;
use super::utils::*;
use std::cmp::*;
use std::env;

pub struct Runner {
    player: u8,
    depth: i32,
    pub gen_move_count: u32,
    pub eval_node: u32,
    debug: bool,
}

impl Runner {
    pub fn new(player: u8, depth: i32) -> Self {
        let debug = match env::var("GOMOKU_DEBUG") {
            Ok(_) => true,
            _ => false,
        };
        Self {
            player: player,
            depth: depth,
            gen_move_count: 0,
            eval_node: 0,
            debug: debug,
        }
    }

    // The naive minimax algorithm
    pub fn gen_move(&mut self, board: &mut Board, player: u8, depth: i32) -> (i32, usize, usize) {
        self.eval_node += 1;
        if depth <= 0 {
            return (board.eval_all(player) as i32, 0, 0);
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
                let score1 = board.eval_pos(player, i, j);
                let (score2, _, _) = self.gen_move(board, cfg::opponent(player), depth - 1);
                if score1 as i32 - score2 as i32 > max_score {
                    max_score = score1 as i32 - score2 as i32;
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
                let (score, _, _) = self.gen_move_negamax(board, cfg::opponent(player), depth - 1);
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

    pub fn run_heuristic(&mut self, board: &mut Board, player: u8) -> (i32, usize, usize) {
        self.gen_move_heuristic(
            board,
            player,
            self.depth,
            std::i32::MIN / 2,
            std::i32::MAX / 2,
        )
    }
    /* The minimax algorithm with alpha-beta tunning
     * In Negamax implmentation
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
        let mut max_score = std::i32::MIN;
        let mut final_move = Move::new(0, 0, 0, 0);
        let mut cur_alpha = alpha;
        let dead_score = 100000;
        let mut block_move = None;
        let mut best_moves: Vec<Move> = vec![];
        let mut candidates = board.gen_ordered_moves(player);
        if depth == self.depth {
            println!("len: {}", candidates.len());
        }
        if candidates.len() == 0 {
            return (0, 0, 0);
        }
        if candidates.len() == 1 || candidates[0].score >= dead_score {
            return (candidates[0].score, candidates[0].x, candidates[0].y);
        }

        let opponent_candidates = board.gen_ordered_moves(cfg::opponent(player));
        // If there are more than 2 threatening choices for opponent, we must lose the game
        // Anyway, try to block the first threatening choice
        if opponent_candidates.len() >= 1 && opponent_candidates[0].is_dead_move() {
            block_move = opponent_candidates.first();
        }

        for i in 0..candidates.len() {
            let mut mv = candidates[i];
            board.place(mv.x, mv.y, player);
            let mut opponent_score = 0;
            if depth > 1 {
                let (s, _, _) = self.gen_move_heuristic(
                    board,
                    cfg::opponent(player),
                    depth - 1,
                    -beta,
                    -cur_alpha + mv.score,
                );
                opponent_score = s;
            }
            board.place(mv.x, mv.y, 0);
            if depth == self.depth && self.debug {
                println!("move: {:?} => oppo_score: {}", mv, opponent_score);
            }
            mv.score -= opponent_score;
            candidates[i].score = mv.score;
            if mv.score > max_score {
                //println!("opponent_score: {}", opponent_score);
                max_score = mv.score;
                final_move = mv;
                best_moves.clear();
                best_moves.push(mv);
            } else if mv.score == max_score {
                best_moves.push(mv);
            }
            cur_alpha = std::cmp::max(cur_alpha, max_score);
            if cur_alpha >= beta {
                break;
            }
        }
        if depth == self.depth && self.debug {
            for i in 0..candidates.len() {
                println!("possible move: {:?}", candidates[i]);
            }
            for i in 0..min(3, opponent_candidates.len()) {
                println!("opponent_move: {:?}", opponent_candidates[i]);
            }
            println!(
                "+Block move: {:?} \n {:?} depth:{} self.depth:{}",
                block_move, final_move, depth, self.depth,
            );
        }
        if depth == self.depth && block_move.is_some() && block_move.unwrap().is_dead_move() {
            println!("Use block move: {:?}", block_move);
            final_move = *block_move.unwrap();
            max_score = final_move.score;
            println!(
                "+Block move: {:?} \n {:?} depth:{} self.depth:{}",
                block_move, final_move, depth, self.depth,
            );
        } else if best_moves.len() > 1 {
            //choose by original score
            best_moves.sort_by(|a, b| b.original_score.cmp(&a.original_score));
            final_move = *(best_moves.first().unwrap());
        }
        let mut prev = String::from("");
        for _ in 0..(self.depth - depth) {
            prev += "---";
        }
        if depth == self.depth {
            println!(
                "Final move: {:?} depth:{} self.depth:{}, max_score: {}",
                final_move, depth, self.depth, max_score
            );
        }
        /* println!(
            "{}Player: {} depth:{} -> Final move: {} {:?}",
            prev, player, depth, max_score, final_move
        ); */
        (max_score, final_move.x, final_move.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_empty_board() -> Board {
        let mut res = String::from("");
        for _ in 0..(15 * 15) {
            res = res + "0";
        }
        Board::from(res)
    }

    #[test]
    fn test_algo_1() {
        let mut board = Board::new(
            String::from(
                "
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . o . + . . . . .
        . . . . . . + . o o o + . . .
        . . . . . + . o + + o . . . .
        . . . . . . o . o + + . . . .
        . . . . . . . o + o o . . . .
        . . . . . . + + o . + . . . .
        . . . . . . o . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        ",
            ),
            15,
            15,
        );

        let mut runner = Runner::new(2, 4);
        let (_, row, col) = runner.run_heuristic(&mut board, 2);
        board.place(row, col, 2);
        board.print();
        assert_eq!(row, 8);
        assert_eq!(col, 11);
    }

    #[test]
    fn test_algo_2() {
        let mut board = Board::new(
            String::from(
                "
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . o . . . . . . .
        . . . . . . . o + . . . . . .
        . . . . . . . + . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        ",
            ),
            15,
            15,
        );

        let mut runner = Runner::new(2, 4);
        let (score, row, col) = runner.run_heuristic(&mut board, 1);
        board.place(row, col, 2);
        board.print();
        assert_eq!(score, -30); //FIXME
    }

    #[test]
    fn test_algo_3() {
        let mut board = Board::new(
            String::from(
                "
            000000100000000
            000000200000000
            000021200000000
            002011210000000
            000112221000000
            002012120000000
            002111120100000
            000022112000000
            000000120000000
            000000200000000
            000000000000000
            000000000000000
            000000000000000
            000000000000000
            000000000000000",
            ),
            15,
            15,
        );

        let mut runner = Runner::new(2, 4);
        let (_, row, col) = runner.run_heuristic(&mut board, 1);
        assert_eq!(row == 4 || row == 7, true);
        assert_eq!(col, 2);
    }

    #[test]
    fn test_algo_4() {
        let mut board = Board::new(
            String::from(
                "
    000000000000000
    000000000000000
    000000000000000
    000000000000000
    000000002020000
    000020000102200
    000212221112100
    000021111211000
    000001212111100
    000000201220020
    000000000100000
    000000000020000
    000000000000000
    000000000000000
    000000000000000",
            ),
            15,
            15,
        );
        let mut runner = Runner::new(2, 4);
        let (_, row, col) = runner.run_heuristic(&mut board, 2);
        assert_eq!(row, 8);
        assert_eq!(col, 13);
    }

    #[test]
    fn test_algo_5() {
        let mut board = Board::new(
            String::from(
                "
                . . . . . . . . . . . . . . .
                . . . . . . . . . . . . . . .
                . . . . . . . . . . . . . . .
                . . . . . . . . . . . . . . .
                . . . . . . . . . . . . . . .
                . . . . . . . . . . . . . . .
                . . . . . . . . . . . . . . .
                . . . . . + o o . . . . . . .
                . . . . . + + o . . . . . . .
                . . . . . . o . . . . . . . .
                . . . . . . . . . . . . . . .
                . . . . . . . . . . . . . . .
                . . . . . . . . . . . . . . .
                . . . . . . . . . . . . . . .
                . . . . . . . . . . . . . . .
    ",
            ),
            15,
            15,
        );

        board.print();
        let mut runner = Runner::new(2, 4);
        let (_, row, col) = runner.run_heuristic(&mut board, 2);
        board.place(row, col, 2);
        board.print();
        assert_eq!(row, 6);
        assert_eq!(col, 5);
    }

    #[test]
    fn test_algo_6() {
        let mut board = Board::new(
            String::from(
                "
 . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . .
 . . . . . . . . + . . . . . .
 . . . . . . . . o . . . . . .
 . . . . . + o o o . . . . . .
 . . . . . . + + o . o . . . .
 . . . . . . + . . . . . . . .
 . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . .
    ",
            ),
            15,
            15,
        );

        let mut runner = Runner::new(2, 4);
        let (_, row, col) = runner.run_heuristic(&mut board, 2);
        assert_eq!(row, 5);
        assert_eq!(col, 7);
    }

    #[allow(unused_assignments)]
    #[test]
    fn test_algo_battle_self() {
        let mut board = make_empty_board();
        let mut winner = 0;
        board.place(7, 7, 1);
        loop {
            let mut runner1 = Runner::new(1, 4);
            let mut runner2 = Runner::new(2, 4);

            if board.empty_cells_count() == 0 {
                break;
            }

            //println!("left: {}", board.empty_cells_count());
            let (_, mv_x1, mv_y1) = runner1.run_heuristic(&mut board, 2);
            board.place(mv_x1, mv_y1, 2);
            if let Some(w) = board.any_winner() {
                println!("winner1: {}", w);
                winner = w;
                break;
            }
            let (_, mv_x2, mv_y2) = runner2.run_heuristic(&mut board, 1);
            if let Some(w) = board.any_winner() {
                println!("winner2: {}", w);
                winner = w;
                break;
            }
            board.place(mv_x2, mv_y2, 1);
        }
        println!(
            "winner: {}, empty_cells: {}",
            winner,
            board.empty_cells_count()
        );
        assert_eq!(winner, 2);
        assert_eq!(board.empty_cells_count(), 187);
    }
}
