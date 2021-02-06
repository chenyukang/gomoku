#![allow(dead_code)]

use super::board::*;
use super::utils;

pub fn gen_move(board: &mut Board, player: u8, depth: i32) -> (i32, usize, usize) {
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
            let (score2, _, _) = gen_move(board, utils::opponent(player), depth - 1);
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

pub fn gen_move_negamax(board: &mut Board, player: u8, depth: i32) -> (i32, usize, usize) {
    if depth <= 0 {
        let flag = if player == 1 { 1 } else { -1 };
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
            let (score, _, _) = gen_move(board, utils::opponent(player), depth - 1);
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

#[cfg(test)]
mod tests {

    #[test]
    fn test_algo() {
        assert_eq!(1, 1);
    }
}
