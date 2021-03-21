use core::panic;

use super::board::*;
use super::minimax::*;
use super::monte::*;

pub trait GomokuSolver {
    fn best_move(input: &str) -> Move;
}

pub fn gomoku_solve(input: &str, algo_type: &str) -> Move {
    match algo_type {
        "minimax" => MiniMax::best_move(input),
        "monte_carlo" => MonteCarlo::best_move(input),
        _ => panic!("invalid algo type"),
    }
}
