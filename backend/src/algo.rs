#[cfg(feature = "alphazero")]
use super::alphazero_solver::*;
use super::board::*;
use super::minimax::*;
use super::monte::*;
use core::panic;

pub trait GomokuSolver {
    fn best_move(input: &str, width: usize, height: usize) -> Move;
}

pub fn gomoku_solve(input: &str, algo_type: &str, width: usize, height: usize) -> Move {
    println!(
        "gomoku_solve: algo_type={}, width={}, height={}",
        algo_type, width, height
    );
    match algo_type {
        "minimax" => MiniMax::best_move(input, width, height),
        "monte_carlo" => MonteCarlo::best_move(input, width, height),
        #[cfg(feature = "alphazero")]
        "alphazero" => AlphaZero::best_move(input, width, height),
        _ => panic!("invalid algo type"),
    }
}
