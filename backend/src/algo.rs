use super::board::*;
use super::minimax::*;
use super::monte::*;
use core::panic;

pub trait GomokuSolver {
    fn best_move(input: &str, width: usize, height: usize) -> Move;
}

pub fn gomoku_solve(input: &str, algo_type: &str, width: usize, height: usize) -> Move {
    match algo_type {
        "minimax" => MiniMax::best_move(input, width, height),
        "monte_carlo" => MonteCarlo::best_move(input, width, height),
        "alphazero" => {
            #[cfg(feature = "alphazero")]
            {
                use crate::alphazero_solver::AdaptiveAlphaZeroSolver;
                use std::path::Path;
                let model_path = Path::new("../data/az_strong_converted.pt");
                if model_path.exists() {
                    match AdaptiveAlphaZeroSolver::from_file(model_path, 32, 2, 10, 50) {
                        Ok(solver) => {
                            let board = Board::new(input.to_string(), width, height);
                            let player = board.next_player();
                            if let Some((x, y)) = solver.solve(&board, player) {
                                return Move::new(x as usize, y as usize, 0, 0);
                            } else {
                                panic!("AlphaZero solver failed to find a move");
                            }
                        }
                        Err(_) => {
                            panic!("Failed to load AlphaZero model from {:?}", model_path);
                        }
                    }
                } else {
                    panic!("AlphaZero model file not found at {:?}", model_path);
                }
            }
            #[cfg(not(feature = "alphazero"))]
            {
                panic!("AlphaZero feature not enabled");
            }
        }
        _ => panic!("invalid algo type"),
    }
}
