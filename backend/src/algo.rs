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
                // 使用 best_model.pt 而不是特定代数的模型
                let model_path = Path::new("../data/generations/gen_0004.pt");
                if model_path.exists() {
                    // 参数必须与训练时一致：128 filters, 6 blocks
                    // min_simulations: 50, max_simulations: 200 (自适应MCTS)
                    match AdaptiveAlphaZeroSolver::from_file(model_path, 128, 6, 50, 200) {
                        Ok(solver) => {
                            let board = Board::new(input.to_string(), width, height);
                            let player = board.next_player();
                            if let Some((x, y)) = solver.solve(&board, player) {
                                return Move::new(x as usize, y as usize, 0, 0);
                            } else {
                                panic!("AlphaZero solver failed to find a move");
                            }
                        }
                        Err(e) => {
                            panic!(
                                "Failed to load AlphaZero model from {:?} with err: {:?}",
                                model_path, e
                            );
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
