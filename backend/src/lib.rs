// The wasm-pack uses wasm-bindgen to build and generate JavaScript binding file.
// Import the wasm-bindgen crate.
use wasm_bindgen::prelude::*;
pub mod algo;
pub mod board;
mod control;
pub mod game_record;
pub mod minimax;
pub mod monte;
pub mod self_play;
mod utils;

// AlphaZero 相关模块（需要 alphazero feature）
pub mod alphazero_mcts;
pub mod alphazero_net;
pub mod alphazero_solver;
pub mod alphazero_trainer;

#[wasm_bindgen]
pub fn gomoku_solve(input: String, algo_type: String, width: usize, height: usize) -> String {
    let board = input.clone();
    let res = control::solve_it(&board, &algo_type, width, height);
    res.into()
}
