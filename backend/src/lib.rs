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
pub mod utils;

// AlphaZero for Connect4
#[cfg(feature = "alphazero")]
pub mod alphazero_solver;
#[cfg(feature = "alphazero")]
pub mod az_eval;
#[cfg(feature = "alphazero")]
pub mod az_mcts;
#[cfg(feature = "alphazero")]
pub mod az_mcts_rollout;
#[cfg(feature = "alphazero")]
pub mod az_net;
#[cfg(feature = "alphazero")]
pub mod az_net_deep;
#[cfg(feature = "alphazero")]
pub mod az_resnet;
#[cfg(feature = "alphazero")]
pub mod az_trainer;
#[cfg(feature = "alphazero")]
pub mod connect4;

#[wasm_bindgen]
pub fn gomoku_solve(input: String, algo_type: String, width: usize, height: usize) -> String {
    let board = input.clone();
    let res = control::solve_it(&board, &algo_type, width, height);
    res.into()
}
