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

#[wasm_bindgen]
pub fn gomoku_solve(input: String, algo_type: String) -> String {
    let board = input.clone();
    let res = control::solve_it(&board, &algo_type);
    res.into()
}
