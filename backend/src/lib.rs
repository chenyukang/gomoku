// The wasm-pack uses wasm-bindgen to build and generate JavaScript binding file.
// Import the wasm-bindgen crate.
use wasm_bindgen::prelude::*;
mod algo;
pub mod board;
mod control;
pub mod monte;
mod utils;

#[wasm_bindgen]
pub fn gomoku_solve(input: String, player: i32) -> String {
    let board = input.clone();
    let res = control::solve_it(board.as_str(), player as u8, 0);
    res.into()
}
