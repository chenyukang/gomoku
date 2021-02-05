use super::board::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct BodyType {
    ai_player: u8,
    build: String,
    cc_0: String,
    cc_1: String,
    cpu_time: String,
    eval_count: u32,
    move_c: usize,
    move_r: usize,
    node_count: i32,
    num_threads: i32,
    pm_count: i32,
    search_depth: i32,
    winning_player: u8,
    score: i32,
}

#[derive(Serialize, Deserialize)]
struct ResponseType {
    message: String,
    result: BodyType,
}

pub fn gen_move(input: &str, player: u8) -> String {
    let mut board = Board::from(input.to_string());
    let mut winner = 0;
    let mut ans_score = 0;
    let mut ans_col = 0;
    let mut ans_row = 0;
    if let Some(w) = board.any_winner() {
        winner = w;
    } else {
        let (score, row, col) = board.gen_move(player, 2);
        board.place(row, col, player);
        if let Some(w) = board.any_winner() {
            println!("winner: {:?}", winner);
            winner = w;
        }
        ans_score = score;
        ans_row = row;
        ans_col = col;
    }
    let result = ResponseType {
        message: String::from("ok"),
        result: BodyType {
            ai_player: 2,
            build: String::from("Feb  4 2021 06:45:24"),
            cc_0: String::from("0"),
            cc_1: String::from("0"),
            cpu_time: String::from("1101"),
            eval_count: 1000,
            move_c: ans_col,
            move_r: ans_row,
            node_count: 100,
            num_threads: 1,
            pm_count: 1,
            search_depth: 9,
            winning_player: winner,
            score: ans_score,
        },
    };
    serde_json::to_string(&result).unwrap()
}
