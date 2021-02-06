use super::algo;
use super::board::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Body {
    ai_player: u8,
    cpu_time: String,
    eval_count: u32,
    move_c: usize,
    move_r: usize,
    node_count: u32,
    num_threads: i32,
    pm_count: i32,
    search_depth: i32,
    winning_player: u8,
    score: i32,
}

#[derive(Serialize, Deserialize)]
struct Message {
    message: String,
    result: Body,
}

pub fn solve_it(input: &str, player: u8) -> String {
    let mut board = Board::from(input.to_string());
    let mut winner = 0;
    let mut ans_score = 0;
    let mut ans_col = 0;
    let mut ans_row = 0;
    let mut runner = algo::Runner::new(player, 10);
    if let Some(w) = board.any_winner() {
        winner = w;
    } else {
        let (score, row, col) = runner.run_heuristic(&mut board, player);
        board.place(row, col, player);
        if let Some(w) = board.any_winner() {
            println!("winner: {:?}", winner);
            winner = w;
        }
        ans_score = score;
        ans_row = row;
        ans_col = col;
    }
    let result = Message {
        message: String::from("ok"),
        result: Body {
            ai_player: player,
            cpu_time: String::from("1101"),
            eval_count: runner.eval_node,
            move_c: ans_col,
            move_r: ans_row,
            node_count: runner.eval_node,
            num_threads: 1,
            pm_count: 1,
            search_depth: 9,
            winning_player: winner,
            score: ans_score,
        },
    };
    serde_json::to_string(&result).unwrap()
}
