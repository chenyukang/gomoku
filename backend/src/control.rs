#![allow(dead_code)]
use super::algo;
use super::board::*;
use build_timestamp::build_time;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::process::Command;
use std::time::Instant;

build_time!("%A %Y-%m-%d/%H:%M:%S");

#[derive(Serialize, Deserialize)]
struct Body {
    ai_player: u8,
    cpu_time: String,
    eval_count: u32,
    move_c: usize,
    move_r: usize,
    node_count: u32,
    num_threads: i32,
    search_depth: i32,
    winning_player: u8,
    score: i32,
    build: String,
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
    let start = Instant::now();
    let mut runner = algo::Runner::new(player, 4);
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
    let duration = start.elapsed();
    println!("duration: {:?}", duration);
    let result = Message {
        message: String::from("ok"),
        result: Body {
            ai_player: player,
            cpu_time: format!("{:?}", start.elapsed()),
            eval_count: runner.eval_node,
            move_c: ans_col,
            move_r: ans_row,
            node_count: runner.eval_node,
            num_threads: 1,
            search_depth: 9,
            winning_player: winner,
            score: ans_score,
            build: BUILD_TIME.to_string(),
        },
    };
    serde_json::to_string(&result).unwrap()
}

pub fn battle_other_self() {
    let mut board = Board::new_default();
    let opponent = 1;
    let me = 2;
    board.place(7, 7, 1);

    loop {
        let output = Command::new("gomoku")
            .arg("-s")
            .arg(board.to_string())
            .arg("-p")
            .arg(me.to_string())
            .output()
            .expect("failed to execute process");
        let command_res = String::from_utf8(output.stdout).unwrap();
        //println!("output: {:?}", command_res);
        let json: Value = serde_json::from_str(command_res.as_str()).unwrap();
        let row_str: String = Value::to_string(&json["result"]["move_r"])
            .chars()
            .filter(|&x| x != '\"')
            .collect();
        let col_str: String = Value::to_string(&json["result"]["move_c"])
            .chars()
            .filter(|&x| x != '\"')
            .collect();

        let row = row_str.parse::<i32>().unwrap();
        let col = col_str.parse::<i32>().unwrap();
        println!("+ row: {:?} col: {:?}", row, col);
        board.place(row as usize, col as usize, me);
        board.print();
        if let Some(w) = board.any_winner() {
            println!("winner is: {} !!!!", w);
            break;
        }

        //let args = format!("-s {} -p {}", board.to_string(), opponent);
        let output = Command::new("gomoku")
            .arg("-s")
            .arg(board.to_string())
            .arg("-p")
            .arg(opponent.to_string())
            .output()
            .expect("failed to execute process");
        let command_res = String::from_utf8(output.stdout).unwrap();

        let json: Value = serde_json::from_str(command_res.as_str()).unwrap();
        let row_str: String = Value::to_string(&json["result"]["move_r"])
            .chars()
            .filter(|&x| x != '\"')
            .collect();
        let col_str: String = Value::to_string(&json["result"]["move_c"])
            .chars()
            .filter(|&x| x != '\"')
            .collect();

        let row = row_str.parse::<i32>().unwrap();
        let col = col_str.parse::<i32>().unwrap();
        println!("o row: {:?} col: {:?}", row, col);
        board.place(row as usize, col as usize, opponent);
        board.print();
        if let Some(w) = board.any_winner() {
            println!("winner is: {} !!!!", w);
            break;
        }
    }
}

pub fn battle() {
    let mut board = Board::new_default();
    let opponent = 1;
    let me = 2;
    board.place(7, 7, 1);
    let mut runner = algo::Runner::new(2, 4);
    //println!("board: {}", board.to_string());
    loop {
        let (_, row, col) = runner.run_heuristic(&mut board, me);
        println!("+ row: {:?} col: {:?}", row, col);
        board.place(row, col, me);
        board.print();
        if let Some(w) = board.any_winner() {
            println!("winner is: {} !!!!", w);
            break;
        }
        //let args = format!("-s {} -p {}", board.to_string(), opponent);
        let output = Command::new("gomoku")
            .arg("-s")
            .arg(board.to_string())
            .arg(opponent.to_string())
            .output()
            .expect("failed to execute process");
        let command_res = String::from_utf8(output.stdout).unwrap();
        //println!("output: {:?}", command_res);
        let json: Value = serde_json::from_str(command_res.as_str()).unwrap();
        let row_str: String = Value::to_string(&json["result"]["move_r"])
            .chars()
            .filter(|&x| x != '\"')
            .collect();
        let col_str: String = Value::to_string(&json["result"]["move_c"])
            .chars()
            .filter(|&x| x != '\"')
            .collect();

        let row = row_str.parse::<i32>().unwrap();
        let col = col_str.parse::<i32>().unwrap();
        println!("o row: {:?} col: {:?}", row, col);
        board.place(row as usize, col as usize, opponent);
        board.print();
        if let Some(w) = board.any_winner() {
            println!("winner is: {} !!!!", w);
            break;
        }
    }
}

pub fn rev_battle() {
    let mut board = Board::new_default();
    let opponent = 2;
    let me = 1;
    board.place(7, 7, 1);
    let mut runner = algo::Runner::new(2, 4);
    loop {
        //let args = format!("-s {} -p {}", board.to_string(), opponent);
        let output = Command::new("gomoku")
            .arg("-s")
            .arg(board.to_string())
            .arg(opponent.to_string())
            .output()
            .expect("failed to execute process");

        let command_res = String::from_utf8(output.stdout).unwrap();
        //println!("output: {:?}", command_res);
        let json: Value = serde_json::from_str(command_res.as_str()).unwrap();
        let row_str: String = Value::to_string(&json["result"]["move_r"])
            .chars()
            .filter(|&x| x != '\"')
            .collect();
        let col_str: String = Value::to_string(&json["result"]["move_c"])
            .chars()
            .filter(|&x| x != '\"')
            .collect();

        let row = row_str.parse::<i32>().unwrap();
        let col = col_str.parse::<i32>().unwrap();
        println!("+ row: {:?} col: {:?}", row, col);
        board.place(row as usize, col as usize, opponent);
        board.print();
        if let Some(w) = board.any_winner() {
            println!("winner is: {} !!!!", w);
            break;
        }

        let (_, row, col) = runner.run_heuristic(&mut board, me);
        println!("o row: {:?} col: {:?}", row, col);
        board.place(row, col, me);
        board.print();
        if let Some(w) = board.any_winner() {
            println!("winner is: {} !!!!", w);
            break;
        }
    }
}

pub fn battle_self() {
    let mut board = Board::new_default();
    let opponent = 2;
    let me = 1;
    board.place(7, 7, 1);
    let mut runner = algo::Runner::new(2, 4);
    //println!("board: {}", board.to_string());
    loop {
        let (_, row, col) = runner.run_heuristic(&mut board, opponent);
        println!("o row: {:?} col: {:?}", row, col);
        board.place(row, col, opponent);
        board.print();
        if let Some(w) = board.any_winner() {
            println!("winner is: {} !!!!", w);
            break;
        }

        let (_, row, col) = runner.run_heuristic(&mut board, me);
        println!("+ row: {:?} col: {:?}", row, col);
        board.place(row, col, me);
        board.print();
        if let Some(w) = board.any_winner() {
            println!("winner is: {} !!!!", w);
            break;
        }
    }
}
