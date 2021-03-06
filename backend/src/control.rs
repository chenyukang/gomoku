#![allow(dead_code)]
#![warn(unused_variables)]
use super::algo;
use super::board::*;
use super::minimax;
use super::monte;
use build_timestamp::build_time;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::process::Command;
#[cfg(feature = "server")]
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

pub fn solve_it(input: &str, algo_type: &str) -> String {
    let mut board = Board::from(input.to_string());
    let player = board.next_player();
    let mut winner = 0;
    cfg_if::cfg_if! {
        if #[cfg(feature = "server")] {
          let start = Instant::now();
        }
    };
    let mut score = 0;
    let mut row = 0;
    let mut col = 0;
    if let Some(w) = board.any_winner() {
        winner = w;
    } else {
        let mv = algo::gomoku_solve(input, algo_type);
        row = mv.x;
        col = mv.y;
        score = mv.score;
        board.place(row, col, player);
        if let Some(w) = board.any_winner() {
            println!("winner: {:?}", winner);
            winner = w;
        }
    }
    cfg_if::cfg_if! {
        if #[cfg(feature = "server")] {
          let duration = start.elapsed();
        } else {
          let duration = 0;
        }
    };
    println!("duration: {:?}", duration);
    let result = Message {
        message: String::from("ok"),
        result: Body {
            ai_player: player,
            cpu_time: format!("{:?}", duration),
            eval_count: 0,
            move_c: col,
            move_r: row,
            node_count: 0,
            num_threads: 1,
            search_depth: 9,
            winning_player: winner,
            score: score,
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
    let mut runner = minimax::MiniMax::new(2, 4);
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
    let mut runner = minimax::MiniMax::new(2, 4);
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
    let mut runner = minimax::MiniMax::new(2, 4);
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

pub fn battle_monte() {
    let mut board = Board::new_default();
    let opponent = 2;
    let me = 1;
    board.place(7, 7, 1);
    //println!("board: {}", board.to_string());
    loop {
        //let (_, row, col) = runner.run_heuristic(&mut board, opponent);
        //println!("o row: {:?} col: {:?}", row, col);
        let mut monte = monte::MonteCarlo::new(board.clone(), opponent, 4000);
        let mv = monte.search_move();
        let row = mv.x;
        let col = mv.y;
        board.place(row, col, opponent);
        board.print();
        if let Some(w) = board.any_winner() {
            println!("winner is: {} !!!!", w);
            break;
        }

        let mut monte = monte::MonteCarlo::new(board.clone(), me, 4000);
        let mv = monte.search_move();
        let row = mv.x;
        let col = mv.y;
        //let (_, row, col) = runner.run_heuristic(&mut board, me);
        println!("+ row: {:?} col: {:?}", row, col);
        board.place(row, col, me);
        board.print();
        if let Some(w) = board.any_winner() {
            println!("winner is: {} !!!!", w);
            break;
        }
    }
}
