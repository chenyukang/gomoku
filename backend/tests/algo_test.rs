extern crate gomoku;
use crate::utils::{BOARD_HEIGHT, BOARD_WIDTH};
use glob::glob;
use gomoku::*;
use std::env;
use std::fs;

fn run_from_data_dir(data_dir: &str, algo_type: &str) {
    let specify_test = env::var("ALGO_TEST").unwrap_or_default();
    for entry in glob(data_dir).expect("expect board input") {
        match entry {
            Ok(path) => {
                let input = String::from(path.to_str().unwrap());
                if specify_test.len() > 0 && input != specify_test {
                    continue;
                }
                println!("Running Board Input: {}", input);
                let content = fs::read_to_string(path).unwrap();
                let mut board = board::Board::new(content.clone(), BOARD_WIDTH, BOARD_HEIGHT);
                board.print();
                let player = board.next_player();
                let mv = algo::gomoku_solve(content.as_str(), algo_type, BOARD_WIDTH, BOARD_HEIGHT);
                println!("move: {:?}", mv);
                let row = mv.x;
                let col = mv.y;
                board.place(row, col, player);
                board.print();
                println!("{:?}", mv);
                let output = format!("row: {} col: {}", row, col);
                println!("output: {}", output);
                fs::write(input.replace(".in", ".out").as_str(), output.clone())
                    .expect("write error");

                // Read expected output(s) - can have multiple valid answers (one per line)
                let cmp_content = fs::read_to_string(input.replace(".in", ".cmp")).unwrap();
                let expected_outputs: Vec<&str> = cmp_content
                    .lines()
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();

                // Check if output matches any of the expected answers
                if expected_outputs.contains(&output.as_str()) {
                    println!("✓ Output matches expected result");
                } else {
                    panic!(
                        "Output '{}' does not match any expected results: {:?}",
                        output, expected_outputs
                    );
                }
            }
            Err(e) => println!("error:  {:?}", e),
        }
    }
}

#[test]
fn run_monte() {
    run_from_data_dir("tests/data/**/*.in", "monte_carlo");
}

#[test]
fn run_minimax() {
    run_from_data_dir("tests/minimax_data/**/*.in", "minimax");
}
