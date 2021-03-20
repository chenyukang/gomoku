extern crate gomoku;
use glob::glob;
use gomoku::*;
use std::fs;

#[test]
fn run_from_data_dir() {
    for entry in glob("tests/data/**/*.in").expect("expect board input") {
        match entry {
            Ok(path) => {
                let input = String::from(path.to_str().unwrap());
                println!("Running Board Input: {}", input);
                let mut board = board::Board::new(fs::read_to_string(path).unwrap(), 15, 15);
                board.print();
                let player = board.next_player();
                let mut monte = monte::MonteCarlo::new(board.clone(), player, 2000);
                let mv = monte.search_move();
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
                let cmp_content = fs::read_to_string(input.replace(".in", ".cmp"))
                    .unwrap()
                    .trim()
                    .to_string();
                assert_eq!(cmp_content, output);
            }
            Err(e) => println!("error:  {:?}", e),
        }
    }
}
