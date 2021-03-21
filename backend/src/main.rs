use clap::clap_app;
use std::env;
mod minimax;
mod board;
mod control;
mod monte;
#[cfg(feature = "server")]
mod server;
mod utils;
mod algo;

fn main() {
    let matches = clap_app!(myapp =>
        (version: "0.1")
        (author: "yukang <moorekang@gmail.com>")
        (about: "Algo backend for Gomoku")
        (@arg input: -i --input +takes_value "Current board of gomoku")
        (@arg verbose: -v --verbose "Print version information verbosely")
        (@arg battle: -b --battle "Run in battle mode")
        (@arg rev_battle: -r --rev_battle "Run in rev battle mode")
        (@arg self_battle: -k --self_battle "Run in battle self mode")
        (@arg monte_battle: -m --monte_battle "Run in battle monte self mode")
        (@arg other_self_battle: -o --other_self_battle "Run in other battle self mode")
        (@arg width: -w --width +takes_value "The board width")
        (@arg height: -h --height +takes_value "The board height")
        (@arg depth: -d --depth +takes_value "The search depth for algo")
        (@arg server: -s --server "Run in Server mode")
    )
    .get_matches();

    let mut search_depth = 14;
    let mut board_width = 4;
    let mut board_height = 4;

    if matches.occurrences_of("battle") > 0 {
        control::battle();
    } else if matches.occurrences_of("monte_battle") > 0 {
        control::battle_monte();
    } else if matches.occurrences_of("rev_battle") > 0 {
        control::rev_battle();
    } else if matches.occurrences_of("self_battle") > 0 {
        control::battle_self();
    } else if matches.occurrences_of("other_self_battle") > 0 {
        control::battle_other_self();
    } else if matches.occurrences_of("server") > 0 {
        let port_key = "FUNCTIONS_CUSTOMHANDLER_PORT";
        let _port: u16 = match env::var(port_key) {
            Ok(val) => val.parse().expect("Custom Handler port is not a number!"),
            Err(_) => 3000,
        };
        cfg_if::cfg_if! {
            if #[cfg(feature = "server")] {
              server::run_server(_port);
            } else {
                panic!("no build with feature : server")
            }
        };
    } else {
        if let Some(depth) = matches.value_of("depth") {
            search_depth = depth.parse::<i32>().unwrap();
        }
        if let Some(width) = matches.value_of("width") {
            board_width = width.parse::<usize>().unwrap();
        }
        if let Some(height) = matches.value_of("height") {
            board_height = height.parse::<usize>().unwrap();
        }
        if let Some(input) = matches.value_of("input") {
            let board = board::Board::new(input.to_string(), board_width, board_height);
            print!("created board: {:?} with depth: {}", board, search_depth);
        } else {
            panic!("Input board is required");
        }
    }
}
