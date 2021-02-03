use clap::clap_app;
mod board;
mod server;

fn main() {
    let matches = clap_app!(myapp =>
        (version: "0.1")
        (author: "yukang <moorekang@gmail.com>")
        (about: "Algo backend for Gomoku")
        (@arg input: -i --input +takes_value "Current board of gomoku")
        (@arg verbose: -v --verbose "Print version information verbosely")
        (@arg width: -w --width +takes_value "The board width")      
        (@arg height: -h --height +takes_value "The board height")  
        (@arg depth: -d --depth +takes_value "The search depth for algo")
        (@arg server: -s --server "Run in Server mode")
    ).get_matches();

    let mut search_depth = 14;
    let mut board_width = 4;
    let mut board_height = 4;
    
    if let Some(depth) = matches.value_of("depth") {
        search_depth = depth.parse::<i32>().unwrap();
    }

    if let Some(width) = matches.value_of("width") {
        board_width = width.parse::<usize>().unwrap();
    }

    if let Some(height) = matches.value_of("height") {
        board_height = height.parse::<usize>().unwrap();
    }

    if matches.occurrences_of("server") > 0 {
        let _ = server::run_server();
    } else {
         if let Some(input) = matches.value_of("input") { 
            let board = board::Board::new(input.to_string(), board_width, board_height);
            print!("created board: {:?} with depth: {}", board, search_depth);
         } else { 
          panic!("Input board is required");
         }
    }
}