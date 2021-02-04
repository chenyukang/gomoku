
use clap::clap_app;

fn main() {
    let matches = clap_app!(myapp =>
        (version: "0.1")
        (author: "yukang <moorekang@gmail.com>")
        (about: "Algo backend for Gomoku")
        (@arg INPUT: +required "Current board of gomoku")
        (@arg verbose: -v --verbose "Print version information verbosely")
        (@arg depth: -d --depth +takes_value "The search depth for algo")        
    ).get_matches();

    let mut search_depth = 14;

    if let Some(depth) = matches.value_of("depth") {
        println!("Value for config depth: {}", depth);
        search_depth = depth.parse::<i32>().unwrap();
        println!("set depth: {}", search_depth);
    }

    let board = matches.value_of("INPUT").unwrap();
    println!("board: {}", board);

    println!("search depth: {}", search_depth);
}