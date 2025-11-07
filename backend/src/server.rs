use super::control;
use crate::utils::{BOARD_HEIGHT, BOARD_WIDTH};
use std::net::Ipv4Addr;

use serde::{Deserialize, Serialize};
use warp::{
    fs::dir,
    http::{Response, StatusCode},
    Filter,
};

#[derive(Deserialize, Serialize)]
struct ReqObject {
    state: String,
    algo_type: String,
    width: Option<usize>,
    height: Option<usize>,
}

#[tokio::main]
pub async fn run_server(port: u16) {
    let opt_query = warp::query::<ReqObject>()
        .map(Some)
        .or_else(|_| async { Ok::<(Option<ReqObject>,), std::convert::Infallible>((None,)) });

    let api_move = warp::get()
        .and(warp::path("api"))
        .and(warp::path("move"))
        .and(opt_query)
        .map(|p: Option<ReqObject>| match p {
            Some(obj) => {
                let width = obj.width.unwrap_or(BOARD_WIDTH);
                let height = obj.height.unwrap_or(BOARD_HEIGHT);
                for i in 0..height {
                    for j in 0..width {
                        let c = (i * width + j) as usize;
                        print!("{}", obj.state.chars().nth(c).unwrap());
                    }
                    println!();
                }
                let result = control::solve_it(&obj.state, &obj.algo_type, width, height);
                Response::builder()
                    .header("Access-Control-Allow-Origin", "*")
                    .body(result)
            }
            None => Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(String::from("Failed to decode query param.")),
        });

    // Serve static files from the client/ directory at the root path
    let static_files = warp::get().and(dir("../client"));

    // Combine API route and static files. API takes precedence.
    let handler = api_move.or(static_files);

    println!("listen to : {} ...", port);
    warp::serve(handler)
        .run((Ipv4Addr::UNSPECIFIED, port))
        .await
}
