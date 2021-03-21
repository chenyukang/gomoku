use super::control;
use std::net::Ipv4Addr;

use serde::{Deserialize, Serialize};
use warp::{
    http::{Response, StatusCode},
    Filter,
};

#[derive(Deserialize, Serialize)]
struct ReqObject {
    state: String,
    algo_type: String,
}

#[tokio::main]
pub async fn run_server(port: u16) {
    let opt_query = warp::query::<ReqObject>()
        .map(Some)
        .or_else(|_| async { Ok::<(Option<ReqObject>,), std::convert::Infallible>((None,)) });

    let handler = warp::get()
        .and(warp::path("api"))
        .and(warp::path("move"))
        .and(opt_query)
        .map(|p: Option<ReqObject>| match p {
            Some(obj) => {
                for i in 0..15 {
                    for j in 0..15 {
                        let c = (i * 15 + j) as usize;
                        print!("{}", obj.state.chars().nth(c).unwrap());
                    }
                    println!();
                }
                let result = control::solve_it(&obj.state, &obj.algo_type);
                Response::builder()
                    .header("Access-Control-Allow-Origin", "*")
                    .body(result)
            }
            None => Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(String::from("Failed to decode query param.")),
        });

    println!("listen to : {} ...", port);
    warp::serve(handler)
        .run((Ipv4Addr::UNSPECIFIED, port))
        .await
}
