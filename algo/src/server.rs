#![allow(unused_must_use)]
#![allow(unused_imports)]
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use serde::{Deserialize, Serialize};

use super::board;
use std::collections::HashMap;
use url::*;

#[derive(Serialize, Deserialize)]
struct BodyType {
    ai_player: u8,
    build: String,
    cc_0: String,
    cc_1: String,
    cpu_time: String,
    eval_count: u32,
    move_c: usize,
    move_r: usize,
    node_count: i32,
    num_threads: i32,
    pm_count: i32,
    search_depth: i32,
    winning_player: i32,
    score: u32,
}

#[derive(Serialize, Deserialize)]
struct ResponseType {
    message: String,
    result: BodyType,
}

async fn process_handler(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/move") => {
            let params: HashMap<String, String> = req
                .uri()
                .query()
                .map(|v| {
                    url::form_urlencoded::parse(v.as_bytes())
                        .into_owned()
                        .collect()
                })
                .unwrap_or_else(HashMap::new);
            let input = params.get("s").unwrap();
            let player: u8 = params.get("p").unwrap().parse().unwrap();
            println!("got request: {} => {}", input, player);
            let mut board = board::Board::from(input.to_string());
            let (score, row, col) = board.gen_move(player, 3);
            let result = ResponseType {
                message: String::from("ok"),
                result: BodyType {
                    ai_player: 2,
                    build: String::from("Feb  4 2021 06:45:24"),
                    cc_0: String::from("0"),
                    cc_1: String::from("0"),
                    cpu_time: String::from("1101"),
                    eval_count: 1000,
                    move_c: col,
                    move_r: row,
                    node_count: 100,
                    num_threads: 1,
                    pm_count: 1,
                    search_depth: 9,
                    winning_player: 0,
                    score: score,
                },
            };
            let result_str = serde_json::to_string(&result).unwrap();
            println!("result:\n {}", result_str);
            let response = Response::builder()
                .status(200)
                .header("Access-Control-Allow-Origin", "*")
                .body(result_str.into())
                .unwrap();
            Ok(response)
        }
        (&Method::GET, "/") | (&Method::GET, "/post") => Ok(Response::new("status".into())),
        _ => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::empty())
            .unwrap()),
    }
}

#[tokio::main]
pub async fn run_server() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    pretty_env_logger::init();

    let addr = ([127, 0, 0, 1], 8001).into();

    let server = Server::bind(&addr).serve(make_service_fn(|_| async {
        Ok::<_, hyper::Error>(service_fn(process_handler))
    }));

    println!("Listening on http://{}", addr);

    server.await?;

    Ok(())
}
