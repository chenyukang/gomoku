use super::control;
use serde::{Deserialize, Serialize};
use warp::{
    http::{Response, StatusCode},
    Filter,
};

#[derive(Deserialize, Serialize)]
struct ReqObject {
    s: String,
    p: u8,
}

#[tokio::main]
pub async fn run_server() {
    let opt_query = warp::query::<ReqObject>()
        .map(Some)
        .or_else(|_| async { Ok::<(Option<ReqObject>,), std::convert::Infallible>((None,)) });

    let handler = warp::get()
        .and(warp::path("move"))
        .and(opt_query)
        .map(|p: Option<ReqObject>| match p {
            Some(obj) => {
                println!("p: {} s: {}", obj.p, obj.s);
                let result = control::solve_it((obj.s).as_str(), obj.p);
                Response::builder()
                    .header("Access-Control-Allow-Origin", "*")
                    .body(result)
            }
            None => Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(String::from("Failed to decode query param.")),
        });
    warp::serve(handler).run(([127, 0, 0, 1], 8002)).await
}
