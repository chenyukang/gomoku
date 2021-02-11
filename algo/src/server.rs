#![allow(unused_must_use)]
#![allow(unused_imports)]
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};

use super::board::*;
use super::control::*;
use std::collections::HashMap;
use url::*;

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
            //println!("got request: {} => {}", input, player);
            let result = solve_it(input, player);
            //println!("result:\n {}", result);
            let response = Response::builder()
                .status(200)
                .header("Access-Control-Allow-Origin", "*")
                .body(result.into())
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

    let addr = ([127, 0, 0, 1], 8002).into();

    let server = Server::bind(&addr).serve(make_service_fn(|_| async {
        Ok::<_, hyper::Error>(service_fn(process_handler))
    }));

    println!("Listening on http://{}", addr);

    server.await?;

    Ok(())
}
