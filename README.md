# gomoku

[![Build Status](https://github.com/chenyukang/gomoku/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/chenyukang/gomoku/actions/workflows/rust.yml)

A Gomoku Web Applition to explore minimax algorithm with alpha-beta tunning,
Azure Function, Rust and WebAssembly.

# Usage

### Build server backend
```sh
cd backend
cargo build --release --features "server"
./target/release/gomoku -s  //start server listen to http://localhost:3000

// open client/index.html in browser and have fun.
```

### Build WASM backend

cargo install wasm-pack
```sh
./wasm.sh
// open wasm-client/index.html in brwoser and have fun
```

# [Demo](https://lemon-hill-0c2cac210.azurestaticapps.net/)

![demo](./client/assets/gomoku_demo.png)