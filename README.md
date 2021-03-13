# gomoku

[![Build Status](https://travis-ci.com/chenyukang/gomoku.svg?branch=main)](https://travis-ci.com/chenyukang/gomoku)

A Gomoku backend to explore minimax algorithm with alpha-beta tunning.

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