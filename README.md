# gomoku

[![Build Status](https://github.com/chenyukang/gomoku/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/chenyukang/gomoku/actions/workflows/rust.yml)

A Gomoku Web Applition to explore minimax algorithm with alpha-beta tunning,
Azure Function, Rust and WebAssembly.

# Usage

### Build server backend
```sh
run.sh
```
Open `http://localhost:3000` in browser and have fun.

### Build WASM backend

```sh
cargo install wasm-bindgen-cli wasm-pack
./wasm.sh
```

# [Demo](https://lemon-hill-0c2cac210.azurestaticapps.net/)

![demo](./client/assets/gomoku_demo.png)

### Credits

Thanks Yunzhu.Li for initial version of [Client](https://github.com/yunzhu-li/blupig-gomoku).