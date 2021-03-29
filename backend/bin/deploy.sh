rustup target add x86_64-unknown-linux-musl
cargo build --release --features "server" --target=x86_64-unknown-linux-musl
cp target/x86_64-unknown-linux-musl/release/gomoku function/
