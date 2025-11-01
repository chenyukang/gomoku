pushd backend
cargo build --lib -p gomoku
rm -rf pkg
wasm-pack build --target web
popd
rm -rf ./client/pkg
cp -rf backend/pkg ./client/
rm -rf ./client/pkg/.gitignore

pushd backend
cargo build --release --features "server, alphazero"
./target/release/gomoku -s

