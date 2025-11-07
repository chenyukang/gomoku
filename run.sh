pushd backend
cargo build --lib -p gomoku
rm -rf pkg
wasm-pack build --target web
popd
rm -rf ./client/pkg
cp -rf backend/pkg ./client/
rm -rf ./client/pkg/.gitignore

pushd backend
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="/Users/yukang/.local/share/mise/installs/python/3.13.3/lib/python3.13/site-packages/torch/lib:$DYLD_LIBRARY_PATH"
cargo build --release --features "server,alphazero"
./target/release/gomoku -s

