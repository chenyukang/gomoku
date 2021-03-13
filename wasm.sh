pushd backend
cargo build
rm -rf pkg
wasm-pack build --target web
popd
rm -rf ./wasm-client/pkg
cp -rf backend/pkg ./wasm-client/
