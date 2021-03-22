pushd backend
cargo build
rm -rf pkg
wasm-pack build --target web
popd
rm -rf ./client/pkg
cp -rf backend/pkg ./client/
rm -rf ./client/pkg/.gitignore
