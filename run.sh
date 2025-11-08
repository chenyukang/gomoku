#!/bin/bash
set -e
set -u
set -o pipefail

pushd backend
cargo build --lib -p gomoku
rm -rf pkg
wasm-pack build --target web
popd
rm -rf ./client/pkg
cp -rf backend/pkg ./client/
rm -rf ./client/pkg/.gitignore

pushd backend
cargo build --release --features "server"

# Only start server if not in CI environment
if [ -z "${CI:-}" ] && [ -z "${GITHUB_ACTIONS:-}" ]; then
    ./target/release/gomoku -s
else
    echo "Running in CI environment, skipping server start"
fi

