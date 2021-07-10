build() {
  cargo build --release
}

# build_offline() {
#   cargo build --release --offline
# }

test() {
  cargo test --release --lib
}

clippy() {
  cargo test --release --all-features
  cargo fmt --all -- --check
  cargo clippy  --all-features --all-targets -- --deny warnings
}

test_verbose() {
  RUST_LOG=debug cargo test --release --lib -- --nocapturecargo
}

run() {
  ./target/release/icfp2021 "@"
}

# interact() {
#   RUST_MIN_STACK=200000000 RUST_LOG=info cargo run --release --bin app -- -v interact
# }

# bench() {
#   cargo build --release && RUST_MIN_STACK=200000000 hyperfine './target/release/app bench'
# }
