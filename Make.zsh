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

# * task
hello() {
  cargo run -- hello
}

download_problem() {
  # [2021-07-10 Sat]
  mkdir -p ./task/problem
  cargo run -- download-problem 1 78
}

post_solution() {
  # [2021-07-10 Sat]
  mkdir -p ./task/solution
  cargo run -- post-solution 1

  # [2021-07-10 Sat]
  # pose_id: {"id":"78841682-a4ac-4b5b-92e1-2d6b37f855eb"}

}

retrive_pose_info() {
  # [2021-07-10 Sat]
  mkdir -p ./task/solution
  cargo run -- retrieve-pose-info 1 78841682-a4ac-4b5b-92e1-2d6b37f855eb
}

visualize_problem() {
  cargo run -- visualize 1
}

visualize_solution() {
  cargo run -- visualize-solution 1
}

solve() {
  RUST_LOG=info cargo run --release -- solve 1
}

# interact() {
#   RUST_MIN_STACK=200000000 RUST_LOG=info cargo run --release --bin app -- -v interact
# }

# bench() {
#   cargo build --release && RUST_MIN_STACK=200000000 hyperfine './target/release/app bench'
# }
