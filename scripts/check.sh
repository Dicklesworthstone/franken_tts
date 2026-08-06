#!/usr/bin/env bash
# The one command used by CI and local development for the Phase-0 gate.
set -euo pipefail

cargo fmt --check
cargo check --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
ubs --diff
