#!/usr/bin/env bash
# Run the Rust clankerprof-core lane: format check, build, and native tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=dev/common.sh
source "${SCRIPT_DIR}/dev/common.sh"

ROOT_DIR="$(dev_repo_root)"
cd "${ROOT_DIR}"

if ! command -v cargo >/dev/null 2>&1; then
    echo "SKIP: cargo is not installed; Rust lane not run."
    exit 0
fi

cargo fmt --check
cargo build --workspace
cargo test --workspace
