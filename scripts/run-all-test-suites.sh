#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

bash "${ROOT_DIR}/scripts/run-full-test-suite.sh"
echo ""
echo "=== Upstream-live smoke tests ==="
bash "${ROOT_DIR}/scripts/test-upstream-live.sh"
