#!/usr/bin/env bash
# Run integration tests first, then the default non-integration lane.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=dev/common.sh
source "${SCRIPT_DIR}/dev/common.sh"

ROOT_DIR="$(dev_repo_root)"
cd "${ROOT_DIR}"
dev_load_repo_dotenv "${ROOT_DIR}"

echo "=== Integration tests ==="
bash "${ROOT_DIR}/scripts/test-integration.sh"

echo ""
echo "=== Unit and default tests ==="
dev_run_tool pytest --cov-fail-under=90 -m "not integration and not upstream_live and not live" -q
