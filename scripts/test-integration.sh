#!/usr/bin/env bash
# Run integration tests when present.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=dev/common.sh
source "${SCRIPT_DIR}/dev/common.sh"

ROOT_DIR="$(dev_repo_root)"
cd "${ROOT_DIR}"
dev_load_repo_dotenv "${ROOT_DIR}"

set +e
dev_run_tool pytest -m integration -q
pytest_rc=$?
set -e

if [[ ${pytest_rc} -eq 5 ]]; then
    echo "No integration tests collected"
    exit 0
fi

exit "${pytest_rc}"
