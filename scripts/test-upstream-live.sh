#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/dev/common.sh"

ROOT_DIR="$(dev_repo_root)"
cd "${ROOT_DIR}"
dev_load_repo_dotenv "${ROOT_DIR}"

source "${ROOT_DIR}/scripts/live/install-upstreams.sh"
autoclanker_install_upstreams

export AUTOCLANKER_TEST_LANE="upstream_live"
export AUTOCLANKER_TEST_LANE_KIND="real_upstream_contract_smoke"

dev_run_tool pytest -m upstream_live -q
