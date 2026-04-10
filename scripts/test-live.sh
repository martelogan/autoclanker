#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/dev/common.sh"

ROOT_DIR="$(dev_repo_root)"
cd "${ROOT_DIR}"
dev_load_repo_dotenv "${ROOT_DIR}"

if [[ -n "${AUTOCLANKER_LLM_ENV_FILE:-}" ]]; then
    dev_source_dotenv_file "${AUTOCLANKER_LLM_ENV_FILE}"
fi

: "${AUTOCLANKER_ENABLE_LLM_LIVE:=0}"
if [[ "${AUTOCLANKER_ENABLE_LLM_LIVE}" != "1" ]]; then
    echo "error: billed LLM live tests are disabled. Set AUTOCLANKER_ENABLE_LLM_LIVE=1 to run them." >&2
    exit 2
fi

if [[ -z "${ANTHROPIC_API_KEY:-${AUTOCLANKER_ANTHROPIC_API_KEY:-}}" ]]; then
    echo "error: set ANTHROPIC_API_KEY or AUTOCLANKER_ANTHROPIC_API_KEY before running ./bin/dev test-live." >&2
    exit 2
fi

export AUTOCLANKER_TEST_LANE="live"
export AUTOCLANKER_TEST_LANE_KIND="billed_model_provider"
export AUTOCLANKER_TEST_LANE_PROVIDER="anthropic"

dev_run_tool pytest -m live -q
