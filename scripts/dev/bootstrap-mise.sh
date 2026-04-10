#!/usr/bin/env bash
# Bootstrap a project-local mise binary when possible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ROOT="$(dev_repo_root)"
INSTALL_ROOT="$(dev_install_root)"
MISE_HOME="${INSTALL_ROOT}/mise"
MISE_BIN="${MISE_HOME}/bin/mise"

mkdir -p "$(dirname "${MISE_BIN}")"

if [[ -x "${MISE_BIN}" ]]; then
    echo "${MISE_BIN}"
    exit 0
fi

if command -v mise >/dev/null 2>&1; then
    ln -sf "$(command -v mise)" "${MISE_BIN}"
    echo "${MISE_BIN}"
    exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required to bootstrap mise" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
INSTALLER="${TMP_DIR}/mise-install.sh"

curl -fsSL "https://mise.run" -o "${INSTALLER}"

export MISE_INSTALL_PATH="${MISE_BIN}"
export MISE_DATA_DIR="${ROOT}/.mise/data"
export MISE_CACHE_DIR="${ROOT}/.mise/cache"
export MISE_CONFIG_DIR="${ROOT}/.mise/config"
export MISE_STATE_DIR="${ROOT}/.mise/state"

sh "${INSTALLER}"

if [[ ! -x "${MISE_BIN}" ]]; then
    echo "error: mise bootstrap completed but binary not found at ${MISE_BIN}" >&2
    exit 1
fi

echo "${MISE_BIN}"
