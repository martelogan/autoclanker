#!/usr/bin/env bash
# Configure repo-local .envrc from strict-mode config templates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ROOT="$(dev_repo_root)"
MODE="${1:-devenv}"

if [[ "${MODE}" != "devenv" ]]; then
    echo "error: unsupported strict env mode '${MODE}'" >&2
    echo "supported: devenv" >&2
    exit 2
fi

TEMPLATE="${ROOT}/configs/strict-env/envrc.${MODE}.example"
TARGET="${ROOT}/.envrc"

if [[ ! -f "${TEMPLATE}" ]]; then
    echo "error: missing template ${TEMPLATE}" >&2
    exit 1
fi

if [[ -f "${TARGET}" ]]; then
    BACKUP="${TARGET}.bak.$(date +%Y%m%d%H%M%S)"
    cp "${TARGET}" "${BACKUP}"
    dev_log "existing .envrc backed up to ${BACKUP}"
fi

cp "${TEMPLATE}" "${TARGET}"
dev_log "wrote ${TARGET} from ${TEMPLATE}"
dev_log "next: run 'direnv allow' in ${ROOT}"
