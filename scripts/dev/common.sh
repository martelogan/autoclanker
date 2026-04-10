#!/usr/bin/env bash
# Shared helpers for autoclanker developer tooling scripts.

set -euo pipefail

_dev_script_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd
}

dev_repo_root() {
    if [[ -n "${AUTOCLANKER_DEV_REPO_ROOT:-}" ]]; then
        echo "${AUTOCLANKER_DEV_REPO_ROOT}"
        return 0
    fi
    cd "$(_dev_script_dir)/../.." >/dev/null 2>&1 && pwd
}

dev_install_root() {
    local repo_root
    repo_root="$(dev_repo_root)"
    echo "${AUTOCLANKER_DEV_INSTALL_ROOT:-${repo_root}/.local/dev}"
}

dev_venv_bin_dir() {
    echo "$(dev_repo_root)/.venv/bin"
}

dev_local_bin_dir() {
    echo "$(dev_install_root)/bin"
}

dev_ensure_dirs() {
    mkdir -p "$(dev_install_root)" "$(dev_local_bin_dir)"
}

dev_log() {
    echo "[autoclanker-dev] $*"
}

dev_source_dotenv_file() {
    local file="$1"
    if [[ ! -f "${file}" ]]; then
        return 0
    fi
    # shellcheck disable=SC1090
    set -a
    source "${file}"
    set +a
}

dev_load_repo_dotenv() {
    local repo_root="${1:-}"
    if [[ -z "${repo_root}" ]]; then
        repo_root="$(dev_repo_root)"
    fi
    dev_source_dotenv_file "${repo_root}/.env"
    dev_source_dotenv_file "${repo_root}/.env.local"
}

dev_run_tool() {
    local tool="$1"
    shift

    local tool_path
    tool_path="$(dev_venv_bin_dir)/${tool}"
    if [[ -x "${tool_path}" ]]; then
        "${tool_path}" "$@"
        return 0
    fi

    uv run "${tool}" "$@"
}

dev_run_python() {
    local python_bin
    python_bin="$(dev_venv_bin_dir)/python"
    if [[ -x "${python_bin}" ]]; then
        "${python_bin}" "$@"
        return 0
    fi

    uv run python "$@"
}
