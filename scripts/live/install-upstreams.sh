#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REAL_UPSTREAM_ROOT="${AUTOCLANKER_REAL_UPSTREAM_WORK_ROOT:-${ROOT_DIR}/.local/real-upstreams}"

mkdir -p "${REAL_UPSTREAM_ROOT}"

sync_repo() {
    local name="$1"
    local path_var="$2"
    local url_var="$3"
    local branch_var="$4"
    local default_url="$5"
    local default_branch="$6"
    local path_value="${!path_var:-}"
    local url_value="${!url_var:-${default_url}}"
    local branch_value="${!branch_var:-${default_branch}}"
    local checkout_path="${REAL_UPSTREAM_ROOT}/${name}"
    local refresh_value="${AUTOCLANKER_LIVE_REFRESH:-0}"

    if [[ -n "${path_value}" ]]; then
        if [[ ! -d "${path_value}" ]]; then
            echo "Configured ${path_var} does not exist: ${path_value}" >&2
            return 1
        fi
        export "${path_var}=${path_value}"
        echo "ready: ${name} -> ${path_value}"
        return 0
    fi

    if [[ -d "${checkout_path}/.git" && "${refresh_value}" != "1" ]]; then
        export "${path_var}=${checkout_path}"
        echo "ready: ${name} -> ${checkout_path}"
        return 0
    fi

    if [[ -z "${url_value}" ]]; then
        echo "Missing ${path_var} and ${url_var}; provide a real upstream checkout path or git URL for ${name}." >&2
        return 1
    fi

    if [[ -d "${checkout_path}/.git" ]]; then
        git -C "${checkout_path}" fetch --depth 1 origin "${branch_value}"
        git -C "${checkout_path}" checkout "${branch_value}"
        git -C "${checkout_path}" pull --ff-only origin "${branch_value}"
    else
        git clone --depth 1 --branch "${branch_value}" "${url_value}" "${checkout_path}"
    fi

    export "${path_var}=${checkout_path}"
    echo "ready: ${name} -> ${checkout_path}"
}

autoclanker_install_upstreams() {
    local requested=("$@")
    if [[ ${#requested[@]} -eq 0 ]]; then
        requested=("autoresearch" "cevolve")
    fi

    local install_autoresearch=0
    local install_cevolve=0

    for name in "${requested[@]}"; do
        case "${name}" in
            all)
                install_autoresearch=1
                install_cevolve=1
                ;;
            autoresearch)
                install_autoresearch=1
                ;;
            cevolve)
                install_cevolve=1
                ;;
            *)
                echo "Unsupported upstream dependency target: ${name}" >&2
                return 2
                ;;
        esac
    done

    if [[ ${install_autoresearch} -eq 1 ]]; then
        sync_repo "autoresearch" \
            "AUTOCLANKER_LIVE_AUTORESEARCH_PATH" \
            "AUTOCLANKER_LIVE_AUTORESEARCH_GIT_URL" \
            "AUTOCLANKER_LIVE_AUTORESEARCH_GIT_REF" \
            "https://github.com/karpathy/autoresearch.git" \
            "master"
    fi

    if [[ ${install_cevolve} -eq 1 ]]; then
        sync_repo "cevolve" \
            "AUTOCLANKER_LIVE_CEVOLVE_PATH" \
            "AUTOCLANKER_LIVE_CEVOLVE_GIT_URL" \
            "AUTOCLANKER_LIVE_CEVOLVE_GIT_REF" \
            "https://github.com/jnormore/cevolve.git" \
            "main"
    fi

    export AUTOCLANKER_LIVE_AUTORESEARCH_ADAPTER_MODULE="autoclanker.bayes_layer.live_upstreams"
    export AUTOCLANKER_LIVE_CEVOLVE_ADAPTER_MODULE="autoclanker.bayes_layer.live_upstreams"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    autoclanker_install_upstreams "$@"
fi
