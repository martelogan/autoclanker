#!/usr/bin/env python3
"""Validate strict-environment manifest parity across dev tooling lanes."""

from __future__ import annotations

import json
import re
import sys
import tomllib

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MISE_TOML = ROOT / "mise.toml"
DEVENV_NIX = ROOT / "devenv.nix"
DEVCONTAINER_JSON = ROOT / ".devcontainer" / "devcontainer.json"
ENVRC_DEVENV = ROOT / "configs" / "strict-env" / "envrc.devenv.example"

REQUIRED_ENV_KEYS = {
    "UV_CACHE_DIR",
    "MISE_DATA_DIR",
    "MISE_CACHE_DIR",
    "MISE_CONFIG_DIR",
    "MISE_STATE_DIR",
}


def _read_mise_tools() -> dict[str, str]:
    data = tomllib.loads(MISE_TOML.read_text(encoding="utf-8"))
    tools = data.get("tools", {})
    if not isinstance(tools, dict):
        msg = "mise.toml [tools] must be a table"
        raise ValueError(msg)
    return {"python": str(tools.get("python", ""))}


def _read_devenv_versions_and_env() -> tuple[dict[str, str], set[str]]:
    text = DEVENV_NIX.read_text(encoding="utf-8")

    versions: dict[str, str] = {}

    py_match = re.search(r"pkgs\.python(\d{2,3})\b", text)
    if py_match:
        digits = py_match.group(1)
        versions["python"] = f"{digits[0]}.{digits[1:]}"

    env_keys = set(re.findall(r"^\s*([A-Z][A-Z0-9_]*)\s*=", text, flags=re.MULTILINE))
    return versions, env_keys


def _read_devcontainer_versions_and_env() -> tuple[dict[str, str], set[str]]:
    data = json.loads(DEVCONTAINER_JSON.read_text(encoding="utf-8"))
    features = data.get("features", {})
    container_env = data.get("containerEnv", {})
    if not isinstance(features, dict):
        msg = "devcontainer features must be an object"
        raise ValueError(msg)
    if not isinstance(container_env, dict):
        msg = "devcontainer containerEnv must be an object"
        raise ValueError(msg)

    versions: dict[str, str] = {}
    for key, value in features.items():
        if not isinstance(value, dict):
            continue
        version = value.get("version")
        if not isinstance(version, str):
            continue
        if "features/python" in key:
            versions["python"] = version
    return versions, set(container_env.keys())


def _check_template(path: Path, required_snippets: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    for snippet in required_snippets:
        if snippet not in text:
            errors.append(f"{path.name} missing snippet: {snippet}")
    return errors


def main() -> int:
    errors: list[str] = []

    for file_path in (MISE_TOML, DEVENV_NIX, DEVCONTAINER_JSON, ENVRC_DEVENV):
        if not file_path.exists():
            errors.append(f"missing required file: {file_path}")

    if errors:
        for message in errors:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1

    mise = _read_mise_tools()
    devenv_versions, devenv_env = _read_devenv_versions_and_env()
    devcontainer_versions, devcontainer_env = _read_devcontainer_versions_and_env()

    expected_python = mise.get("python", "")
    if not expected_python:
        errors.append("mise.toml missing tools.python")
    elif devcontainer_versions.get("python") != expected_python:
        errors.append(
            ".devcontainer python version "
            f"{devcontainer_versions.get('python')!r} does not match "
            f"mise.toml {expected_python!r}"
        )

    if devenv_versions.get("python") != expected_python:
        errors.append(
            f"devenv.nix python version {devenv_versions.get('python')!r} "
            f"does not match mise.toml {expected_python!r}"
        )

    for lane_name, env_keys in (
        ("devenv.nix", devenv_env),
        (".devcontainer/devcontainer.json", devcontainer_env),
    ):
        missing_keys = REQUIRED_ENV_KEYS - env_keys
        if missing_keys:
            missing_display = ", ".join(sorted(missing_keys))
            errors.append(f"{lane_name} missing required env keys: {missing_display}")

    errors.extend(
        _check_template(
            ENVRC_DEVENV,
            (
                "dotenv_if_exists .env",
                "dotenv_if_exists .env.local",
                'eval "$(devenv direnvrc)"',
                "use devenv",
            ),
        )
    )

    if errors:
        for message in errors:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1

    print("OK: strict-environment manifests are coherent across supported lanes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
