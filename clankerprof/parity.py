from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any, cast

from clankerprof.cli import main as clankerprof_main
from clankerprof.jsonio import parse_strict_json, parse_strict_yaml

OPT_IN_ENV = "CLANKERPROF_REAL_PROFILE_PARITY"
ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def _pushd(path: Path) -> Generator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _json(path: str | Path) -> dict[str, Any]:
    payload = parse_strict_json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return cast(dict[str, Any], payload)


def _csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _assert_json_equal(actual: dict[str, Any], expected_path: str | Path) -> None:
    expected = _json(expected_path)
    if actual != expected:
        raise ValueError(f"JSON parity mismatch against {expected_path}.")


def _assert_csv_equal(actual_path: str | Path, expected_path: str | Path) -> None:
    actual = _csv_rows(actual_path)
    expected = _csv_rows(expected_path)
    if actual != expected:
        raise ValueError(f"CSV parity mismatch against {expected_path}.")


def _run_to_json(argv: list[str], output_path: Path) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        status = clankerprof_main([*argv, "--output", str(output_path)])
    if status != 0:
        raise ValueError(f"clankerprof command failed with status {status}: {argv}")
    return _json(output_path)


def _run_rust_to_json(argv: list[str], output_path: Path) -> dict[str, Any]:
    if shutil.which("cargo") is None:
        raise ValueError("cargo is required for --check-rust-core.")
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "clankerprof-core",
            "--bin",
            "clankerprof-rs",
            "--",
            *argv,
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            "clankerprof-rs command failed with status "
            f"{completed.returncode}: {argv}\n{completed.stderr}"
        )
    return _json(output_path)


def _assert_payload_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    if actual != expected:
        raise ValueError(f"{label} parity mismatch.")


def _append_runtime_args(args: argparse.Namespace, argv: list[str]) -> None:
    argv.extend(["--runtime", args.runtime])
    if args.runtime_rules:
        argv.extend(["--runtime-rules", args.runtime_rules])
    if args.core_classes:
        argv.extend(["--core-classes", args.core_classes])
    if args.verbose_runtime_internals:
        argv.append("--verbose-runtime-internals")


def _target_base_argv(args: argparse.Namespace) -> list[str]:
    if args.target_config is None:
        raise ValueError("--target-config is required for target parity checks.")
    argv = [
        "targets",
        "--profile",
        args.profile,
        "--config",
        args.target_config,
    ]
    _append_runtime_args(args, argv)
    if args.fold_runtime_internals:
        argv.append("--fold-runtime-internals")
    if args.track_semantic_callers or args.expected_semantic_callers_csv:
        argv.append("--track-semantic-callers")
    if args.no_enhanced:
        argv.append("--no-enhanced")
    if args.attributables:
        argv.extend(["--attributables", args.attributables])
    return argv


def _run_target_json(
    args: argparse.Namespace,
    tmpdir: Path,
) -> dict[str, Any] | None:
    if args.expected_target_json is None:
        return None
    target_json = _run_to_json(
        [*_target_base_argv(args), "--format", "json"],
        tmpdir / "targets.json",
    )
    _assert_json_equal(target_json, args.expected_target_json)
    return target_json


def _run_target_csv(
    args: argparse.Namespace,
    tmpdir: Path,
) -> None:
    if args.expected_target_csv is None and args.expected_verbose_target_csv is None:
        return
    argv = [*_target_base_argv(args), "--format", "csv"]
    semantic_path = tmpdir / "semantic-callers.csv"
    if args.expected_semantic_callers_csv:
        argv.extend(["--semantic-callers-csv", str(semantic_path)])

    compat_target_csv_layout = (
        args.legacy_target_csv_layout or args.target_csv_layout == "compat"
    )
    if compat_target_csv_layout:
        with _pushd(tmpdir), contextlib.redirect_stdout(io.StringIO()):
            status = clankerprof_main(
                [
                    *argv,
                    "--target-csv-layout",
                    "compat",
                    "--output",
                    "targets.csv",
                ]
            )
        if status != 0:
            raise ValueError("clankerprof compatibility target CSV command failed.")
        if args.expected_target_csv:
            _assert_csv_equal(
                tmpdir / "output" / "targets.csv", args.expected_target_csv
            )
        if args.expected_verbose_target_csv:
            _assert_csv_equal(
                tmpdir / "output" / "verbose" / "targets.csv",
                args.expected_verbose_target_csv,
            )
    elif args.expected_verbose_target_csv:
        raise ValueError(
            "--expected-verbose-target-csv requires --target-csv-layout=compat."
        )
    elif args.expected_target_csv:
        csv_path = tmpdir / "targets.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            status = clankerprof_main([*argv, "--output", str(csv_path)])
        if status != 0:
            raise ValueError("clankerprof target CSV command failed.")
        _assert_csv_equal(csv_path, args.expected_target_csv)

    if args.expected_semantic_callers_csv:
        _assert_csv_equal(semantic_path, args.expected_semantic_callers_csv)


def _run_targets(args: argparse.Namespace, tmpdir: Path) -> dict[str, Any] | None:
    if args.target_config is None:
        return None
    target_json = _run_target_json(args, tmpdir)
    _run_target_csv(args, tmpdir)
    return target_json or {"checked": True}


def _run_slices(args: argparse.Namespace, tmpdir: Path) -> dict[str, Any] | None:
    if args.slice_config is None:
        return None
    argv = [
        "slices",
        "--profile",
        args.profile,
        "--config",
        args.slice_config,
    ]
    _append_runtime_args(args, argv)
    slice_json = _run_to_json(argv, tmpdir / "slices.json")
    if args.expected_slice_json:
        _assert_json_equal(slice_json, args.expected_slice_json)
    return slice_json


def _boundary_base_argv(args: argparse.Namespace) -> list[str]:
    if args.boundary_config is None:
        raise ValueError(
            "--scope-config or --boundary-config is required for scope parity checks."
        )
    argv = [
        "boundaries",
        "--profile",
        args.profile,
        "--config",
        args.boundary_config,
    ]
    _append_runtime_args(args, argv)
    if args.fold_runtime_internals:
        argv.append("--fold-runtime-internals")
    if args.no_enhanced:
        argv.append("--no-enhanced")
    if args.boundary_top is not None:
        argv.extend(["--top", str(args.boundary_top)])
    return argv


def _run_boundaries(args: argparse.Namespace, tmpdir: Path) -> dict[str, Any] | None:
    if args.boundary_config is None:
        return None
    boundary_json = _run_to_json(
        _boundary_base_argv(args),
        tmpdir / "boundaries.json",
    )
    if args.expected_boundary_json:
        _assert_json_equal(boundary_json, args.expected_boundary_json)
    return boundary_json


def _run_facts(args: argparse.Namespace, tmpdir: Path) -> dict[str, Any] | None:
    if args.expected_facts_json is None:
        return None
    facts_json = _run_to_json(
        ["facts", "--profile", args.profile],
        tmpdir / "facts.json",
    )
    _assert_json_equal(facts_json, args.expected_facts_json)
    return facts_json


def _run_rust_core(args: argparse.Namespace, tmpdir: Path) -> list[str]:
    if not args.check_rust_core:
        return []
    checked: list[str] = []
    python_facts = _run_to_json(
        ["facts", "--profile", args.profile],
        tmpdir / "python-facts.json",
    )
    rust_facts = _run_rust_to_json(
        ["facts", "--profile", args.profile],
        tmpdir / "rust-facts.json",
    )
    _assert_payload_equal(rust_facts, python_facts, label="Rust facts")
    checked.append("rust_facts")

    if args.target_config is not None:
        python_targets = _run_to_json(
            [*_target_base_argv(args), "--format", "json"],
            tmpdir / "python-targets.json",
        )
        rust_targets = _run_rust_to_json(
            [
                "targets",
                "--profile",
                args.profile,
                "--config",
                args.target_config,
            ],
            tmpdir / "rust-targets.json",
        )
        _assert_payload_equal(rust_targets, python_targets, label="Rust targets")
        checked.append("rust_targets")
    if args.slice_config is not None:
        rust_slice_argv = _rust_slice_argv(args, args.slice_config)
        if rust_slice_argv is not None:
            python_slices = _run_to_json(
                [
                    "slices",
                    "--profile",
                    args.profile,
                    "--config",
                    args.slice_config,
                ],
                tmpdir / "python-slices.json",
            )
            rust_slices = _run_rust_to_json(
                rust_slice_argv,
                tmpdir / "rust-slices.json",
            )
            _assert_payload_equal(rust_slices, python_slices, label="Rust slices")
            checked.append("rust_slices")
    return checked


def _load_config_mapping(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    if config_path.suffix == ".toml":
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    else:
        payload = parse_strict_yaml(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a config object.")
    return cast(dict[str, object], payload)


def _string_values(payload: dict[str, object], *keys: str) -> list[str]:
    result: list[str] = []
    for key in keys:
        value = payload.get(key, [])
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"{key} in slice config must be an array.")
        result.extend(str(item) for item in cast(list[object], value))
    return result


def _rust_slice_argv(
    args: argparse.Namespace,
    config_path: str,
) -> list[str] | None:
    config = _load_config_mapping(config_path)
    if config.get("attribute"):
        return None
    argv = ["slices", "--profile", args.profile]
    raw_slices = config.get("slices")
    if raw_slices is not None:
        argv.extend(["--slices", str(raw_slices)])
    for raw_filter in _string_values(config, "filters", "filter"):
        argv.extend(["--filter", raw_filter])
    for raw_collapse in _string_values(config, "collapse"):
        argv.extend(["--collapse", raw_collapse])
    if config.get("top") is not None:
        argv.extend(["--top", str(config["top"])])
    raw_by_slice = config.get("by_slice")
    if raw_by_slice is not None:
        if isinstance(raw_by_slice, bool):
            if raw_by_slice:
                argv.extend(["--by-slice", "0.1%"])
        else:
            argv.extend(["--by-slice", str(raw_by_slice)])
    if config.get("show_paths"):
        argv.append("--show-paths")
    if config.get("no_collapse_native"):
        argv.append("--no-collapse-native")
    raw_unattributed = config.get("unattributed_libraries")
    if raw_unattributed is None:
        raw_unattributed = config.get("unattributed_gems")
    if raw_unattributed not in (None, False, True):
        return None
    if raw_unattributed is True:
        argv.append("--unattributed-libraries")
    return argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare clankerprof output against local real-profile reference "
            "artifacts. This helper never downloads or commits profile data."
        )
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--runtime", choices=("generic", "ruby"), default="generic")
    parser.add_argument("--runtime-rules")
    parser.add_argument("--core-classes", "--ruby-core-classes", dest="core_classes")
    parser.add_argument("--target-config")
    parser.add_argument("--slice-config")
    parser.add_argument("--scope-config", "--boundary-config", dest="boundary_config")
    parser.add_argument("--attributables", "--cpu-attributables", dest="attributables")
    parser.add_argument("--expected-facts-json")
    parser.add_argument("--expected-target-json")
    parser.add_argument("--expected-target-csv")
    parser.add_argument("--expected-verbose-target-csv")
    parser.add_argument("--expected-semantic-callers-csv")
    parser.add_argument("--expected-slice-json")
    parser.add_argument("--expected-boundary-json")
    parser.add_argument("--boundary-top", type=int)
    parser.add_argument("--fold-runtime-internals", action="store_true")
    parser.add_argument("--track-semantic-callers", action="store_true")
    parser.add_argument("--verbose-runtime-internals", action="store_true")
    parser.add_argument("--no-enhanced", action="store_true")
    parser.add_argument("--target-csv-layout", choices=("standard", "compat"))
    parser.add_argument("--legacy-target-csv-layout", action="store_true")
    parser.add_argument(
        "--check-rust-core",
        action="store_true",
        help=(
            "Also compare clankerprof-rs facts and generic target JSON against "
            "Python output for this local profile. This reads only caller-provided "
            "local inputs and writes temporary outputs."
        ),
    )
    parser.add_argument(
        "--allow-local-inputs",
        action="store_true",
        help=f"Bypass the {OPT_IN_ENV}=1 safety gate for local-only runs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.legacy_target_csv_layout and args.target_csv_layout == "standard":
        parser.error(
            "--legacy-target-csv-layout conflicts with --target-csv-layout=standard."
        )
    if os.environ.get(OPT_IN_ENV) != "1" and not args.allow_local_inputs:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"Set {OPT_IN_ENV}=1 or pass --allow-local-inputs to "
                        "confirm local profile artifacts should be read."
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="clankerprof-parity-") as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            outputs = {
                "facts": _run_facts(args, tmpdir),
                "targets": _run_targets(args, tmpdir),
                "slices": _run_slices(args, tmpdir),
                "boundaries": _run_boundaries(args, tmpdir),
            }
            rust_outputs = _run_rust_core(args, tmpdir)
        print(
            json.dumps(
                {
                    "ok": True,
                    "checked": sorted(
                        [
                            *(
                                key
                                for key, value in outputs.items()
                                if value is not None
                            ),
                            *rust_outputs,
                        ]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
