from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import tempfile

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any, cast

from clankerprof.cli import main as clankerprof_main

OPT_IN_ENV = "CLANKERPROF_REAL_PROFILE_PARITY"


@contextlib.contextmanager
def _pushd(path: Path) -> Generator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
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


def _run_facts(args: argparse.Namespace, tmpdir: Path) -> dict[str, Any] | None:
    if args.expected_facts_json is None:
        return None
    facts_json = _run_to_json(
        ["facts", "--profile", args.profile],
        tmpdir / "facts.json",
    )
    _assert_json_equal(facts_json, args.expected_facts_json)
    return facts_json


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
    parser.add_argument("--attributables", "--cpu-attributables", dest="attributables")
    parser.add_argument("--expected-facts-json")
    parser.add_argument("--expected-target-json")
    parser.add_argument("--expected-target-csv")
    parser.add_argument("--expected-verbose-target-csv")
    parser.add_argument("--expected-semantic-callers-csv")
    parser.add_argument("--expected-slice-json")
    parser.add_argument("--fold-runtime-internals", action="store_true")
    parser.add_argument("--track-semantic-callers", action="store_true")
    parser.add_argument("--verbose-runtime-internals", action="store_true")
    parser.add_argument("--no-enhanced", action="store_true")
    parser.add_argument("--target-csv-layout", choices=("standard", "compat"))
    parser.add_argument("--legacy-target-csv-layout", action="store_true")
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
            }
        print(
            json.dumps(
                {
                    "ok": True,
                    "checked": sorted(
                        key for key, value in outputs.items() if value is not None
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
