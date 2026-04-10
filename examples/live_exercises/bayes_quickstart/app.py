"""Small single-file log parser app for the Bayes quickstart exercise.

This file is intentionally simple to read on its own:

- it parses a few representative application log lines,
- it exposes a small set of app knobs that `autoclanker` can reason about,
- and it shows the exact option strings a user would reference in beginner Bayes
  ideas such as `parser.matcher=matcher_compiled`.

The real optimization run still happens through `autoclanker` session commands.
Running this file directly explains the app and previews why one candidate might
be better than another.
"""

from __future__ import annotations

import argparse
import json
import re

from dataclasses import dataclass
from typing import Any

LOG_LINES = (
    "ts=2026-04-09T09:00:01Z level=INFO route=/import trace=ab1 msg=batch_start",
    "ts=2026-04-09T09:00:02Z level=ERROR route=/import trace=ab1 code=E_PARSE msg=unterminated_quote",
    "ts=2026-04-09T09:00:03Z level=INFO route=/search trace=cd4 msg=query_received",
    "ts=2026-04-09T09:00:04Z level=WARN route=/search trace=cd4 code=W_RETRY msg=retrying_backend",
    "ts=2026-04-09T09:00:05Z level=INFO route=/billing trace=ef7 msg=invoice_loaded",
    "ts=2026-04-09T09:00:06Z level=ERROR route=/billing trace=ef7 code=E_TIMEOUT msg=ledger_timeout",
)

CANDIDATE_PRESETS: dict[str, dict[str, str]] = {
    "cand_a_default": {
        "parser.matcher": "matcher_basic",
        "parser.plan": "plan_default",
        "capture.window": "window_default",
        "io.chunk": "chunk_default",
        "emit.summary": "summary_default",
    },
    "cand_b_compiled_matcher": {
        "parser.matcher": "matcher_compiled",
        "parser.plan": "plan_default",
        "capture.window": "window_default",
        "io.chunk": "chunk_default",
        "emit.summary": "summary_default",
    },
    "cand_c_compiled_context_pair": {
        "parser.matcher": "matcher_compiled",
        "parser.plan": "plan_context_pair",
        "capture.window": "window_default",
        "io.chunk": "chunk_default",
        "emit.summary": "summary_default",
    },
    "cand_d_wide_capture_window": {
        "parser.matcher": "matcher_basic",
        "parser.plan": "plan_default",
        "capture.window": "window_wide",
        "io.chunk": "chunk_default",
        "emit.summary": "summary_default",
    },
    "cand_e_wide_window_large_chunk": {
        "parser.matcher": "matcher_basic",
        "parser.plan": "plan_default",
        "capture.window": "window_wide",
        "io.chunk": "chunk_large",
        "emit.summary": "summary_default",
    },
}

COMPILED_PATTERN = re.compile(
    r"level=(?P<level>[A-Z]+)\s+route=(?P<route>\S+)\s+trace=(?P<trace>\S+)"
    r"(?:\s+code=(?P<code>\S+))?\s+msg=(?P<msg>\S+)"
)


@dataclass(frozen=True, slots=True)
class AppConfig:
    matcher_mode: str
    plan_mode: str
    window_mode: str
    chunk_mode: str
    summary_mode: str


def _config_from_candidate(candidate_id: str) -> AppConfig:
    try:
        raw = CANDIDATE_PRESETS[candidate_id]
    except KeyError as exc:
        known = ", ".join(sorted(CANDIDATE_PRESETS))
        raise SystemExit(
            f"Unknown candidate id {candidate_id!r}. Known: {known}."
        ) from exc
    return AppConfig(
        matcher_mode=raw["parser.matcher"],
        plan_mode=raw["parser.plan"],
        window_mode=raw["capture.window"],
        chunk_mode=raw["io.chunk"],
        summary_mode=raw["emit.summary"],
    )


def _chunk_size(config: AppConfig) -> int:
    return 4 if config.chunk_mode == "chunk_large" else 2


def _capture_radius(config: AppConfig) -> int:
    return 2 if config.window_mode == "window_wide" else 1


def _parse_line(line: str, config: AppConfig) -> dict[str, str]:
    if config.matcher_mode == "matcher_basic":
        tokens = dict(token.split("=", 1) for token in line.split() if "=" in token)
        return {
            "level": tokens.get("level", "INFO"),
            "route": tokens.get("route", "/unknown"),
            "trace": tokens.get("trace", "trace-missing"),
            "code": tokens.get("code", ""),
            "msg": tokens.get("msg", "missing_message"),
        }
    match = COMPILED_PATTERN.search(line)
    if match is None:
        raise ValueError(f"Could not parse log line: {line!r}")
    return {
        "level": match.group("level"),
        "route": match.group("route"),
        "trace": match.group("trace"),
        "code": match.group("code") or "",
        "msg": match.group("msg"),
    }


def summarize_logs(config: AppConfig) -> dict[str, Any]:
    parsed = [_parse_line(line, config) for line in LOG_LINES]
    chunk_size = _chunk_size(config)
    capture_radius = _capture_radius(config)

    chunk_routes: list[list[str]] = []
    for start in range(0, len(parsed), chunk_size):
        chunk_routes.append(
            [row["route"] for row in parsed[start : start + chunk_size]]
        )

    incident_groups: list[dict[str, object]] = []
    for index, row in enumerate(parsed):
        if row["level"] not in {"WARN", "ERROR"}:
            continue
        context_routes = [row["route"]]
        if config.plan_mode == "plan_context_pair" and index > 0:
            context_routes.append(parsed[index - 1]["route"])
        elif config.plan_mode == "plan_full_scan":
            start = max(index - capture_radius, 0)
            stop = min(index + capture_radius + 1, len(parsed))
            context_routes.extend(parsed[item]["route"] for item in range(start, stop))
        incident_groups.append(
            {
                "route": row["route"],
                "level": row["level"],
                "code": row["code"] or "no_code",
                "context_routes": sorted(set(context_routes)),
            }
        )

    summary_lines = [
        f"{item['level']} {item['route']} {item['code']}" for item in incident_groups
    ]
    if config.summary_mode == "summary_streaming":
        summary_output = " | ".join(summary_lines)
    else:
        summary_output = "\n".join(summary_lines)

    return {
        "parsed_rows": parsed[:3],
        "incident_groups": incident_groups,
        "chunk_routes": chunk_routes,
        "summary_output": summary_output,
        "capture_radius_lines": capture_radius,
    }


def benchmark_preview(config: AppConfig) -> dict[str, Any]:
    usefulness = 0.55
    estimated_memory_mb = 180.0
    warning: str | None = None

    contribution_table = {
        "parser.matcher:matcher_compiled": 0.35,
        "parser.matcher:matcher_jit": 0.18,
        "parser.plan:plan_context_pair": 0.18,
        "parser.plan:plan_full_scan": -0.08,
        "capture.window:window_wide": 0.12,
        "io.chunk:chunk_large": 0.10,
        "emit.summary:summary_streaming": 0.08,
    }
    memory_table = {
        "capture.window:window_wide": 70.0,
        "io.chunk:chunk_large": 60.0,
        "parser.plan:plan_full_scan": 18.0,
    }
    active_keys = (
        f"parser.matcher:{config.matcher_mode}",
        f"parser.plan:{config.plan_mode}",
        f"capture.window:{config.window_mode}",
        f"io.chunk:{config.chunk_mode}",
        f"emit.summary:{config.summary_mode}",
    )
    for key in active_keys:
        usefulness += contribution_table.get(key, 0.0)
        estimated_memory_mb += memory_table.get(key, 0.0)

    if {
        "parser.matcher:matcher_compiled",
        "parser.plan:plan_context_pair",
    }.issubset(active_keys):
        usefulness += 0.24
    if {"capture.window:window_wide", "io.chunk:chunk_large"}.issubset(active_keys):
        usefulness -= 0.40
        estimated_memory_mb += 55.0
        warning = "Likely to run out of memory on long traces."
    elif {"parser.matcher:matcher_jit", "parser.plan:plan_full_scan"}.issubset(
        active_keys
    ):
        usefulness -= 0.18
        estimated_memory_mb += 12.0
        warning = "Aggressive parsing plan is likely to fail at runtime."

    return {
        "metric_name": "usefulness_score",
        "optimize_direction": "maximize",
        "estimated_usefulness_score": round(usefulness, 3),
        "estimated_memory_mb": round(estimated_memory_mb, 1),
        "warning": warning,
    }


def render_candidate(candidate_id: str) -> dict[str, Any]:
    config = _config_from_candidate(candidate_id)
    log_summary = summarize_logs(config)
    return {
        "app_kind": "single_file_log_parser",
        "candidate_id": candidate_id,
        "what_the_app_does": (
            "Parse application log lines, group related warnings/errors, and emit "
            "a short incident summary for an operator."
        ),
        "optimization_goal": (
            "Improve parser usefulness on repeated log formats while avoiding "
            "high-memory capture strategies."
        ),
        "what_an_option_string_means": (
            "Each beginner Bayes option is one app knob in the form "
            "`gene_id=state_id`, for example "
            "`parser.matcher=matcher_compiled`."
        ),
        "gene_guide": {
            "parser.matcher": "How the parser matches tokens inside each log line.",
            "parser.plan": "How much cross-line context the parser reconstructs.",
            "capture.window": "How many neighboring lines the parser keeps in memory.",
            "io.chunk": "How many log lines the parser reads per batch.",
            "emit.summary": "How the incident summary is emitted.",
        },
        "allowed_states": {
            "parser.matcher": {
                "matcher_basic": "Simple token splitting.",
                "matcher_compiled": "Compiled regex matching for repeated formats.",
                "matcher_jit": "Aggressive regex plan with more capture overhead.",
            },
            "parser.plan": {
                "plan_default": "Summarize each error line on its own.",
                "plan_context_pair": "Pair each error with its closest context line.",
                "plan_full_scan": "Scan a wider context neighborhood for every error.",
            },
            "capture.window": {
                "window_default": "Keep a small amount of context in memory.",
                "window_wide": "Keep a wider context window in memory.",
            },
            "io.chunk": {
                "chunk_default": "Process a small chunk of lines at a time.",
                "chunk_large": "Process a larger chunk of lines at a time.",
            },
            "emit.summary": {
                "summary_default": "Emit a normal end-of-run summary.",
                "summary_streaming": "Stream summary lines as incidents are found.",
            },
        },
        "active_config": {
            "parser.matcher": config.matcher_mode,
            "parser.plan": config.plan_mode,
            "capture.window": config.window_mode,
            "io.chunk": config.chunk_mode,
            "emit.summary": config.summary_mode,
        },
        "minimum_required_files": {
            "belief_preview": [
                "examples/live_exercises/bayes_quickstart/beliefs.yaml",
            ],
            "exact_documented_ranking": [
                "examples/live_exercises/bayes_quickstart/beliefs.yaml",
                "examples/live_exercises/bayes_quickstart/candidates.json",
            ],
            "optional_explanatory": [
                "examples/live_exercises/bayes_quickstart/app.py",
            ],
        },
        "belief_field_reference": "docs/BELIEF_INPUT_REFERENCE.md",
        "sample_logs": list(LOG_LINES),
        "log_summary": log_summary,
        "benchmark_preview": benchmark_preview(config),
        "next_step_preview_command": (
            "./bin/dev exec -- autoclanker beliefs preview "
            "--input examples/live_exercises/bayes_quickstart/beliefs.yaml"
        ),
        "next_step_shortest_replay_command": (
            "./bin/dev exec -- python scripts/live/replay_ideas_demo.py "
            "--exercise bayes_quickstart"
        ),
        "next_step_replay_command": (
            "./bin/dev exec -- python scripts/live/replay_bayes_quickstart.py"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the tiny log parser behind the Bayes quickstart exercise."
    )
    parser.add_argument(
        "--candidate-id",
        default="cand_a_default",
        choices=tuple(sorted(CANDIDATE_PRESETS)),
        help="Which candidate preset to render.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(render_candidate(args.candidate_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
