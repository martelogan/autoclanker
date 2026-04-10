from __future__ import annotations

import argparse
import io
import json
import tempfile

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import cast

from autoclanker.cli import main

ROOT = Path(__file__).resolve().parents[2]


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return cast(dict[str, object], value)


def _run_cli(argv: list[str]) -> dict[str, object]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(argv)
    if exit_code != 0:
        raise RuntimeError(
            f"autoclanker {' '.join(argv)!r} failed: {stderr.getvalue().strip()}"
        )
    return _require_mapping(json.loads(stdout.getvalue()))


def _top_candidate_id(payload: dict[str, object]) -> str:
    ranked_candidates = cast(list[object], payload["ranked_candidates"])
    return str(_require_mapping(ranked_candidates[0])["candidate_id"])


def _candidate_utility(payload: dict[str, object], candidate_id: str) -> float:
    ranked_candidates = cast(list[object], payload["ranked_candidates"])
    for item in ranked_candidates:
        mapping = _require_mapping(item)
        if mapping["candidate_id"] != candidate_id:
            continue
        value = mapping.get("predicted_utility")
        if not isinstance(value, int | float):
            raise ValueError("predicted_utility must be numeric.")
        return float(value)
    raise ValueError(f"Missing candidate {candidate_id!r}.")


def _init_and_apply(
    *,
    session_root: Path,
    session_id: str,
    ideas_json: str | None = None,
    canonicalization_model: str | None = None,
) -> None:
    argv = [
        "session",
        "init",
        "--session-id",
        session_id,
        "--era-id",
        "era_log_parser_v1",
        "--session-root",
        str(session_root),
    ]
    if ideas_json is not None:
        argv.extend(["--ideas-json", ideas_json])
    if canonicalization_model is not None:
        argv.extend(["--canonicalization-model", canonicalization_model])
    init_output = _run_cli(argv)
    preview_digest = init_output.get("preview_digest")
    if preview_digest is None:
        return
    _run_cli(
        [
            "session",
            "apply-beliefs",
            "--session-id",
            session_id,
            "--preview-digest",
            str(preview_digest),
            "--session-root",
            str(session_root),
        ]
    )


def _render_report() -> dict[str, object]:
    candidates_path = (
        ROOT / "examples" / "live_exercises" / "bayes_quickstart" / "candidates.json"
    )
    with tempfile.TemporaryDirectory(prefix="autoclanker-benchmark-") as tempdir:
        session_root = Path(tempdir) / "sessions"
        _init_and_apply(session_root=session_root, session_id="outer_control")
        _init_and_apply(
            session_root=session_root,
            session_id="proposal_only",
            ideas_json='["Try a moonbeam dragon refactor with kaleidoscope anchors."]',
        )
        _init_and_apply(
            session_root=session_root,
            session_id="deterministic_bayes",
            ideas_json='["Compiled regex matching probably helps this parser on repeated log formats.","Compiled matching works best together with the context pair plan."]',
        )
        _init_and_apply(
            session_root=session_root,
            session_id="hybrid_bayes",
            ideas_json='["A repeated-format fast path probably helps this parser."]',
            canonicalization_model="stub",
        )

        lanes = {}
        for session_id in (
            "outer_control",
            "proposal_only",
            "deterministic_bayes",
            "hybrid_bayes",
        ):
            payload = _run_cli(
                [
                    "session",
                    "suggest",
                    "--session-id",
                    session_id,
                    "--session-root",
                    str(session_root),
                    "--candidates-input",
                    str(candidates_path),
                ]
            )
            lanes[session_id] = {
                "top_candidate": _top_candidate_id(payload),
                "good_pair_predicted_utility": _candidate_utility(
                    payload, "cand_c_compiled_context_pair"
                ),
            }

    deterministic_gain = cast(
        float, lanes["deterministic_bayes"]["good_pair_predicted_utility"]
    ) - cast(float, lanes["outer_control"]["good_pair_predicted_utility"])
    hybrid_gain = cast(
        float, lanes["hybrid_bayes"]["good_pair_predicted_utility"]
    ) - cast(float, lanes["proposal_only"]["good_pair_predicted_utility"])
    return {
        "target": "bayes_quickstart_parser",
        "comparison_type": "zero_eval_cold_start",
        "cold_start_evals": 0,
        "lanes": lanes,
        "conclusion": {
            "control_vs_proposal_only_same": (
                lanes["outer_control"]["top_candidate"]
                == lanes["proposal_only"]["top_candidate"]
            ),
            "deterministic_bayes_improves_good_pair": deterministic_gain,
            "hybrid_bayes_improves_good_pair": hybrid_gain,
            "bayes_top_candidate": lanes["hybrid_bayes"]["top_candidate"],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare control, proposal-only, deterministic Bayes, and hybrid Bayes lanes on the parser quickstart target."
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON report instead of stdout.",
    )
    return parser.parse_args()


def main_cli() -> int:
    args = _parse_args()
    report = _render_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
