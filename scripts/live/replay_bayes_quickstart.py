from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile

from pathlib import Path
from typing import cast

from autoclanker.cli import main as autoclanker_main

ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the beginner-friendly Bayes quickstart exercise through the "
            "public autoclanker CLI."
        )
    )
    parser.add_argument(
        "--session-root",
        help="Optional session root. Defaults to a temporary directory.",
    )
    return parser.parse_args()


def _run_cli(argv: list[str]) -> dict[str, object]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exit_code = autoclanker_main(argv)
    if exit_code != 0:
        raise RuntimeError(
            f"autoclanker {' '.join(argv)!r} failed with exit code {exit_code}: "
            f"{stderr.getvalue().strip()}"
        )
    payload = json.loads(stdout.getvalue())
    if not isinstance(payload, dict):
        raise RuntimeError("Expected a JSON object from autoclanker CLI.")
    return cast(dict[str, object], payload)


def _candidate_by_id(
    ranked_candidates: list[dict[str, object]],
    candidate_id: str,
) -> dict[str, object]:
    for candidate in ranked_candidates:
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    raise RuntimeError(f"Missing candidate {candidate_id!r}.")


def main() -> int:
    args = _parse_args()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.session_root:
        session_root = Path(args.session_root)
        session_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="autoclanker-bayes-quickstart-")
        session_root = Path(temp_dir.name)

    try:
        beliefs_path = (
            ROOT / "examples" / "live_exercises" / "bayes_quickstart" / "beliefs.yaml"
        )
        candidates_path = (
            ROOT
            / "examples"
            / "live_exercises"
            / "bayes_quickstart"
            / "candidates.json"
        )

        init_output = _run_cli(
            [
                "session",
                "init",
                "--beliefs-input",
                str(beliefs_path),
                "--session-root",
                str(session_root),
            ]
        )
        belief_session_id = str(init_output["session_id"])
        preview_digest = str(init_output["preview_digest"])

        _run_cli(
            [
                "session",
                "init",
                "--session-id",
                "quickstart_control",
                "--era-id",
                "era_log_parser_v1",
                "--session-root",
                str(session_root),
            ]
        )
        _run_cli(
            [
                "session",
                "apply-beliefs",
                "--session-id",
                belief_session_id,
                "--preview-digest",
                preview_digest,
                "--session-root",
                str(session_root),
            ]
        )

        beliefs_payload = _run_cli(
            [
                "session",
                "suggest",
                "--session-id",
                belief_session_id,
                "--candidates-input",
                str(candidates_path),
                "--session-root",
                str(session_root),
            ]
        )
        control_payload = _run_cli(
            [
                "session",
                "suggest",
                "--session-id",
                "quickstart_control",
                "--candidates-input",
                str(candidates_path),
                "--session-root",
                str(session_root),
            ]
        )

        preview_payload = json.loads(
            (session_root / belief_session_id / "compiled_preview.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(preview_payload, dict):
            raise RuntimeError("Preview payload must be an object.")
        preview_mapping = cast(dict[str, object], preview_payload)
        preview_items = cast(list[object], preview_mapping["belief_previews"])
        preview_beliefs = [cast(dict[str, object], item) for item in preview_items]

        ranked_with_beliefs = [
            cast(dict[str, object], item)
            for item in cast(list[object], beliefs_payload["ranked_candidates"])
        ]
        ranked_control = [
            cast(dict[str, object], item)
            for item in cast(list[object], control_payload["ranked_candidates"])
        ]

        good_pair = _candidate_by_id(
            ranked_with_beliefs,
            "cand_c_compiled_context_pair",
        )
        lr_only = _candidate_by_id(ranked_with_beliefs, "cand_b_compiled_matcher")
        risky = _candidate_by_id(ranked_with_beliefs, "cand_d_wide_capture_window")
        proposal_preview = next(
            item for item in preview_beliefs if item.get("belief_id") == "qs4"
        )

        payload = {
            "backing_live_exercise": "examples/live_exercises/bayes_quickstart",
            "goal": (
                "Show that a few rough optimization ideas can change Bayes "
                "suggestions before any evals exist."
            ),
            "manual_replay_commands": [
                "./bin/dev exec -- python examples/live_exercises/bayes_quickstart/app.py",
                (
                    "./bin/dev exec -- autoclanker beliefs preview "
                    "--input examples/live_exercises/bayes_quickstart/beliefs.yaml"
                ),
                (
                    "./bin/dev exec -- autoclanker session init "
                    "--beliefs-input examples/live_exercises/bayes_quickstart/beliefs.yaml "
                    "--session-root .autoclanker-exercises"
                ),
                (
                    "./bin/dev exec -- autoclanker session apply-beliefs "
                    "--session-id quickstart_log_parser "
                    "--preview-digest <preview_digest_from_init> "
                    "--session-root .autoclanker-exercises"
                ),
                (
                    "./bin/dev exec -- autoclanker session suggest "
                    "--session-id quickstart_log_parser "
                    "--candidates-input examples/live_exercises/bayes_quickstart/candidates.json "
                    "--session-root .autoclanker-exercises"
                ),
                "./bin/dev exec -- python scripts/live/replay_bayes_quickstart.py",
            ],
            "observed": {
                "beliefs_top_candidate": ranked_with_beliefs[0]["candidate_id"],
                "control_top_candidate": ranked_control[0]["candidate_id"],
                "good_pair_margin_over_lr_only": float(good_pair["predicted_utility"])
                - float(lr_only["predicted_utility"]),
                "risky_valid_probability": float(risky["valid_probability"]),
                "risky_predicted_utility": float(risky["predicted_utility"]),
                "proposal_compile_status": proposal_preview["compile_status"],
            },
            "preview_summary": {
                "compiled_belief_count": sum(
                    1
                    for item in preview_beliefs
                    if item.get("compile_status") == "compiled"
                ),
                "metadata_only_belief_count": sum(
                    1
                    for item in preview_beliefs
                    if item.get("compile_status") == "metadata_only"
                ),
            },
            "session_root": str(session_root),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
