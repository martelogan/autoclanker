from __future__ import annotations

import argparse
import json

from pathlib import Path

from autoclanker.bayes_layer import load_serialized_payload
from autoclanker.bayes_layer.adapters.fixture import FixtureAdapter
from autoclanker.bayes_layer.types import (
    GeneStateRef,
    ValidAdapterConfig,
    to_json_value,
)

DEFAULT_CANDIDATE_SEEDS: dict[str, int] = {
    "cand_a_default": 11,
    "cand_b_compiled_matcher": 13,
    "cand_d_wide_window_large_chunk": 17,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic fixture evals for the bayes_complex showcase."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where eval JSON files will be written.",
    )
    parser.add_argument(
        "--candidates-input",
        default="examples/live_exercises/bayes_complex/candidates.json",
        help="Candidate payload to read.",
    )
    parser.add_argument(
        "--era-id",
        default="era_parser_advanced",
        help="Era identifier to stamp into the generated evals.",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        help=(
            "Candidate ID to generate. Defaults to the named observation candidates "
            "used by the showcase regression."
        ),
    )
    return parser.parse_args()


def _candidate_payload_to_refs(
    payload: dict[str, object],
) -> dict[str, tuple[GeneStateRef, ...]]:
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("Candidate payload must contain a 'candidates' list.")
    parsed: dict[str, tuple[GeneStateRef, ...]] = {}
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise ValueError("Each candidate must be an object.")
        candidate_id = raw_candidate.get("candidate_id")
        raw_genotype = raw_candidate.get("genotype")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("Each candidate must provide a non-empty candidate_id.")
        if not isinstance(raw_genotype, list):
            raise ValueError(
                f"Candidate {candidate_id!r} must provide a genotype list."
            )
        genotype: list[GeneStateRef] = []
        for raw_ref in raw_genotype:
            if not isinstance(raw_ref, dict):
                raise ValueError(
                    f"Candidate {candidate_id!r} contains a non-object genotype entry."
                )
            gene_id = raw_ref.get("gene_id")
            state_id = raw_ref.get("state_id")
            if not isinstance(gene_id, str) or not gene_id.strip():
                raise ValueError(
                    f"Candidate {candidate_id!r} contains an empty gene_id."
                )
            if not isinstance(state_id, str) or not state_id.strip():
                raise ValueError(
                    f"Candidate {candidate_id!r} contains an empty state_id."
                )
            genotype.append(
                GeneStateRef(gene_id=gene_id.strip(), state_id=state_id.strip())
            )
        parsed[candidate_id] = tuple(genotype)
    return parsed


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_payload = load_serialized_payload(Path(args.candidates_input))
    candidates = _candidate_payload_to_refs(candidate_payload)
    requested_ids = tuple(args.candidate_id or DEFAULT_CANDIDATE_SEEDS.keys())

    adapter = FixtureAdapter(
        ValidAdapterConfig(kind="fixture", mode="fixture", session_root=".autoclanker")
    )
    for candidate_id in requested_ids:
        genotype = candidates.get(candidate_id)
        if genotype is None:
            known_ids = ", ".join(sorted(candidates))
            raise ValueError(
                f"Unknown candidate_id {candidate_id!r}. Known candidates: {known_ids}."
            )
        eval_result = adapter.evaluate_candidate(
            era_id=str(args.era_id),
            candidate_id=candidate_id,
            genotype=genotype,
            seed=DEFAULT_CANDIDATE_SEEDS.get(candidate_id, 0),
        )
        output_path = output_dir / f"{candidate_id}.json"
        output_path.write_text(
            json.dumps(to_json_value(eval_result), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
