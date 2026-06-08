from __future__ import annotations

import argparse
import json
import re
import shlex

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from autoclanker.bayes_layer import load_serialized_payload
from autoclanker.bayes_layer.types import JsonValue, ValidationFailure

ISSUE_SEED_SCHEMA_VERSION = "autoclanker.issue-seed.v1"
RUN_CONTRACT_SCHEMA_VERSION = "autoclanker.run-contract.v1"


@dataclass(frozen=True, slots=True)
class IssueSeedArtifact:
    name: str
    url: str
    kind: str = "artifact"
    expected_path: str | None = None
    digest: str | None = None


@dataclass(frozen=True, slots=True)
class IssueSeedInput:
    title: str
    goal: str
    target_repo: str
    label: str = "agent-loop-idea-family"
    issue_number: int | None = None
    era_id: str | None = None
    session_id: str | None = None
    run_intensity: str = "mega"
    ideas: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    artifacts: tuple[IssueSeedArtifact, ...] = ()
    evidence_notes: str = ""
    adapter_config_path: str | None = None
    benchmark_snapshot: str | None = None
    corpus_snapshot: str | None = None
    canonicalization_mode: str | None = None
    canonicalization_model: str | None = None
    big_bet: str | None = None
    priority: str | None = None
    next_action: str | None = None


@dataclass(frozen=True, slots=True)
class IssueSeedBundle:
    issue_body: str
    autoclanker_ideas: dict[str, JsonValue]
    artifact_manifest: dict[str, JsonValue]
    run_contract: dict[str, JsonValue]
    lane_ledger: str
    pi_prompt: str
    headless_command: str
    host_adapter_contract: str

    def artifacts(self) -> dict[str, JsonValue]:
        return {
            "issue_body.md": self.issue_body,
            "autoclanker.ideas.json": self.autoclanker_ideas,
            "artifact-manifest.json": self.artifact_manifest,
            "run-contract.json": self.run_contract,
            "lane-ledger.md": self.lane_ledger,
            "pi.prompt.txt": self.pi_prompt,
            "headless-command.sh": self.headless_command,
            "host-adapter-contract.md": self.host_adapter_contract,
        }


def load_issue_seed_input(payload: Mapping[str, object]) -> IssueSeedInput:
    title = _required_string(payload, "title")
    goal = _required_string(payload, "goal")
    target_repo = _required_string(payload, "target_repo", "repo")
    return IssueSeedInput(
        title=title,
        goal=goal,
        target_repo=target_repo,
        label=_optional_string(payload, "label") or "agent-loop-idea-family",
        issue_number=_optional_positive_int(payload, "issue_number", "issue"),
        era_id=_optional_identifier(payload, "era_id"),
        session_id=_optional_identifier(payload, "session_id"),
        run_intensity=_optional_choice(
            payload,
            "run_intensity",
            choices=("standard", "deep", "mega"),
        )
        or _optional_choice(
            payload,
            "intensity",
            choices=("standard", "deep", "mega"),
        )
        or "mega",
        ideas=_string_tuple(payload, "ideas", "lanes"),
        constraints=_string_tuple(payload, "constraints"),
        artifacts=_artifact_tuple(payload.get("artifacts")),
        evidence_notes=_optional_string(payload, "evidence_notes", "evidence") or "",
        adapter_config_path=_optional_string(payload, "adapter_config_path"),
        benchmark_snapshot=_optional_string(payload, "benchmark_snapshot"),
        corpus_snapshot=_optional_string(payload, "corpus_snapshot"),
        canonicalization_mode=_optional_choice(
            payload,
            "canonicalization_mode",
            choices=("deterministic", "hybrid", "llm"),
        ),
        canonicalization_model=_optional_string(payload, "canonicalization_model"),
        big_bet=_optional_string(payload, "big_bet"),
        priority=_optional_string(payload, "priority"),
        next_action=_optional_string(payload, "next_action"),
    )


def build_issue_seed_bundle(seed: IssueSeedInput) -> IssueSeedBundle:
    seed_slug = _slug(seed.title or seed.goal)
    ideas = list(seed.ideas) or ["Infer the first candidate lane from evidence."]
    session_context: dict[str, JsonValue] = {}
    if seed.era_id:
        session_context["era_id"] = seed.era_id
    if seed.session_id:
        session_context["session_id"] = seed.session_id

    artifact_entries = [_artifact_payload(artifact) for artifact in seed.artifacts]
    if seed.benchmark_snapshot:
        artifact_entries.append(
            {
                "name": "benchmark-snapshot",
                "url": seed.benchmark_snapshot,
                "kind": "benchmark_snapshot",
            }
        )
    if seed.corpus_snapshot:
        artifact_entries.append(
            {
                "name": "corpus-snapshot",
                "url": seed.corpus_snapshot,
                "kind": "corpus_snapshot",
            }
        )

    autoclanker_ideas: dict[str, JsonValue] = {
        "ideas": [
            {
                "idea": idea,
                "confidence": 2,
                "evidence_sources": ["benchmark"]
                if artifact_entries
                else ["intuition"],
            }
            for idea in ideas
        ],
    }
    if session_context:
        autoclanker_ideas["session_context"] = session_context

    artifact_manifest: dict[str, JsonValue] = {
        "schema_version": ISSUE_SEED_SCHEMA_VERSION,
        "seed_slug": seed_slug,
        "target_repo": seed.target_repo,
        "title": seed.title,
        "issue_label": seed.label,
        "artifacts": cast(list[JsonValue], artifact_entries),
        "expected_workspace_files": [
            "autoclanker.ideas.json",
            "artifact-manifest.json",
            "run-contract.json",
            "lane-ledger.md",
            "pi.prompt.txt",
            "headless-command.sh",
            "host-adapter-contract.md",
        ],
    }
    if seed.issue_number is not None:
        artifact_manifest["issue_number"] = seed.issue_number
    if seed.big_bet:
        artifact_manifest["big_bet"] = seed.big_bet
    if seed.priority:
        artifact_manifest["priority"] = seed.priority

    run_contract: dict[str, JsonValue] = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "goal": seed.goal,
        "run_intensity": seed.run_intensity,
        "acceptance_gates": [
            "fixed_eval_surface_preserved",
            "evidence_artifacts_ingested_before_candidate_selection",
            "multi_lane_measurements_bound_to_candidate_identity",
            "posterior_fit_or_explicit_blocker_recorded",
            "proposal_or_blocker_recorded_before_exit",
        ],
        "evidence_policy": (
            "Artifact references are exploratory input signals. Promotion requires "
            "measured eval evidence under the locked eval contract."
        ),
        "promotion_policy": (
            "Promote independently reviewable proposals only after measured "
            "keep/reject/blocker decisions are recorded for relevant lanes."
        ),
        "stop_conditions": [
            "one_or_more_proposals_ready",
            "all_plausible_lanes_rejected_or_blocked",
            "true_hard_blocker_recorded",
        ],
        "evidence_intake_checklist": [
            "materialize_or_read_artifact_manifest_references",
            "summarize_clankergraph_benchmark_corpus_and_prior_run_signals",
            "turn_evidence_into_explicit_candidate_lanes_before_measurement",
            "record_lane_ledger_state_before_first_eval",
            "bind_every_eval_result_to_candidate_identity",
        ],
        "required_session_outputs": [
            "updated lane-ledger.md",
            "posterior or explicit blocker summary",
            "measured keep/reject/blocker decision per attempted lane",
            "draft proposal links for independently reviewable candidates",
        ],
    }
    if seed.constraints:
        run_contract["constraints"] = list(seed.constraints)
    if seed.adapter_config_path:
        run_contract["adapter_config_path"] = seed.adapter_config_path
    if seed.canonicalization_mode:
        run_contract["canonicalization_mode"] = seed.canonicalization_mode
    if seed.canonicalization_model:
        run_contract["canonicalization_model"] = seed.canonicalization_model

    lane_ledger = _lane_ledger(ideas, seed.next_action)
    pi_prompt = _pi_prompt(seed)
    headless_command = _headless_command(seed)
    host_adapter_contract = _host_adapter_contract()
    issue_body = _issue_body(
        seed,
        artifact_manifest=artifact_manifest,
        run_contract=run_contract,
        lane_ledger=lane_ledger,
        pi_prompt=pi_prompt,
        headless_command=headless_command,
        host_adapter_contract=host_adapter_contract,
    )
    return IssueSeedBundle(
        issue_body=issue_body,
        autoclanker_ideas=autoclanker_ideas,
        artifact_manifest=artifact_manifest,
        run_contract=run_contract,
        lane_ledger=lane_ledger,
        pi_prompt=pi_prompt,
        headless_command=headless_command,
        host_adapter_contract=host_adapter_contract,
    )


def bundle_payload(bundle: IssueSeedBundle) -> dict[str, JsonValue]:
    return {
        "ok": True,
        "tool": "autoclanker_issue_seed",
        "schema_version": ISSUE_SEED_SCHEMA_VERSION,
        "artifacts": bundle.artifacts(),
    }


def handle_generate(args: argparse.Namespace) -> dict[str, JsonValue]:
    seed = load_issue_seed_input(load_serialized_payload(Path(args.input)))
    bundle = build_issue_seed_bundle(seed)
    payload = bundle_payload(bundle)
    output_dir = cast(str | None, getattr(args, "output_dir", None))
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for name, content in bundle.artifacts().items():
            path = output_path / name
            if isinstance(content, str):
                path.write_text(content, encoding="utf-8")
            else:
                path.write_text(
                    json.dumps(content, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if name.endswith(".sh"):
                path.chmod(path.stat().st_mode | 0o111)
        payload["output_dir"] = str(output_path)
    return payload


def register_issue_seed_commands(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "issue-seed",
        help="Generate ready-to-run optimization issue seed artifacts.",
    )
    issue_subparsers = parser.add_subparsers(dest="issue_seed_command", required=True)
    generate_parser = issue_subparsers.add_parser(
        "generate",
        help="Generate issue body, autoclanker ideas, run contract, lane ledger, and kickoff prompts.",
    )
    generate_parser.add_argument("--input", required=True)
    generate_parser.add_argument(
        "--output-dir",
        help="Optional directory to materialize the generated artifact bundle.",
    )
    generate_parser.set_defaults(handler=handle_generate)


def _artifact_payload(artifact: IssueSeedArtifact) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "name": artifact.name,
        "url": artifact.url,
        "kind": artifact.kind,
    }
    if artifact.expected_path:
        payload["expected_path"] = artifact.expected_path
    if artifact.digest:
        payload["digest"] = artifact.digest
    return payload


def _lane_ledger(ideas: Sequence[str], next_action: str | None) -> str:
    rows: list[str] = []
    for index, idea in enumerate(ideas):
        escaped_idea = idea.replace("|", "\\|")
        rows.append(
            f"| lane-{index + 1} | active | {escaped_idea} | pending first measurement |"
        )
    if not rows:
        rows = [
            "| lane-1 | active | Infer the first candidate lane from evidence | pending first measurement |"
        ]
    next_line = next_action or "Run the strongest evidence-backed lane first."
    return (
        "# Lane Ledger\n\n"
        "| Lane | Status | Hypothesis | Latest decision |\n"
        "| --- | --- | --- | --- |\n" + "\n".join(rows) + "\n\n"
        f"Next action: {next_line}\n\n"
        "Update this before first measurement, after each keep/reject/blocker "
        "decision, and before stopping.\n"
    )


def _pi_prompt(seed: IssueSeedInput) -> str:
    ideas_arg = "--ideas-input autoclanker.ideas.json"
    intensity = "--overnight" if seed.run_intensity in {"deep", "mega"} else ""
    return f"""Use pi-autoclanker to run this seeded optimization issue.

Goal: {seed.goal}

Read autoclanker.ideas.json, artifact-manifest.json, run-contract.json,
lane-ledger.md, clankergraph files, benchmark snapshots, and corpus snapshots
before choosing candidate lanes. Start or resume with /autoclanker run
{intensity} {ideas_arg}. The slash command prepares or resumes the workspace;
read and execute its returned handoffPrompt.

During execution, keep candidate lanes explicit, preserve the fixed eval
surface, bind every multi-lane eval ingest to the measured candidate, use
evidence artifacts as search inputs, fit/suggest between measurements, and
continue until each valuable lane has a measured keep/reject/blocker decision
or an independently reviewable proposal is ready."""


def _headless_command(seed: IssueSeedInput) -> str:
    intensity = "--overnight" if seed.run_intensity in {"deep", "mega"} else ""
    canonicalization_mode = seed.canonicalization_mode or "deterministic"
    session_default = seed.session_id or "seeded-run"
    era_default = seed.era_id or "seeded-era"
    adapter_arg = (
        f" \\\n  --adapter-config {shlex.quote(seed.adapter_config_path)}"
        if seed.adapter_config_path
        else ""
    )
    apply_adapter_arg = (
        f" \\\n#   --adapter-config {shlex.quote(seed.adapter_config_path)}"
        if seed.adapter_config_path
        else ""
    )
    model_arg = (
        f" \\\n  --canonicalization-model {shlex.quote(seed.canonicalization_model)}"
        if seed.canonicalization_model
        else ""
    )
    return f"""AUTOCLANKER_SESSION_ID="${{AUTOCLANKER_SESSION_ID:-{session_default}}}"
AUTOCLANKER_ERA_ID="${{AUTOCLANKER_ERA_ID:-{era_default}}}"

autoclanker session init \\
  --beliefs-input autoclanker.ideas.json \\
  --session-id "$AUTOCLANKER_SESSION_ID" \\
  --era-id "$AUTOCLANKER_ERA_ID" \\
  --canonicalization-mode {canonicalization_mode}{model_arg}{adapter_arg} > autoclanker.init.json

# Then apply the preview digest from autoclanker.init.json before evals:
# autoclanker session apply-beliefs --session-id "$AUTOCLANKER_SESSION_ID" \\
#   --preview-digest "<preview_digest>"{apply_adapter_arg}

pi-autoclanker command run {intensity} \\
  --workspace "$PWD" \\
  --ideas-input autoclanker.ideas.json > autoclanker.run.json

# Give the supervising agent the handoffPrompt from autoclanker.run.json.
# The command prepares or resumes state; it is not the full autonomous run."""


def _host_adapter_contract() -> str:
    return """# Optional Host Adapter Contract

The default issue seeder is static and local-first. A hosted deployment may add
an adapter, but must keep secrets out of browser-local seed JSON.

Recommended optional host methods:

- `loadSeeds({ appId })`
- `saveSeed({ appId, seed })`
- `uploadArtifact({ appId, name, contentType, body })`
- `fetchIssue({ repo, issue, url })`
- `createIssue({ repo, title, body, labels })`
- `canonicalizeIdeas({ seed, artifacts })`

LLM provider keys belong behind `canonicalizeIdeas` or the CLI environment, not
in the static browser app. Artifact uploads should return immutable URLs or
object keys that can be copied into the generated issue body.
"""


def _issue_body(
    seed: IssueSeedInput,
    *,
    artifact_manifest: Mapping[str, JsonValue],
    run_contract: Mapping[str, JsonValue],
    lane_ledger: str,
    pi_prompt: str,
    headless_command: str,
    host_adapter_contract: str,
) -> str:
    artifacts = cast(Sequence[Mapping[str, JsonValue]], artifact_manifest["artifacts"])
    artifact_lines = (
        "\n".join(
            f"- {item.get('name')}: {item.get('url')} ({item.get('kind')})"
            for item in artifacts
        )
        if artifacts
        else "- No external artifacts declared yet."
    )
    lane_lines = (
        "\n".join(f"- active: {idea}" for idea in seed.ideas)
        if seed.ideas
        else "- Seed lanes should be inferred during intake."
    )
    constraint_lines = (
        "\n".join(f"- {constraint}" for constraint in seed.constraints)
        if seed.constraints
        else "- Preserve correctness, rollback safety, and the locked eval surface."
    )
    bigbets_block = ""
    if seed.big_bet or seed.priority or seed.next_action:
        bigbets_block = f"""
<!-- bigbets:idea-family
slug: {_slug(seed.title)}
big_bet: {seed.big_bet or "unmapped"}
priority: {seed.priority or "P1"}
status: active
role: ideas-lane
artifact: autoclanker.ideas.json
next_action: {seed.next_action or "Run the strongest evidence-backed lane first."}
-->
"""
    return f"""## Start Here

This issue seeds a long-running optimization exploration for:

{seed.goal}

Use the artifact references and run contract below as the current source of
truth. Start or resume the workspace, execute the returned autoclanker handoff
prompt, compare multiple lanes when practical, and post draft proposals or
blockers as they become independently reviewable.

<details>
<summary>Local Pi Kickoff</summary>

```text
{pi_prompt.strip()}
```

</details>

<details>
<summary>Headless CLI Kickoff</summary>

```bash
{headless_command.strip()}
```

</details>

<details>
<summary>Seed Artifacts</summary>

{artifact_lines}

```json
{json.dumps(artifact_manifest, indent=2, sort_keys=True)}
```

</details>

<details>
<summary>Seed Lanes</summary>

{lane_lines}

</details>

<details>
<summary>Constraints</summary>

{constraint_lines}

</details>

<details>
<summary>Run Contract</summary>

```json
{json.dumps(run_contract, indent=2, sort_keys=True)}
```

</details>

<details>
<summary>Lane Ledger</summary>

```markdown
{lane_ledger.strip()}
```

</details>

<details>
<summary>Optional Host Adapter Contract</summary>

```markdown
{host_adapter_contract.strip()}
```

</details>

## Evidence Notes

{seed.evidence_notes or "No additional evidence notes yet."}

## Target

- Repo: {seed.target_repo}
- Suggested title: {seed.title}
- Suggested label: {seed.label}
{bigbets_block}"""


def _required_string(payload: Mapping[str, object], *keys: str) -> str:
    value = _optional_string(payload, *keys)
    if value is None:
        raise ValidationFailure(f"Issue seed input is missing {keys[0]!r}.")
    return value


def _optional_string(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValidationFailure(f"Expected {key!r} to be a string.")
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _optional_choice(
    payload: Mapping[str, object], key: str, *, choices: Sequence[str]
) -> str | None:
    value = _optional_string(payload, key)
    if value is None:
        return None
    if value not in choices:
        expected = ", ".join(choices)
        raise ValidationFailure(f"Expected {key!r} to be one of: {expected}.")
    return value


def _optional_identifier(payload: Mapping[str, object], key: str) -> str | None:
    value = _optional_string(payload, key)
    if value is None:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise ValidationFailure(
            f"Expected {key!r} to contain only letters, numbers, '.', '_', ':', or '-'."
        )
    return value


def _optional_positive_int(payload: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.strip().isdigit() and int(value) > 0:
            return int(value)
        raise ValidationFailure(f"Expected {key!r} to be a positive integer.")
    return None


def _string_tuple(payload: Mapping[str, object], *keys: str) -> tuple[str, ...]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return tuple(line.strip() for line in value.splitlines() if line.strip())
        if not isinstance(value, list):
            raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
        items: list[str] = []
        for item in cast(list[object], value):
            if not isinstance(item, str):
                raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
            normalized = item.strip()
            if normalized:
                items.append(normalized)
        return tuple(items)
    return ()


def _artifact_tuple(value: object) -> tuple[IssueSeedArtifact, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValidationFailure("Expected 'artifacts' to be a list.")
    artifacts: list[IssueSeedArtifact] = []
    for index, item in enumerate(cast(list[object], value)):
        if isinstance(item, str):
            url = item.strip()
            if not url:
                raise ValidationFailure("Artifact URL strings must be non-empty.")
            artifacts.append(IssueSeedArtifact(name=f"artifact-{index + 1}", url=url))
            continue
        if not isinstance(item, Mapping):
            raise ValidationFailure("Each artifact must be a string or object.")
        mapping = cast(Mapping[str, object], item)
        artifacts.append(
            IssueSeedArtifact(
                name=_required_string(mapping, "name"),
                url=_required_string(mapping, "url"),
                kind=_optional_string(mapping, "kind") or "artifact",
                expected_path=_optional_string(mapping, "expected_path"),
                digest=_optional_string(mapping, "digest"),
            )
        )
    return tuple(artifacts)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:80] or "optimization-seed"
