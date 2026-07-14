# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A uv-managed Python monorepo (Python 3.11+, pyright strict) shipping four console scripts plus a Rust crate:

- `autoclanker` — Bayesian guidance layer over agent eval loops (core in `autoclanker/bayes_layer/`)
- `bigbets` — portfolio compiler: one registry file → many rendered views (`bigbets/`)
- `clankerprof` — pprof CPU-profile analyzer (`clankerprof/`, Rust parity port in `crates/clankerprof-core/`)
- `goalloop` — deterministic goal loops for agent harnesses (`goalloop/`, contract in `docs/GOALLOOP.md`)

Library-first, CLI-first: every CLI is non-interactive, emits JSON to stdout by default, and uses stable nonzero exit codes (2 validation, 3 session, 4 adapter). `AGENTS.md`, `docs/SPEC.md` (normative), and `docs/STYLE.md` are the binding contributor contracts; keep `docs/` and `README.md` synchronized with behavior changes.

## Commands

`./bin/dev` is the canonical command surface (`make` and `mise` are wrappers over the same tasks; `bin/dev` falls back to inline equivalents when mise is unavailable).

```bash
./bin/dev setup        # uv sync --dev
./bin/dev format       # ruff check --fix . && ruff format .
./bin/dev lint         # ruff check .
./bin/dev typecheck    # pyright (strict mode; must stay clean)
./bin/dev pylint       # pylint --errors-only autoclanker  (autoclanker package only)
./bin/dev test         # default lane: pytest --cov-fail-under=90
./bin/dev test-full    # integration lane + default lane
./bin/dev test-rust    # cargo fmt-check + build + test (self-skips when cargo is absent)
./bin/dev check        # required gate = what CI runs: lint → typecheck → pylint → test-full → rust lane → build → strict-env validate
```

Test lanes (pytest markers in `pyproject.toml`):

- Default lane `addopts` bakes in `-m 'not integration and not upstream_live and not live'` and `--cov=autoclanker --cov=bigbets` (clankerprof is not a coverage target). It is self-contained: no upstreams or API keys (though with `cargo` installed, the Rust parity tests build `clankerprof-core`, which fetches crates on a cold cargo cache).
- `./bin/dev test-integration` — tests marked `integration` (only `tests/test_cli_integration.py`).
- `./bin/dev test-upstream-live` — non-billed; shallow-clones real autoresearch/cevolve into `.local/real-upstreams` and runs `-m upstream_live`.
- `AUTOCLANKER_ENABLE_LLM_LIVE=1 ./bin/dev test-live` — billed LLM lane; also needs `ANTHROPIC_API_KEY` (or `AUTOCLANKER_ANTHROPIC_API_KEY`). Never make the required gate billing-dependent or nondeterministic.
- `.env` / `.env.local` are auto-sourced by the test-lane scripts.

Single test:

```bash
uv run pytest tests/test_session_cli.py::test_name        # addopts still applies: coverage prints but no fail-under gate
uv run pytest tests/test_foo.py --no-cov                  # silence coverage output
uv run pytest -m integration tests/test_cli_integration.py  # marker-excluded tests need an -m override
```

`xfail_strict = true` is on. Note the trap: `tests/test_live_upstreams_unit.py` has no marker and runs in the default lane despite its name.

Rust (also runnable directly):

```bash
cargo test                                                            # clankerprof-core
cargo run -p clankerprof-core --bin clankerprof-rs -- facts …         # facts|targets|slices|scopes|compare|report
```

Python↔Rust parity tests (`tests/test_clankerprof_rust_parity.py`) run in the default pytest lane but skip silently when `cargo` is absent; `./bin/dev test-rust` (part of `check`) runs the native cargo lane and also self-skips without cargo.

## Architecture

### One umbrella CLI, four packages

`autoclanker/cli.py` is a thin argparse dispatcher: each subcommand family self-registers via a `register_*_commands(subparsers)` function that installs a `handler` returning a JSON-able dict. `bigbets`, `clankerprof`, and `goalloop` register their subcommands into it too (`autoclanker bigbets …`, `autoclanker pprof …`, `autoclanker goalloop …`) while also shipping standalone entry points. Dependency direction is one-way: the sibling packages never import from `autoclanker`. Exception to the dict-handler protocol: `goalloop` handlers print their own JSON and return exit codes, so the umbrella wraps them to raise `SystemExit` (see `_register_goalloop_family`).

Gotcha: the global `--output` flag is repositioned by `_normalize_global_output_position`; commands that define their own local `--output` must be listed in `_LOCAL_OUTPUT_COMMAND_PREFIXES` in `autoclanker/cli.py`.

### bayes_layer (the core engine)

Session pipeline, all wired in `bayes_layer/cli_session.py`; state is entirely file-based under `.autoclanker/<session_id>/` (`session_store.py`, filenames from `configs/default_bayes_layer.yaml`):

```
belief intake (belief_io) → canonicalization (deterministic or LLM provider; free text NEVER
updates the posterior directly) → compile/preview (belief_compiler → CompiledPriorBundle)
→ apply (gated: caller must echo the preview_digest from session init) → run/ingest evals
(eval_contract locks the benchmark surface; results appended to observations.jsonl)
→ fit (surrogate_objective: exact Bayesian linear posterior with explicit fallback_reason
when unsafe; surrogate_feasibility) → suggest (acquisition: constrained Thompson sampling
with deterministic optimistic fallback) → query_policy → commit_policy
```

Invariants from `docs/SPEC.md`: preview-then-apply is mandatory; canonicalization must work without any model provider; fallbacks are recorded in artifacts, never silent; eval results with a mismatched contract are rejected.

Adapters (`bayes_layer/adapters/`): `EvalLoopAdapter` protocol dispatched by `config.kind` — `fixture` (deterministic, keeps the repo self-contained), `autoresearch`/`cevolve` (real upstreams via `live_upstreams.py`, falling back to fixtures), `python_module`/`subprocess` (external). Upstream loops are adapters, never vendored source.

`schemas/*.schema.json` are the payload contracts, loaded and enforced at runtime with jsonschema (Draft 2020-12) via `validate_payload_against_schema` in `bayes_layer/belief_io.py`, layered with additional hand-written semantic validation (clankergraph validation alone is fully hand-rolled in `autoclanker/clankergraph.py`; keep it in sync with `schemas/clankergraph.schema.json`). `bayes_layer/config.py` locates defaults by requiring both `configs/` and `schemas/` dirs to exist.

### clankerprof (sample-facts architecture)

One durable fact layer; projections never walk raw profiles ad hoc:
`proto.py` (hand-rolled pprof protobuf decoder; strict about malformed input, signed int64s, value types) → `model.py` (`Profile.to_sample_facts()` with per-profile frame interning and memoized facts) → `facts.py` (versioned `clankerprof.sample_facts.v2` compact interned export/replay contract with strict import validation; legacy v1 imports still accepted) → projections in `targets.py`, `slices.py` (typed filter DSL with frame-identity memoization), and `scopes.py` (boundary decomposition), all built on `patterns.py` (path/regex/library matching), `categorize.py` (the shared categorization engine), and `stats.py` (projection accumulators); `analysis.py` is a compatibility shim re-exporting the historical surface → `render.py` / `compare.py` (strict-JSON regression gates dispatching on the report `tool` field). Runtime semantics (Ruby, generic) live in strict, versioned YAML rule packs under `clankerprof/runtime_rules/` (`clankerprof.runtime_rules.v1`), never hardcoded; Ruby is opt-in via `--runtime ruby`. The projection commands (targets/slices/scopes/boundaries) accept `--profile` or previously exported facts JSON (mutually exclusive) with identical output guaranteed by tests; `facts` requires `--profile` (compact by default, `--pretty` opt-in), and `compare` consumes two slice or scope reports.

The Rust crate (`crates/clankerprof-core`, binary `clankerprof-rs`) is capabilities-complete: facts export/replay, targets (all formats incl. the compat CSV pair), slices, scopes/boundaries with TOML/YAML configs, compare gates, and a single-pass `report` mode. It loads the same packaged rule packs via `include_str!` (drift is impossible by construction) and supports `--runtime ruby`, external packs, and core-class overrides. Python remains the reference implementation; the parity suite pins artifacts byte-for-byte across a per-subcommand flag matrix, and a test enforces `SAMPLE_FACTS_SCHEMA_VERSION` symmetry between `facts.py` and `facts.rs`. Any clankerprof behavior change must land in Python and Rust together and update the capability matrix in `docs/CLANKERPROF_PARITY.md`.

### goalloop

Host-neutral goal loops: `goalloop/model.py` holds the file-backed state contract (charter with YAML frontmatter gates/audit policy, wave-grouped tracker rows as the single source of execution state, append-only audit log and history JSONL); `goalloop/cli.py` exposes init/status/assert/gate/goal/handoff/audit. `goalloop goal` exits 0 only when all rows are `done`/`dropped`-with-reason, gates pass (exit codes propagate verbatim), and any configured adversarial audit has converged (a round confirming nothing new). Confirmed audit findings are appended to the tracker as `R<N>` waves. Everything is documented in `docs/GOALLOOP.md`; the operator workflow ships as `skills/goal-loop`.

### bigbets

`bigbets/core.py` is the whole engine: `load_bigbets_registry` parses one YAML/JSON registry into frozen dataclasses with heavy semantic validation (wave/priority consistency, P0 caps, link vocabularies), then `render_bigbets` derives all artifacts (JSON graph, CSV, Mermaid, SVG, Markdown, Excalidraw, static site via `site.py`). The registry is the single source of truth — derived artifacts are regenerated, never hand-edited. Schema versions live in `bigbets/version.py`; changing an emitted format requires bumping them. Host-neutral: no external service calls anywhere in the package.

## The compliance matrix (biggest non-obvious convention)

`tests/compliance_matrix.json` holds every product requirement (IDs like `M3-014`, gates `required`/`live`). Tests declare coverage with the `@covers("M#-###")` decorator from `tests/compliance.py`, and `tests/test_compliance_matrix.py` enforces: every required ID is covered by an unskipped, unmarked test; live IDs by live-lane tests; and `docs/COMPLIANCE_MATRIX.md` contains each entry's ID, gate, and description as literal substrings (one-way containment — stale extra rows in the doc are not caught).

Consequences: adding or changing a requirement means editing the JSON, the doc mirror, and a covering test together. Adding `skipif` to any required-gate `@covers` test breaks `./bin/dev check`; so does renaming/deleting a test that is the sole coverage for an ID — and the tests hard-pinned by node ID in `_FOCUSED_CONTRACT_CHECKS` (`tests/test_compliance_matrix.py`) break it on rename regardless of other coverage. Doc-sync tests assert literal substrings across `README.md`, `.github/CONTRIBUTING.md`, `docs/{INTEGRATIONS,ISSUE_SEEDER,HOST_ADAPTERS,BELIEF_INPUT_REFERENCE,LIVE_EXERCISES,STYLE}.md`, and `examples/` READMEs, and tie `skills/advanced-belief-author` to the actual CLI surface — rewording any of these files can fail `./bin/dev check`.

## Conventions

- Frozen, slotted dataclasses and `Protocol`s; no pydantic. Public APIs fully annotated; keep `py.typed`.
- CLI handlers return JSON-able dicts — no prose printing; JSON artifacts are written with sorted keys.
- Preferred deps: numpy/scipy/jsonschema/PyYAML; avoid pymc/stan/gpytorch and heavy orchestration frameworks.
- Era IDs are free-form strings, but cross-era prior decay (`_era_distance`, duplicated in `surrogate_objective.py` and `surrogate_feasibility.py`) gives graded distances only when both IDs are exactly `era_<digits>`; any other pair of distinct eras counts as distance 1.
- `skills/` contains operator-facing SKILL.md workflow packs (belief authoring, bigbets curation, clankerprof operation); update the matching skill when changing the CLI surface it documents.
- New code goes in the existing package trees; tests in `tests/` (flat, filename-mapped), docs in `docs/`, schemas in `schemas/`, default configs in `configs/`, example payloads in `examples/`.
