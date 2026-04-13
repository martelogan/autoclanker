# SPEC.md

This file is normative. Code should match this behavior unless the repository context forces an equivalent implementation.

## 1. Non-negotiable implementation decisions

1. Keep the outer evolutionary / search loop.
2. Add an **era-local** Bayesian guidance layer.
3. Build the system as **library-first, CLI-first, and extension-ready**.
4. Build `autoclanker` as a standalone project, not as a vendored fork of upstream loops.
5. Default users must **not** be required to think about posterior graphs, direct prior scales, or internal acquisition details.
6. Use a **dual-lane typed belief API** with schema validation.
7. Allow natural language only as:
   - optional rationale,
   - optional proposal text before canonicalization,
   - rough ideas that must canonicalize into typed beliefs or remain metadata-only.
8. Free text must never directly influence the posterior.
9. Require a **compiled-prior preview / round-trip** before applying belief batches.
10. Default to a sparse Bayesian interaction model for objective utility.
11. Use a separate feasibility model.
12. Use decayed human priors.
13. Use bounded active querying.
14. Emit structured eval results and structured session artifacts.
15. Separate deterministic tests from noisy benchmark logic.
16. Support a generic adapter contract plus first-party adapters for `autoresearch` and `cevolve`.
17. Keep the repo self-contained with fixture-backed tests when upstream repos are absent.

## 2. Project and package contract

### 2.1 Package placement

Put runtime code under the top-level package directory:

```text
autoclanker/
```

Do not create a second competing package root.

### 2.2 CLI is the primary UX in v1

At minimum, expose commands or functionally equivalent entry points for:

- validating human-belief payloads,
- previewing compiled priors,
- compiling belief batches,
- validating eval-result payloads,
- validating adapter configs,
- listing / probing adapters,
- initializing a Bayesian session,
- ingesting new eval results,
- fitting / refreshing the posterior state,
- suggesting next actions or next candidates,
- recommending commit / no-commit decisions,
- reading session status.

Suggested shape:

```text
autoclanker beliefs validate
autoclanker beliefs canonicalize-ideas
autoclanker beliefs expand-ideas
autoclanker beliefs preview
autoclanker beliefs compile
autoclanker eval validate
autoclanker adapter registry
autoclanker adapter validate-config
autoclanker adapter list
autoclanker adapter probe
autoclanker adapter surface
autoclanker session init
autoclanker session apply-beliefs
autoclanker session ingest-eval
autoclanker session run-eval
autoclanker session run-frontier
autoclanker session fit
autoclanker session suggest
autoclanker session frontier-status
autoclanker session recommend-commit
autoclanker session render-report
autoclanker session status
```

### 2.3 Machine-readable I/O contract

The CLI must be non-interactive.

Required behavior:
- accept `--input` paths where appropriate;
- support stdin where practical for payload-style commands;
- accept JSON and YAML inputs for belief batches and adapter configs;
- accept inline JSON rough-idea input where practical for beginner flows;
- emit JSON to stdout by default for machine-readable commands;
- support `--output` to write artifacts to files;
- return stable nonzero exit codes for validation errors and session errors.

### 2.4 Session-store and adapter boundary

The implementation must define a `SessionStore` protocol and at least one local-filesystem implementation.

Requirements:
- configurable root directory;
- default root `.autoclanker/`;
- support alternate placement under another tool’s session directory;
- append-only observation logging via `observations.jsonl`;
- structured JSON or YAML artifacts for session manifest, locked eval contract, belief batch, preview, compiled priors, posterior summary, query, frontier status, and commit decision;
- per-eval execution records under `eval_runs/` when the session itself executes candidates;
- a human-readable `RESULTS.md` artifact describing the current run state;
- rendered report artifacts for convergence, candidate rankings, and prior-vs-posterior graph views.

The adapter boundary must make it trivial for a future wrapper or companion tool to place session files under:
- `.autoclanker/<session_id>/`, or
- `.cevolve/<session_id>/autoclanker/`.

### 2.5 Validation surface

The implementation must support the repo-native commands:

- `./bin/dev format`
- `./bin/dev lint`
- `./bin/dev typecheck`
- `./bin/dev pylint`
- `./bin/dev test`
- `./bin/dev test-full`
- `./bin/dev test-upstream-live`
- `./bin/dev test-live`
- `./bin/dev test-max-live`
- `./bin/dev check`

Validation-lane rules:
- `./bin/dev check` remains the self-contained required gate;
- `./bin/dev test-upstream-live` is the non-billed real-upstream lane;
- `./bin/dev test-live` is the billed real-model-provider lane;
- `./bin/dev test-max-live` runs both optional live lanes;
- if the relevant upstream checkouts or model credentials are present, those optional live lanes must be run before claiming full live coverage.

## 3. Required external interfaces

### 3.1 Belief ingestion

```python
ingest_human_beliefs(payload: dict) -> ValidatedBeliefBatch
```

Requirements:
- validate against `schemas/human_belief.schema.json`;
- canonicalize registry references;
- reject invalid or ambiguous control fields;
- return a typed internal representation.

Rough ideas must enter through an explicit canonicalization step before they become active priors.

```python
canonicalize_belief_input(payload: dict, ...) -> CanonicalizationOutcome
```

Requirements:
- deterministic canonicalization must work without a model provider;
- optional model assistance must be provider-agnostic and produce inspectable typed artifacts;
- unresolved ideas must remain proposals or metadata-only suggestions;
- session-local surface overlays must be explicit artifacts, not hidden state.
- when a real model provider is configured, the public CLI must support a billed live canonicalization path over at least one shipped example target.

### 3.2 Compiled-prior preview

```python
preview_compiled_beliefs(
    beliefs: ValidatedBeliefBatch,
    registry: GeneRegistry,
    era_state: EraState,
) -> CompiledPriorPreview
```

Requirements:
- compile using the same mapping logic as the real compiler;
- validate against `schemas/compiled_prior_preview.schema.json`;
- expose warnings, metadata-only items, and rejected items;
- expose numeric prior targets in optimizer-relevant units;
- expose graph hints when present.

### 3.3 Belief compilation

```python
compile_beliefs(
    beliefs: ValidatedBeliefBatch,
    registry: GeneRegistry,
    era_state: EraState,
) -> CompiledPriorBundle
```

The result must include:
- main-effect priors,
- pair priors,
- feasibility priors,
- hard masks / exclusions,
- optional candidate-generation hints.

### 3.4a Evaluation-contract capture

```python
capture_eval_contract(adapter_config: ValidAdapterConfig, ...) -> EvalContractSnapshot
```

Requirements:
- digest the benchmark tree, eval harness, adapter config, and environment inputs;
- resolve an effective eval policy for measured execution, including whether the
  benchmark should take a contract-scoped lease and whether soft stabilization
  is enabled;
- persist the locked contract at `session init`;
- expose current-versus-locked drift in session status;
- reject missing or mismatched digests during `session ingest-eval` for newly hardened sessions.

### 3.5 Eval-result validation

```python
validate_eval_result(payload: dict) -> ValidEvalResult
```

Requirements:
- validate against `schemas/eval_result.schema.json`;
- preserve intended and realized genotypes;
- preserve `patch_hash`;
- preserve optional eval-contract echoes and execution metadata;
- support repeated observations of the same realized configuration.

### 3.6 Adapter-config validation

```python
validate_adapter_config(payload: dict) -> ValidAdapterConfig
```

Requirements:
- validate against `schemas/adapter_config.schema.json`;
- support fixture, autoresearch, cevolve, and generic external modes;
- support local-path or import-based configuration;
- expose clean errors for missing paths or unsupported modes.

### 3.7 Frontier inputs and family-aware suggestion

`session suggest` must accept either the legacy candidate-pool shape:

```json
{"candidates": [...]}
```

or a richer frontier document with lineage and family metadata.

Requirements:
- preserve candidate `family_id`, `origin_kind`, parent refs, notes, and budget weight;
- persist a stable `frontier_status.json` artifact;
- expose frontier summaries through `session suggest` and `session frontier-status`;
- keep legacy candidate pools valid by normalizing them into a default frontier family.

### 3.8 Generic adapter protocol

Implement a typed adapter protocol that can support:
- registry discovery,
- eval-contract capture,
- candidate materialization,
- evaluation under an explicit execution context,
- optional commit,
- optional rethink summary hooks,
- status / probe reporting.

The protocol must be generic enough for future external loops and concrete enough to ship first-party support for `autoresearch` and `cevolve`.

## 4. Modeling rules

### 4.1 Objective surrogate

Use a Bayesian linear interaction model over:
- main effects,
- a screened subset of pair effects,
- small global features such as active-gene count.

### 4.2 Feasibility surrogate

Use a separate feasibility model, ideally Bayesian logistic or a robust regularized approximation.

### 4.3 Utility target

Model utility rather than raw benchmark score:

```text
utility = delta_perf - lambda_sparsity * complexity - lambda_vram * soft_vram_overage
```

### 4.4 Replication and aggregation

- aggregate repeated observations by `patch_hash`;
- preserve count and empirical variance where possible;
- reserve replication for baseline, incumbent, near-commit, and high-value uncertain candidates.

## 5. User-lane rules

### 5.1 Simple lane

Default users should mostly work through:
- canonical IDs,
- bounded enums,
- relation hints,
- optional rationale,
- YAML-friendly payloads.

### 5.2 Expert lane

Power users may additionally specify:
- `expert_prior`,
- `graph_directive`,
- direct prior or exclusion overrides within the schema bounds.

### 5.3 Preview requirement

Preview remains mandatory before apply.

### 5.4 Common advanced-authoring workflow

The common expert-authoring path should remain:

```text
rough ideas
→ canonicalize-ideas
→ expand-ideas (optional alias)
→ preview
→ assistant-guided refinement only where needed
→ expert_prior / graph_directive
→ apply
```

Requirements:
- the assistant path must still produce typed inspectable beliefs, not hidden free-text influence;
- JSON should be the default machine-authored advanced belief format;
- YAML may remain available when hand editing is the expected next step.

## 6. Integration rules

### 6.1 Fixture adapter

A self-contained fixture adapter is required for deterministic tests and local development.

### 6.2 First-party upstream adapters

Ship first-party adapter modules for:
- `autoresearch`
- `cevolve`

They must not make the repo untestable when upstream repos are absent.
They should support `mode: auto` as the default front door, so users can bind
through a checkout path, an installed adapter module, or a runnable adapter command.

### 6.3 Optional local upstream checkout

If a local upstream path is configured, the adapter should be able to probe and use
it. But checkout paths are optional, not mandatory. If the path is absent, the error
must be explicit and optional tests must skip cleanly.
