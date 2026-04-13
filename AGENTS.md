# AGENTS.md

## Repository expectations

This repository is for the project: `autoclanker`.

### Canonical workflow

- Use `./bin/dev` as the primary command surface.
- `make` is only a compatibility layer.
- Keep code inside the existing top-level package directory.
- Keep tests in `tests/`.
- Keep docs in `docs/`.
- Keep default runtime config in `configs/`.
- Keep schemas in `schemas/`.
- Keep example payloads in `examples/`.

### Product stance

Build this as a **library-first, CLI-first, extension-ready** system.

### Quality bar

After each milestone, run the smallest relevant validation slice. Before declaring the task done, run:

- `./bin/dev format`
- `./bin/dev lint`
- `./bin/dev typecheck`
- `./bin/dev pylint`
- `./bin/dev test`
- `./bin/dev test-full`
- `./bin/dev check`

### Typing and style

- Keep public functions fully annotated.
- Preserve compatibility with `pyright` strict mode.
- Prefer small, typed modules over large multi-purpose modules.
- Use `dataclasses`, `typing`, and `Protocol` where appropriate.
- Maintain `py.typed` support in the top-level package.

### Preferred dependencies for v1

Preferred runtime dependencies, if needed:

- `numpy`
- `scipy`
- `jsonschema`
- `PyYAML`

Allowed if they materially simplify implementation:

- `scikit-learn`
- `networkx`

Avoid in v1 unless already present in the repo:

- `pymc`
- `stan`
- `gpytorch`
- heavyweight distributed orchestration frameworks

### autoclanker rules

- Keep the outer search / evolutionary loop.
- Build an **era-local** Bayesian layer.
- Use the dual-lane typed belief API from `schemas/human_belief.schema.json`.
- Support both `expert_prior` and `graph_directive` for advanced users.
- Always provide a compiled-prior preview before applying human beliefs.
- Emit eval results that validate against `schemas/eval_result.schema.json`.
- Keep deterministic tests separate from noisy benchmark logic.
- Default users must not be required to think about graph structure or direct prior scales.
- Treat upstream eval loops as adapters, not as vendored source.
- Ship first-party adapter support for `autoresearch` and `cevolve` from the outset.
- Keep the repo self-contained through fixture-backed tests when real upstream paths are absent.

### CLI contract

The primary UX is the `autoclanker` CLI.

Required behavior:
- non-interactive only;
- accept `--input` file paths or stdin where practical;
- accept JSON and YAML inputs;
- emit JSON to stdout by default for machine-readable commands;
- support `--output` when writing artifacts;
- keep human-readable noise low;
- use stable nonzero exit codes for validation or session errors.

Suggested command families:
- `beliefs validate`
- `beliefs preview`
- `beliefs compile`
- `eval validate`
- `session init`
- `session ingest-eval`
- `session fit`
- `session suggest`
- `session recommend-commit`
- `session status`
- `adapter list`
- `adapter probe`
- `adapter validate-config`

### Session and adapter contract

Implement a small `SessionStore` protocol plus one local-filesystem implementation.

Requirements:
- configurable session root;
- default root `.autoclanker/`;
- append-only `observations.jsonl`;
- stable JSON/YAML artifacts for manifest, preview, compiled priors, posterior summary, query, and commit decision;
- no hardcoded assumption that sessions must live under one fixed directory.

A future adapter must be able to place the store under another tool’s session path, for example:
- `.autoclanker/<session_id>/`
- `.cevolve/<session_id>/autoclanker/`

Implement an adapter protocol plus:
- a fixture adapter for tests,
- a first-party `autoresearch` adapter,
- a first-party `cevolve` adapter.

Adapters should support local-path or import-based integration when upstream code is available, but fixture-backed tests must keep the repo self-contained when it is not.

### File placement

Use this module tree:

```text
autoclanker/
  bayes_layer/
    __init__.py
    config.py
    types.py
    registry.py
    belief_io.py
    belief_compiler.py
    feature_encoder.py
    surrogate_objective.py
    surrogate_feasibility.py
    posterior_graph.py
    acquisition.py
    query_policy.py
    commit_policy.py
    session_store.py
    cli_beliefs.py
    cli_eval.py
    cli_session.py
    cli_adapter.py
    adapters/
      __init__.py
      protocols.py
      fixture.py
      autoresearch.py
      cevolve.py
```

### Documentation discipline

Keep these synchronized with the code:

- `README.md`
- `.github/CONTRIBUTING.md`
- `docs/STYLE.md`
- `docs/SPEC.md`
- `docs/DESIGN.md`
- `docs/INTEGRATIONS.md`
- `docs/BELIEF_INPUT_REFERENCE.md`
- `docs/LIVE_EXERCISES.md`
- `docs/TOY_EXAMPLES.md`
- `docs/WHITEPAPER.md`
- `docs/COMPLIANCE_MATRIX.md`
- examples and schemas
