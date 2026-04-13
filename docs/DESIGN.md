# DESIGN.md

## 1. System boundary

The Bayesian layer sits between an outer search loop and an expensive evaluation harness.

### Inputs
- candidate genotypes from the outer loop,
- realized eval results,
- structured human / LLM beliefs,
- rough high-level ideas that must be canonicalized into typed beliefs or remain metadata-only,
- current-era baseline and metadata,
- adapter metadata from the active integration target,
- optional benchmark history from previous eras.

### Outputs
- ranked candidate scores,
- posterior summaries,
- suggested human queries,
- commit / no-commit recommendations,
- rethink summaries,
- previewable compiled priors,
- stable session artifacts for CLI / wrapper tooling.

## 2. High-level architecture

```text
Human / LLM input (JSON or YAML)
    ↓
schema validation
    ↓
deterministic canonicalization
    ↓
optional provider-agnostic model canonicalization
    ↓
typed beliefs + optional session-local surface overlay
    ↓
preview / apply boundary + influence summary
    ↓
preview compiler  ───────────────┐
    ↓                           │
belief compiler                  │
    ↓                           │
prior store ────────────────────┤
                                ↓
outer-loop adapter → feature encoder → objective surrogate
                    ↓                ↓
                    └────────────→ feasibility surrogate
                                     ↓
                        posterior interaction graph
                                     ↓
                       acquisition + batch select
                                     ↓
                  session store + CLI JSON outputs
```

The key layering is:
- **library core** for all real logic,
- **thin CLI** for humans and scripts,
- **thin adapters** for autoresearch / cevolve / future wrappers.

Important constraint:

- free text never directly updates the posterior;
- it must become an inspectable typed belief or a metadata-only proposal first.

## 3. Repository layout

Canonical runtime layout:

```text
autoclanker/
  bayes_layer/
```

The CLI should be rooted at `autoclanker`, not at a separate helper binary.

## 4. Core module layout

Suggested module tree:

```text
autoclanker/
  __init__.py
  cli.py
  py.typed
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
    adapters/
      __init__.py
      protocols.py
      fixture.py
      autoresearch.py
      cevolve.py
```

## 5. Adapter architecture

### 5.1 Why adapters exist

`autoclanker` should be a reusable Bayesian layer, not a hard-coded rewrite of one upstream loop.

The adapter seam is what makes all of these possible at once:
- first-party support for `autoresearch`,
- first-party support for `cevolve`,
- future third-party loops,
- fixture-backed self-contained tests,
- future wrappers or companion tools.

### 5.2 Adapter modes

Support at least these modes:

- `fixture`
- `auto` for first-party adapters
- `local_repo_path`
- `installed_module`
- `subprocess_cli` (optional but useful)

Not every adapter has to support every mode, but the config surface should allow them.
For `autoresearch` and `cevolve`, `auto` is the recommended front door because it
can use a checkout path, importable adapter module, or runnable adapter command
without forcing users into one integration shape.

### 5.3 Adapter protocol responsibilities

The generic adapter should be able to:
- identify itself and report status,
- discover or build a gene / idea registry,
- expose a richer optimization surface with semantic metadata, not just low-level knobs,
- materialize a candidate genotype into an eval-ready form,
- run an evaluation and emit a schema-valid result,
- optionally recommend or apply a commit step,
- expose enough metadata for rethink summaries.

### 5.4 Fixture adapter

The fixture adapter is required so the repo can be implemented and tested with no external dependencies beyond the repo itself.

It should simulate:
- a registry,
- candidate evaluation,
- intended vs realized genotype,
- patch-hash aggregation,
- feasibility failure modes.

### 5.5 First-party upstream adapters

`autoresearch` and `cevolve` adapters should be part of the repo from the outset.

They should:
- load config from a validated adapter config file,
- probe for local paths or importability,
- fail clearly when a requested path is missing,
- allow optional integration tests that skip when upstream references are unavailable.

Current implementation note:

- the fixture adapter is the fully self-contained execution target;
- first-party upstream adapters probe and report local path / module / subprocess availability;
- when `allow_missing` is enabled and the upstream target is absent, those adapters may fall back to the fixture-backed contract shim so the repo stays runnable and testable without external checkouts.

## 6. Session-store design

Default session layout:

```text
.autoclanker/<session_id>/
  session_manifest.yaml
  beliefs.yaml
  compiled_preview.json
  compiled_priors.json
  observations.jsonl
  posterior_summary.json
  query.json
  commit_decision.json
  influence_summary.json
  RESULTS.md
  convergence.png
  candidate_rankings.png
  belief_graph_prior.png
  belief_graph_posterior.png
```

The same store must be relocatable under another parent path, e.g.:

```text
.cevolve/<session_id>/autoclanker/
```

## 7. CLI design

The CLI is the primary user surface in v1.

Recommended command mapping:

```text
beliefs canonicalize-ideas
beliefs expand-ideas
beliefs validate
beliefs preview
beliefs compile
eval validate
adapter registry
adapter surface
adapter validate-config
adapter list
adapter probe
session init
session apply-beliefs
session ingest-eval
session fit
session suggest
session recommend-commit
session status
```

The CLI should be:
- non-interactive,
- JSON-by-default on stdout for machine-readable commands,
- file-friendly,
- script-friendly,
- easy for future wrappers to call.

## 8. User-lane design

### 8.1 Simple lane

A normal user should be able to operate through inline rough ideas or small JSON / YAML files with bounded fields.

### 8.2 Expert lane

A power user should be able to supply direct priors and graph directives, but those must remain optional.

### 8.3 Round-trip preview

Every belief batch should be previewable in optimizer-relevant units before it is actually applied.

### 8.4 Hybrid canonicalization lane

When a canonicalization model is configured, the beginner path becomes:

```text
rough ideas
→ deterministic canonicalization
→ optional model-assisted typed suggestions / overlays
→ preview
→ apply
→ fit / suggest / recommend
```

This lane must stay inspectable and reproducible through session artifacts rather than hidden model state.

### 8.5 Common assistant-authoring workflow

The common advanced workflow should be:

```text
rough ideas
→ adapter registry / adapter surface
→ beliefs canonicalize-ideas
→ beliefs expand-ideas (optional alias)
→ preview
→ assistant-guided escalation for unresolved or high-impact beliefs
→ expert_prior / graph_directive in JSON by default
→ apply
```

This is the right division of labor:
- deterministic canonicalization handles easy cases without a model;
- provider-backed canonicalization resolves rough strategy text into typed candidate beliefs;
- `expand-ideas` is currently an alias when the caller wants the normalized typed payload under that name;
- the assistant skill only escalates the cases that truly need direct prior or graph control.

### 8.6 Validation lanes

The implementation maintains three validation classes:

- `check`: required, self-contained, non-billed;
- `test-upstream-live`: optional real-upstream, non-billed;
- `test-live`: optional billed model-provider validation.

This split is deliberate. Real-model canonicalization is part of the product, but it should not make the core required gate nondeterministic or billing-dependent.

## 9. Wrapper mapping

A future wrapper or companion tool should be able to wrap:

```text
create session      → session init
preview beliefs     → beliefs preview
append eval         → session ingest-eval
refresh posterior   → session fit
get next step       → session suggest
read status         → session status
commit gate         → session recommend-commit
adapter probe       → adapter probe
```

This is why the library / CLI / file contracts matter more than a rich interactive UI in v1.
