# COMPLIANCE_MATRIX.md

This document mirrors the machine-readable acceptance matrix at [`tests/compliance_matrix.json`](../tests/compliance_matrix.json). The matrix is enforced by targeted behavioral tests plus focused doc-sync checks; it is not intended to replace those tests with substring matching alone. Live entries additionally require running their corresponding live lane so the live behavioral tests execute inside the real lane rather than relying on mark metadata alone.

| ID | Gate | Description |
| --- | --- | --- |
| `M0-001` | `required` | Project surfaces expose autoclanker identity cleanly and consistently. |
| `M0-002` | `required` | CLI exposes the required command families for beliefs, eval, adapter, and session workflows. |
| `M1-001` | `required` | JSON and YAML belief examples validate into typed belief batches. |
| `M1-002` | `required` | Invalid payloads fail with stable machine-readable validation errors. |
| `M1-003` | `required` | Compiled-prior preview is machine-readable and schema-valid. |
| `M1-004` | `required` | Adapter config examples validate successfully. |
| `M1-005` | `required` | Rough ideas canonicalize through deterministic or hybrid pipelines with inspectable provenance, while unresolved text stays metadata-only. |
| `M2-001` | `required` | Evaluation aggregation is patch-hash-aware and deterministic. |
| `M2-002` | `required` | Objective and feasibility surrogates apply runtime prior decay and observation weight. |
| `M2-003` | `required` | Eval-result validation preserves intended and realized genotype order. |
| `M3-001` | `required` | Session artifacts live under a configurable filesystem root. |
| `M3-002` | `required` | Session state resumes from files with manifest-backed status metadata. |
| `M3-003` | `required` | Commit decisions are persisted as structured session artifacts. |
| `M3-004` | `required` | Session ingest rejects eval results from the wrong era. |
| `M3-005` | `required` | Session artifacts persist surface snapshots, overlays, canonicalization summaries, and influence summaries for reproducible hybrid runs. |
| `M3-007` | `required` | session init captures and persists a locked eval contract with digests for the benchmark tree, eval harness, adapter config, and environment inputs. |
| `M3-008` | `required` | Session status exposes current-versus-locked eval-contract drift for hardened sessions. |
| `M3-009` | `required` | session run-eval executes one candidate under isolated execution and records execution metadata. |
| `M3-010` | `required` | session ingest-eval rejects missing or mismatched eval-contract digests for hardened sessions. |
| `M3-011` | `required` | Per-candidate eval run artifacts persist under eval_runs/ with echoed eval-contract metadata. |
| `M3-012` | `required` | session run-frontier executes a frontier batch under the locked eval contract and persists the resulting frontier state. |
| `M3-013` | `required` | session run-eval records contract-scoped lease and soft-stabilization metadata for measured execution when the active eval policy requires it. |
| `M4-001` | `required` | The fixture adapter exercises the full loop without external dependencies. |
| `M4-002` | `required` | Generic python_module adapters can probe and execute without fixture fallback. |
| `M4-003` | `required` | Generic subprocess adapters can probe and execute through JSON-over-stdio. |
| `M4-004` | `required` | Adapter CLI commands work non-interactively for list, validate-config, and probe. |
| `M4-005` | `required` | adapter surface exposes semantic metadata plus higher-level strategy and risk families. |
| `M4-006` | `required` | Frontier inputs preserve candidate lineage metadata, normalized family budget allocations, and heuristic merge suggestions across suggest, status, and persisted session artifacts. |
| `M4-007` | `required` | First-party adapters capture eval contracts and preserve isolated execution metadata when delegating through repo, module, or subprocess shims. |
| `M5-001` | `required` | The first-party autoresearch adapter uses a real contract target when present. |
| `M5-002` | `required` | The first-party cevolve adapter uses a real contract target when present. |
| `M1-LIVE-001` | `live` | A billed model-provider canonicalization lane can resolve rough parser ideas into typed beliefs through the public CLI. |
| `M5-LIVE-001` | `live` | The live autoresearch lane binds a real upstream checkout through the first-party adapter without fixture fallback and surfaces whether metrics came from repo subprocess output or the repo-subprocess heuristic fallback. |
| `M5-LIVE-002` | `live` | The live cevolve lane binds a real upstream checkout through the first-party adapter without fixture fallback and surfaces whether evaluation used the repo benchmark subprocess or the private-session fallback. |
| `M6-001` | `required` | Session workflows enforce a preview-then-apply gate before beliefs become active. |
| `M6-002` | `required` | CLI failures return stable JSON errors and nonzero exit codes. |
| `M6-003` | `required` | session apply-beliefs is available as a non-interactive CLI command. |
| `M6-004` | `required` | Machine-readable CLI commands support --output when users place it after the subcommand as well as in the root-global position. |
| `M6-LIVE-001` | `live` | A billed live canonicalization session measurably changes parser candidate ranking relative to unresolved proposal-only ideas. |
| `M7-LIVE-001` | `live` | A billed model-provider path can promote rough advanced-authoring ideas into expert_prior and graph_directive beliefs through the public canonicalize-ideas CLI. |
| `M7-001` | `required` | A human-readable compliance mirror stays synchronized with the machine-readable matrix. |
| `M7-002` | `required` | Example adapter configs and docs stay runtime-consistent while treating checkout paths as optional rather than mandatory. |
| `M7-003` | `required` | The Bayes showcase example stays executable and proves belief-guided CLI sessions outperform control on the documented toy scenario. |
| `M7-004` | `required` | Human-readable code showcase examples stay documented, runnable, and numerically aligned with their intended optimization stories. |
| `M7-005` | `required` | Beginner live-exercise docs expose minimum required files, bounded belief fields, and low-cruft replay commands. |
| `M7-006` | `required` | Comparative zero-eval cold-start lanes show typed canonical Bayes guidance outperforming control and proposal-only baselines on the parser target. |
| `M7-007` | `required` | The advanced belief authoring skill stays aligned with the CLI and a provider-agnostic workflow for promoting rough ideas into expert_prior and graph_directive authoring. |
| `M7-008` | `required` | Public docs clearly distinguish the non-billed upstream-live lane from the billed model-provider live lane. |
| `M7-009` | `required` | The primary repo docs focus on the current library and contributor workflow rather than internal handoff or scaffold artifacts. |
| `M7-010` | `required` | The advanced skill and belief-input docs describe the common LLM-assisted workflow from rough ideas to advanced typed Bayes specs, with JSON preferred for machine-authored outputs. |
| `M7-011` | `required` | A second deterministic frontier-heavy benchmark shows normalized budget allocations and heuristic merge-suggestion artifacts on the parser target rather than only cold-start ranking lift. |
