---
name: repo-polish
description: "Use when turning a worked-on repository into a clean public-facing product surface: remove internal handoff artifacts, polish README and contributor docs, sweep private/local references, and leave the repo in a shareable state."
---

# Repo Polish

Use this skill near the end of a project when the repo should read like a
finished product instead of an implementation worksite.

## Workflow

1. Audit the public surface first.

- inspect `README.md`
- inspect top-level docs, examples, package metadata, and any shipped skills
- sweep for local absolute paths, private project names, scaffold/template
  references, task artifacts, audit leftovers, and generated state

2. Keep only product-useful docs.

- keep docs that help users or contributors understand the current library
- remove or rewrite transient files such as handoff notes, one-off audit docs,
  scaffold alignment notes, or autonomous-run task artifacts
- prefer a small coherent doc set over an exhaustive historical archive

3. Rewrite the README as the canonical public entry point.

- explain what the project is, why it exists, and how to try it quickly
- separate user quickstart from contributor workflow
- link only to docs that should actually be part of the shared repo surface
- avoid internal project-history narration
- surface the smallest runnable example early, ideally inline
- prefer compact linked tables over wide README grids that force horizontal scroll
- use visuals only when they clarify the product surface: a crisp logo, one clean
  diagram, or one concrete screenshot is usually enough

4. Add or normalize public project docs.

- `LICENSE`
- `CONTRIBUTING.md`
- `STYLE.md` when the repo benefits from a short contributor style guide
- `.env.example` plus a short environment note when the repo has live/auth/env
  toggles that users are likely to need

5. Normalize developer entry points and local env ergonomics.

- keep one primary command surface such as `./bin/dev`
- mirror high-value tasks in `mise` and `make` when those surfaces exist
- make local env handling obvious and safe:
  - ignore `.env`, `.env.local`, `.env.*.local`, and `.envrc`
  - keep only committed templates or examples
  - document how live/provider credentials are supplied without committing them

6. Sweep for embarrassing leftovers.

- machine-local absolute paths
- references to unrelated private repos or projects
- scaffold/template wording in public docs
- stale references to deleted files
- generated directories such as `dist/`, `build/`, or `*.egg-info/` when they
  should not be committed

7. Reconcile tests and acceptance docs.

- update compliance docs or acceptance matrices so they describe the cleaned repo
- remove tests that only enforce transient task-history files
- add focused checks for the public surface when helpful

8. Validate after cleanup.

- run the smallest targeted tests first
- then run the repo’s normal quality gate
- if validation regenerates build artifacts, remove them again before finishing

## Output Standard

The repo should feel like:

- a normal product repository
- easy to understand from `README.md`
- free of internal handoff history
- free of machine-local/private references
- consistent between docs, examples, and tests
