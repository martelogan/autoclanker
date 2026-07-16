# clankerprof Rust core

`crates/clankerprof-core` is the Rust port of the public clankerprof
sample-facts core. Its first compatibility target is the versioned JSON schema
documented in `docs/CLANKERPROF_SPEC.md`:

```text
clankerprof.sample_facts.v2
```

The crate intentionally builds every projection from the durable fact layer.
That keeps target, slice, scope/boundary, and comparison views tied to one
decoded stack accounting model instead of separate ad hoc profile walkers.

## Run

```bash
cargo run -p clankerprof-core --bin clankerprof-rs -- \
  facts --profile profile.pb.gz --output sample-facts.json
```

Without `--output`, the command writes compact JSON to stdout; `--pretty`
opts in to indented output.

Generic target attribution:

```bash
cargo run -p clankerprof-core --bin clankerprof-rs -- \
  targets --profile profile.pb.gz --config targets.json
```

Generic slice attribution:

```bash
cargo run -p clankerprof-core --bin clankerprof-rs -- \
  slices --profile profile.pb.gz --slices slices.yml \
  --collapse 'library:*' --unattributed-libraries
```

Single-pass multi-projection (decode once, emit several sections):

```bash
cargo run -p clankerprof-core --bin clankerprof-rs -- \
  report --profile profile.pb.gz --config targets.json \
  --slices slices.yml --scopes-config scopes.toml --include-facts
```

Each report section applies the same validation as the standalone
subcommand it composes: an empty target config errors exactly like
`targets` (`--config or --target is required.`), and option grammars
(`--top`) are validated unconditionally, before any section renders.

Slice comparison:

```bash
cargo run -p clankerprof-core --bin clankerprof-rs -- \
  compare --before before-slices.json --after after-slices.json
```

## Parity

The Python test suite includes `tests/test_clankerprof_rust_parity.py`, which
generates public synthetic pprof fixtures and compares Rust output against
Python `clankerprof` output. The fixture matrix covers raw and gzipped
profiles, inline frames, folded locations, sparse pprof function IDs, generic
target attribution, generic slice attribution, scope decomposition, slice and
boundary comparison, facts export/replay, and the report/compare flag matrix.

Python `clankerprof scopes` remains the reference implementation for
scope/cost-kind/rollup/owner decomposition. The Rust core has byte-level
fixture parity for that projection — including cached predicates, residual
exclusions, owner cost-kind rows, and fact replay — pinned by
`test_clankerprof_rust_scopes_match_python_boundary_decomposition`,
`test_clankerprof_rust_scopes_legacy_aliases_and_boundaries_subcommand`, and
`test_clankerprof_rust_scopes_replay_facts_identically` (see
`docs/CLANKERPROF_PARITY.md` for the full capability matrix).

Cargo-specific validation:

```bash
cargo fmt --check
cargo test
```

## Local real-profile parity

Private profiles should stay outside the repo. To compare local real-profile
goldens against current Python output, and optionally compare the Rust core
against Python where Rust supports the projection, opt into the local-input
safety gate:

```bash
CLANKERPROF_REAL_PROFILE_PARITY=1 \
  scripts/clankerprof/check_real_profile_parity.py \
  --profile /path/to/profile.pb.gz \
  --target-config /path/to/targets.json \
  --scope-config /path/to/scopes.toml \
  --check-rust-core
```

The helper writes only temporary outputs. It reports Python-side `facts`,
`targets`, `boundaries`, and `slices` when caller-provided expected artifacts
match. It reports `rust_facts`, `rust_targets`, and compatible `rust_slices`
when those Rust parity checks pass. The helper does not yet run a Rust scopes
leg against local goldens; Rust scope decomposition parity is covered at the
fixture level by the synthetic parity suite (`docs/CLANKERPROF_PARITY.md`).
