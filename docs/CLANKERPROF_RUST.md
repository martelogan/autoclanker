# clankerprof Rust core

`crates/clankerprof-core` is the Rust port of the public clankerprof
sample-facts core. Its first compatibility target is the versioned JSON schema
documented in `docs/CLANKERPROF_SPEC.md`:

```text
clankerprof.sample_facts.v1
```

The crate intentionally builds every projection from the durable fact layer.
That keeps tree, slice, target, opportunity, and comparison views tied to one
decoded stack accounting model instead of separate ad hoc profile walkers.

## Run

```bash
cargo run -p clankerprof-core --bin clankerprof-rs -- \
  facts --profile profile.pb.gz --output sample-facts.json
```

Without `--output`, the command writes pretty JSON to stdout.

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
target attribution, generic slice attribution, and slice comparison.

Cargo-specific validation:

```bash
cargo fmt --check
cargo test
```

## Local real-profile parity

Private profiles should stay outside the repo. To compare the Rust core against
Python on a local profile, opt into the local-input safety gate:

```bash
CLANKERPROF_REAL_PROFILE_PARITY=1 \
  scripts/clankerprof/check_real_profile_parity.py \
  --profile /path/to/profile.pb.gz \
  --target-config /path/to/targets.json \
  --check-rust-core
```

The helper writes only temporary outputs and reports `rust_facts` plus
`rust_targets` when those parity checks pass. It also reports `rust_slices`
when `--slice-config` uses fields the current Rust CLI can represent directly
without changing semantics.
