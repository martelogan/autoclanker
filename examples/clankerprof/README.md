# clankerprof examples

These files are copyable config shapes for `clankerprof`. They intentionally do
not include a profile fixture: use any pprof CPU profile exported as raw `.pb` or
gzipped `.pb.gz`.

## Target attribution

Use `targets` when you know the parent boundary you want to explain.

```bash
clankerprof targets \
  --profile profile.pb.gz \
  --config examples/clankerprof/target_config.json \
  --runtime ruby \
  --fold-runtime-internals \
  --track-semantic-callers \
  --semantic-callers-csv tmp/semantic-callers.csv \
  --format json \
  --output tmp/profile-targets.json
```

## Scope decomposition

Use `scopes` when you want one parent denominator split into display rollups,
atomic cost kinds, owner frames, and top caller -> hot leaf evidence.

```bash
clankerprof scopes \
  --profile profile.pb.gz \
  --config examples/clankerprof/scopes.toml \
  --output tmp/profile-scopes.json
```

The config is intentionally generic. Replace the cost-kind and owner selectors
with project-local paths, dependency selectors, expression tables such as
`{ all = [...], not = ... }`, `cost_kind:<label>` selectors for configured
cost-kind labels, `runtime_label:<label>` selectors for runtime-rule labels,
or `slice:<name>` predicates when you want to reuse a slice file as an owner
taxonomy. The older `boundaries` command and `boundaries.toml` vocabulary remain
accepted compatibility aliases.

## Slice attribution

Use `slices` when you want ownership-style views, collapse rules, and JSON that
can be compared before and after a change.

```bash
clankerprof slices \
  --profile profile.pb.gz \
  --config examples/clankerprof/clankerprof-slices.yml \
  --output tmp/profile-slices.json
```

The config references `examples/clankerprof/slices.yml` from the repository
root, which shows generic
slice metadata. Unknown slice fields are preserved in JSON output under
`metadata`, so teams can attach owners, contacts, docs, escalation hints, or
other domain-specific labels without changing `clankerprof`.

## Compare

```bash
clankerprof compare \
  --before tmp/before.json \
  --after tmp/after.json \
  --threshold-abs 2.0 \
  --threshold-rel 15.0
```

The compare command accepts slice or boundary/scope JSON. It exits `2` when a focused
row regresses beyond both thresholds.
