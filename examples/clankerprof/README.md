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
  --before tmp/before-slices.json \
  --after tmp/after-slices.json \
  --threshold-abs 2.0 \
  --threshold-rel 15.0
```

The compare command exits `2` when a focused slice regresses beyond both
thresholds.
