# Idea Inputs

These are the smallest example inputs meant to teach `autoclanker` usage directly.

If you want the absolute shortest starter path:

1. run `autoclanker adapter registry`
2. paste one inline JSON string idea or copy `minimal.json`
3. run `autoclanker beliefs canonicalize-ideas` to see how the high-level idea maps onto your adapter registry

You do not need a file everywhere you go. All belief commands also accept stdin:

```bash
cat ideas.yaml | autoclanker beliefs preview --input - --era-id era_my_app_v1
```

Or pass inline JSON directly:

```bash
autoclanker beliefs preview \
  --era-id era_my_app_v1 \
  --ideas-json '["Compiled regex matching probably helps this parser on repeated log formats."]'
```

Start with one of these:

- `minimal.json`: smallest reusable JSON example
- `bayes_quickstart.json`: beginner Bayes quickstart with a parser-focused relation and risk
- `autoresearch_simple.json`: first-party autoresearch starter ideas in JSON
- `cevolve_synergy.json`: first-party cEvolve starter ideas in JSON
- `minimal.yaml`: same shape in YAML
- `bayes_quickstart.yaml`: the same Bayes quickstart with inline comments
- `autoresearch_simple.yaml`: the same autoresearch starter ideas with inline comments
- `cevolve_synergy.yaml`: the same cEvolve starter ideas with inline comments

The JSON files are meant to be copied or pointed to directly.
The YAML files are the commented teaching versions.
This README contains the exact commands needed to:

- inspect the registry,
- preview the ideas,
- canonicalize them against the registry,
- expand them into the full typed belief schema,
- and run the starter demo.
