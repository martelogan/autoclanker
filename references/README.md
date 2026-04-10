# Optional upstream references

This directory is intentionally optional.

You do **not** need to populate it for the core `autoclanker` implementation.

If you later want to run the first-party `autoresearch` or `cevolve` adapters against real local checkouts, you can place them here:

```text
references/autoresearch/
references/cevolve/
```

Path-based integration tests should skip cleanly when these folders are absent.
