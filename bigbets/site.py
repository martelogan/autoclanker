from __future__ import annotations

import html
import json

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from bigbets.core import (
    BigBetsRegistry,
    JsonValue,
    registry_to_input_payload,
    render_artifact_metadata_json,
    render_bigbets,
)
from bigbets.version import (
    BIGBETS_ARTIFACT_SCHEMA_VERSION,
    BIGBETS_REGISTRY_SCHEMA_VERSION,
    BIGBETS_SITE_SCHEMA_VERSION,
    BIGBETS_VERSION,
    generator_metadata,
)


@dataclass(frozen=True, slots=True)
class StaticSiteScaffold:
    output_dir: Path
    app_id: str
    site_schema_version: str
    storage_adapter: str
    files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class StorageAdapter:
    name: str
    description: str
    source: str


_STORAGE_ADAPTERS = {
    "none": StorageAdapter(
        name="none",
        description="Do not write storage-adapter.js; the app falls back to localStorage.",
        source="",
    ),
    "local-storage": StorageAdapter(
        name="local-storage",
        description="Write the documented no-op adapter stub for host-specific replacement.",
        source="built-in",
    ),
}


def list_storage_adapters() -> tuple[StorageAdapter, ...]:
    return tuple(_STORAGE_ADAPTERS.values())


def write_static_site(
    registry: BigBetsRegistry,
    output_dir: Path,
    *,
    app_id: str = "bigbets-portfolio",
    storage_adapter: str = "local-storage",
    storage_adapter_file: Path | None = None,
    overwrite_storage_adapter: bool = False,
) -> StaticSiteScaffold:
    if storage_adapter_file is not None and storage_adapter != "local-storage":
        raise ValueError(
            "--storage-adapter-file cannot be combined with a named adapter."
        )
    if storage_adapter_file is None and storage_adapter not in _STORAGE_ADAPTERS:
        known = ", ".join(sorted(_STORAGE_ADAPTERS))
        raise ValueError(
            f"Unknown storage adapter {storage_adapter!r}; expected one of: {known}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_payload = registry_to_input_payload(registry)
    rendered = render_bigbets(registry)
    files = {
        "index.html": _index_html(registry, app_id),
        "styles.css": _STYLES_CSS,
        "app.js": _APP_JS,
        "registry.seed.json": json.dumps(seed_payload, indent=2, sort_keys=True) + "\n",
        "big_bets.artifact_metadata.json": render_artifact_metadata_json(
            site_schema_version=BIGBETS_SITE_SCHEMA_VERSION
        ),
        "big_bets.registry.json": rendered.registry_json,
        "big_bets.rankings.csv": rendered.rankings_csv,
        "big_bets.md": rendered.markdown,
        "big_bets.mmd": rendered.mermaid,
        "big_bets.excalidraw": rendered.excalidraw,
        "big_bets.svg": rendered.svg,
        "SITE.md": _site_md(output_dir, app_id),
    }
    written: list[Path] = []
    for name, content in files.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        written.append(path)

    font_path = output_dir / "Excalifont-Regular.woff2"
    font_path.write_bytes(
        resources.files("bigbets.assets")
        .joinpath("Excalifont-Regular.woff2")
        .read_bytes()
    )
    written.append(font_path)

    adapter_path = output_dir / "storage-adapter.js"
    adapter_exists_before = adapter_path.exists()
    adapter_written = _write_storage_adapter(
        adapter_path,
        storage_adapter=storage_adapter,
        storage_adapter_file=storage_adapter_file,
        overwrite=overwrite_storage_adapter,
    )
    if adapter_written is not None:
        written.append(adapter_written)

    return StaticSiteScaffold(
        output_dir=output_dir,
        app_id=app_id,
        site_schema_version=BIGBETS_SITE_SCHEMA_VERSION,
        storage_adapter=_storage_adapter_label(
            adapter_path,
            storage_adapter=storage_adapter,
            storage_adapter_file=storage_adapter_file,
            adapter_exists_before=adapter_exists_before,
            adapter_written=adapter_written,
        ),
        files=tuple(written),
    )


def _storage_adapter_label(
    adapter_path: Path,
    *,
    storage_adapter: str,
    storage_adapter_file: Path | None,
    adapter_exists_before: bool,
    adapter_written: Path | None,
) -> str:
    if adapter_written is not None:
        return (
            str(storage_adapter_file)
            if storage_adapter_file is not None
            else storage_adapter
        )
    if adapter_exists_before:
        return f"existing:{adapter_path}"
    return storage_adapter


def site_result_payload(scaffold: StaticSiteScaffold) -> dict[str, JsonValue]:
    return {
        "output_dir": str(scaffold.output_dir),
        "app_id": scaffold.app_id,
        "site_schema_version": scaffold.site_schema_version,
        "storage_adapter": scaffold.storage_adapter,
        "local_preview_command": (
            f"python3 -m http.server --directory {scaffold.output_dir} 8000"
        ),
        "files": [str(path) for path in scaffold.files],
    }


def _write_storage_adapter(
    path: Path,
    *,
    storage_adapter: str,
    storage_adapter_file: Path | None,
    overwrite: bool,
) -> Path | None:
    if storage_adapter_file is not None:
        if not storage_adapter_file.is_file():
            raise ValueError(
                f"Storage adapter file does not exist: {storage_adapter_file}"
            )
        if path.exists() and not overwrite:
            return None
        path.write_text(
            storage_adapter_file.read_text(encoding="utf-8"), encoding="utf-8"
        )
        return path

    if storage_adapter == "none":
        return None

    if path.exists() and not overwrite:
        return None

    path.write_text(_STORAGE_ADAPTER_JS, encoding="utf-8")
    return path


def _index_html(registry: BigBetsRegistry, app_id: str) -> str:
    title = html.escape(registry.title, quote=False)
    description = html.escape(
        registry.description
        or "Ranked big-bet portfolio for weekly planning and agent-loop review.",
        quote=False,
    )
    app_id_json = json.dumps(app_id)
    generator_json = json.dumps(generator_metadata(), sort_keys=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="generator" content="bigbets {BIGBETS_VERSION}">
  <meta name="bigbets-registry-schema-version" content="{BIGBETS_REGISTRY_SCHEMA_VERSION}">
  <meta name="bigbets-artifact-schema-version" content="{BIGBETS_ARTIFACT_SCHEMA_VERSION}">
  <meta name="bigbets-site-schema-version" content="{BIGBETS_SITE_SCHEMA_VERSION}">
  <title>{title}</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="hero">
    <div class="hero-copy">
      <div class="eyebrow">Big Bets Portfolio</div>
      <h1>{title}</h1>
      <p>{description}</p>
      <nav id="portfolio-links" class="portfolio-links" aria-label="Portfolio links"></nav>
    </div>
    <div class="action-strip" aria-label="Board actions">
      <button type="button" id="save-button" title="Persist the current canonical board.">Save board</button>
      <button type="button" id="snapshot-button" title="Keep a dated restorable copy before larger edits.">Snapshot</button>
      <input id="snapshot-name" aria-label="Snapshot name" placeholder="Snapshot name">
      <select id="snapshot-select" aria-label="Plan snapshots">
        <option value="">No snapshots yet</option>
      </select>
      <button type="button" id="restore-snapshot-button">Restore</button>
      <button type="button" id="add-family-button">New lane</button>
      <details class="tool-drawer">
        <summary>More</summary>
        <div>
          <button type="button" id="add-bet-button">New bet</button>
          <button type="button" data-export="excalidraw">Download Excalidraw</button>
          <button type="button" id="write-artifacts-button">Publish current artifacts</button>
        </div>
      </details>
    </div>
    <div class="status-line">
      <span id="storage-status">Loading storage</span>
      <span id="validation-status">Validation pending</span>
      <span id="saved-status">Not saved yet</span>
      <span id="snapshot-status">No snapshot selected</span>
      <span id="artifact-status">Artifacts not published</span>
    </div>
  </header>

  <main>
    <section class="board-stage" id="dependency-board">
      <div class="section-heading">
        <div>
          <span class="label">Dependency board</span>
          <h2>Big bets</h2>
        </div>
        <div class="button-row">
          <button type="button" data-export="svg">Download SVG</button>
          <button type="button" data-export="excalidraw">Download Excalidraw</button>
          <button type="button" data-export="mermaid">Download Mermaid</button>
        </div>
      </div>
      <div id="graph" class="paper-board"></div>
    </section>

    <section class="sheet-stage" id="plan-table-panel">
      <div class="section-heading">
        <div>
          <span class="label">Idea-family sheet</span>
          <h2>Idea lanes</h2>
        </div>
        <div class="button-row">
          <button type="button" id="table-add-family-button">Insert family</button>
          <button type="button" data-export="csv">Download CSV</button>
        </div>
      </div>
      <div class="sheet-wrap">
        <table class="plan-sheet" aria-label="Editable idea-family ranking">
          <thead>
            <tr>
              <th class="w-drag"></th>
              <th class="w-issue">Issue/PR</th>
              <th class="w-slug">Slug</th>
              <th class="w-priority">P</th>
              <th class="w-title">Idea family</th>
              <th class="w-state">Status</th>
              <th class="w-role" title="Lane type: ideas-lane, wip, evidence, proof, follow-up, blocked, or shipped.">Type</th>
              <th class="w-action">Next action</th>
              <th class="w-links">Refs</th>
              <th class="w-artifact" title="Optional lane seed, often an *.ideas.json file.">Seed</th>
              <th class="w-actions">Edit</th>
            </tr>
          </thead>
          <tbody id="plan-table-body"></tbody>
        </table>
      </div>
    </section>

    <details class="json-drawer">
      <summary>Canonical JSON and generated artifacts</summary>
      <div class="json-toolbar">
        <button type="button" id="validate-button">Validate JSON</button>
        <button type="button" id="reset-button">Reset to seed</button>
        <button type="button" data-export="json">Download input JSON</button>
        <button type="button" data-export="normalized-json">Download normalized JSON</button>
        <button type="button" data-export="metadata">Download metadata</button>
        <button type="button" data-export="markdown">Download Markdown</button>
      </div>
      <textarea id="registry-editor" spellcheck="false"></textarea>
      <pre id="validation-log" class="validation-log"></pre>
      <div id="artifact-links" class="artifact-links"></div>
      <p class="adapter-note">
        Persistence uses optional <code>window.bigbetsStorageAdapter</code>
        methods when a host provides them, otherwise browser localStorage.
      </p>
    </details>

    <aside class="inspector-panel" id="inspector-panel" aria-live="polite">
      <div class="section-heading">
        <div>
          <span class="label" id="inspector-kicker">Inspector</span>
          <h2 id="inspector-title">Select a bet or idea family</h2>
          <p id="inspector-hint">Use Edit, double-click, or Enter for details.</p>
        </div>
        <button type="button" id="inspector-close-button">Close</button>
      </div>
      <div class="inspector-toolbar" id="inspector-toolbar"></div>
      <div class="form-grid" id="inspector-fields"></div>
      <div class="button-row inspector-actions">
        <button type="button" id="inspector-apply-button">Apply changes</button>
        <button type="button" id="inspector-duplicate-button">Duplicate</button>
        <button type="button" id="inspector-delete-button">Delete</button>
      </div>
    </aside>
  </main>

  <script>
    window.BIGBETS_APP_ID = {app_id_json};
    window.BIGBETS_REGISTRY_SCHEMA_VERSION = "{BIGBETS_REGISTRY_SCHEMA_VERSION}";
    window.BIGBETS_ARTIFACT_SCHEMA_VERSION = "{BIGBETS_ARTIFACT_SCHEMA_VERSION}";
    window.BIGBETS_SITE_SCHEMA_VERSION = "{BIGBETS_SITE_SCHEMA_VERSION}";
    window.BIGBETS_GENERATOR = {generator_json};
  </script>
  <script src="storage-adapter.js"></script>
  <script src="app.js"></script>
</body>
</html>
"""


def _site_md(output_dir: Path, app_id: str) -> str:
    return f"""# Big Bets Static Site

This directory is generated by `bigbets site scaffold`.

- Generator: `bigbets {BIGBETS_VERSION}`
- Registry schema: `{BIGBETS_REGISTRY_SCHEMA_VERSION}`
- Artifact schema: `{BIGBETS_ARTIFACT_SCHEMA_VERSION}`
- Site schema: `{BIGBETS_SITE_SCHEMA_VERSION}`

## Local Preview

```bash
python3 -m http.server --directory {output_dir} 8000
```

Then open `http://localhost:8000`.

## Update From CLI

```bash
bigbets site scaffold \\
  --input /path/to/bigbets.yaml \\
  --output-dir {output_dir} \\
  --app-id {app_id}
```

The generated app is static-host neutral. It persists through
`window.bigbetsStorageAdapter` when that adapter exists and otherwise uses
browser localStorage. The scaffold preserves an existing `storage-adapter.js`
file so host-specific persistence code can live beside the generated site
without being overwritten by future CLI refreshes.

The generated UI can snapshot dated plan versions. Hosts that want durable
snapshots may implement `listSnapshots`, `saveSnapshot`, and `loadSnapshot` on
`window.bigbetsStorageAdapter`; otherwise snapshots are stored in localStorage.
"""


_STORAGE_ADAPTER_JS = """\
// Optional host adapter.
//
// Replace this file beside the generated site if your static host provides a
// durable database or file store. The generic app will use this adapter when
// window.bigbetsStorageAdapter is defined, and will otherwise fall back to
// browser localStorage.
//
// window.bigbetsStorageAdapter = {
//   name: "Durable host storage",
//   async load({ appId }) {
//     return { registry: null, savedAt: null, sourceLabel: "Durable host" };
//   },
//   async save({ appId, registry }) {
//     return { savedAt: new Date().toISOString(), sourceLabel: "Durable host" };
//   },
//   async listSnapshots({ appId }) {
//     return [];
//   },
//   async saveSnapshot({ appId, registry, name }) {
//     return { id: "snapshot-id", name, createdAt: new Date().toISOString() };
//   },
//   async loadSnapshot({ appId, snapshotId }) {
//     return { registry: null, id: snapshotId, name: null, createdAt: null };
//   },
//   async writeArtifacts({ appId, artifacts }) {
//     return { links: [], message: "Artifacts written." };
//   },
//   async fetchIssue({ url, issue }) {
//     return { url, issue, title: null };
//   },
// };
"""


_STYLES_CSS = """\
@font-face {
  font-family: "Excalifont";
  src: url("Excalifont-Regular.woff2") format("woff2");
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}

:root {
  color-scheme: light;
  --ink: #111827;
  --muted: #5d6c84;
  --paper: #fdfbf4;
  --grid: #ece5d8;
  --line: #111827;
  --line-soft: #d8cbb8;
  --blue: #dbeafe;
  --green: #d8f3dc;
  --yellow: #fff0bf;
  --pink: #f9dadd;
  --lavender: #e9e3ff;
  --white: #fffefa;
  --accent: #0f766e;
  --burnt: #b75b36;
  --danger: #b42318;
  --shadow: 0 12px 28px rgba(30, 30, 30, 0.095);
  --sketch-font: "Excalifont", "Virgil", "Comic Sans MS", "Marker Felt", cursive;
  --ui-font: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
}

* { box-sizing: border-box; }

html { scroll-behavior: smooth; }

body {
  margin: 0;
  color: var(--ink);
  background:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px),
    radial-gradient(circle at 16% 10%, rgba(219, 234, 254, 0.42), transparent 30rem),
    radial-gradient(circle at 84% 12%, rgba(216, 243, 220, 0.4), transparent 29rem),
    var(--paper);
  background-size: 56px 56px, 56px 56px, auto, auto, auto;
  font: 15px/1.5 var(--ui-font);
}

button, input, select, textarea { font: inherit; }

button {
  border: 1.2px solid var(--line);
  background: var(--white);
  color: var(--ink);
  border-radius: 14px 17px 13px 18px;
  padding: 0.46rem 0.68rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 1.5px 2.5px 0 rgba(17, 24, 39, 0.08);
}

button:hover {
  transform: translate(-1px, -1px);
  box-shadow: 4px 5px 0 rgba(17, 24, 39, 0.11);
}

button.linkish {
  border: 1px solid var(--line-soft);
  box-shadow: none;
  padding: 0.28rem 0.45rem;
  font-size: 0.78rem;
}

input, select, textarea {
  border: 2px solid var(--line-soft);
  border-radius: 12px;
  background: var(--white);
  color: var(--ink);
}

input:focus, select:focus, textarea:focus, [contenteditable="true"]:focus {
  outline: 3px solid rgba(15, 118, 110, 0.18);
  border-color: var(--accent);
}

a { color: #0b6795; text-decoration-thickness: 0.09em; }

code {
  border: 1px solid var(--line-soft);
  border-radius: 0.35rem;
  background: #fff7dc;
  padding: 0.08rem 0.25rem;
}

.hero, main {
  max-width: 1320px;
  margin: 0 auto;
}

.hero {
  padding: 52px 30px 18px;
}

.hero h1 {
  max-width: 1040px;
  margin: 10px 0 14px;
  font-family: var(--sketch-font);
  font-size: clamp(2.6rem, 6vw, 5.8rem);
  font-weight: 400;
  line-height: 0.92;
  letter-spacing: -0.025em;
}

.hero p {
  max-width: 820px;
  color: var(--muted);
  font-size: 1.08rem;
}

.portfolio-links {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}

.portfolio-links:empty { display: none; }

.portfolio-links a {
  border: 1.1px solid var(--line);
  border-radius: 999px;
  background: rgba(255, 254, 250, 0.7);
  padding: 0.22rem 0.58rem;
  color: var(--ink);
  text-decoration: none;
  box-shadow: 1.4px 2px 0 rgba(17, 24, 39, 0.05);
}

.eyebrow, .label {
  color: var(--burnt);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.action-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 9px;
  align-items: center;
  margin-top: 20px;
  padding: 10px;
  border: 1.2px solid var(--line);
  border-radius: 24px 28px 22px 26px;
  background: rgba(255, 254, 250, 0.62);
  box-shadow: 2px 3px 0 rgba(17, 24, 39, 0.05);
}

.tool-drawer {
  position: relative;
}

.tool-drawer summary {
  list-style: none;
  cursor: pointer;
  border: 1.2px solid var(--line);
  border-radius: 14px 17px 13px 18px;
  background: var(--white);
  padding: 0.46rem 0.68rem;
  box-shadow: 1.5px 2.5px 0 rgba(17, 24, 39, 0.08);
  font-weight: 600;
}

.tool-drawer summary::-webkit-details-marker { display: none; }

.tool-drawer[open] div {
  position: absolute;
  right: 0;
  z-index: 10;
  display: grid;
  gap: 7px;
  min-width: 180px;
  margin-top: 8px;
  padding: 10px;
  border: 1.5px solid var(--line);
  border-radius: 18px;
  background: var(--white);
  box-shadow: var(--shadow);
}

.action-strip input {
  width: 150px;
  padding: 0.5rem 0.65rem;
}

.action-strip select {
  max-width: 230px;
  padding: 0.5rem 0.65rem;
}

.status-line {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
  color: var(--muted);
  font-size: 0.86rem;
}

.status-line span {
  display: inline-flex;
  align-items: center;
  min-height: 1.55rem;
  border: 1px solid var(--line-soft);
  border-radius: 999px;
  background: rgba(255, 254, 250, 0.74);
  padding: 0.08rem 0.58rem;
}

main {
  padding: 0 30px 56px;
}

.board-stage, .sheet-stage, .json-drawer, .inspector-panel {
  border: 1.4px solid var(--line);
  border-radius: 28px 34px 25px 31px;
  background: rgba(255, 250, 240, 0.82);
  box-shadow: var(--shadow);
}

.board-stage, .sheet-stage, .json-drawer {
  margin-top: 22px;
  padding: 20px;
}

.section-heading {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
  margin-bottom: 14px;
}

.section-heading h2 {
  margin: 2px 0 0;
  max-width: 820px;
  font-family: var(--sketch-font);
  font-size: clamp(1.35rem, 2.4vw, 2.05rem);
  font-weight: 400;
  line-height: 1.05;
  letter-spacing: -0.015em;
}

.section-heading p,
#inspector-hint,
.adapter-note {
  color: var(--muted);
  margin-bottom: 0;
}

.button-row, .json-toolbar, .inspector-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.paper-board {
  overflow: auto;
  min-height: 420px;
  border: 1.2px solid var(--line);
  border-radius: 24px 29px 22px 27px;
  background:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px),
    var(--paper);
  background-size: 56px 56px;
  box-shadow: inset 0 0 0 8px rgba(255, 254, 250, 0.44);
}

.paper-board svg {
  display: block;
  min-width: 1040px;
}

.paper-board [data-bet-id] {
  cursor: default;
}

.paper-board .card-drag-handle {
  cursor: grab;
}

.paper-board .card-drag-handle:active {
  cursor: grabbing;
}

.paper-board .card-action-hint {
  opacity: 0;
  transition: opacity 120ms ease;
  pointer-events: none;
}

.paper-board [data-bet-id]:hover .card-action-hint,
.paper-board .selected-card .card-action-hint {
  opacity: 1;
}

.paper-board [data-bet-id]:hover .card-outline {
  stroke-width: 2;
}

.paper-board .selected-card .card-outline {
  stroke: var(--burnt);
  stroke-width: 2.3;
}

.paper-board .dragging-card {
  opacity: 0.74;
}

.board-drop-preview {
  fill: rgba(183, 91, 54, 0.08);
  stroke: rgba(183, 91, 54, 0.7);
  stroke-width: 1.4;
  stroke-dasharray: 8 7;
  pointer-events: none;
  transition: x 120ms ease, y 120ms ease;
}

.sheet-stage {
  background: rgba(255, 254, 250, 0.86);
}

.sheet-wrap {
  overflow: auto;
  border: 1.2px solid var(--line);
  border-radius: 18px;
  background: var(--white);
}

.plan-sheet {
  width: 100%;
  min-width: 1120px;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 0.79rem;
}

.plan-sheet th {
  position: sticky;
  top: 0;
  z-index: 2;
  background: #fff7dc;
  border-bottom: 2px solid var(--line);
  color: var(--muted);
  font-size: 0.66rem;
  letter-spacing: 0.05em;
  padding: 0.35rem 0.42rem;
  text-align: left;
  text-transform: uppercase;
}

.plan-sheet td {
  border: 1px solid var(--line-soft);
  padding: 0.18rem 0.28rem;
  vertical-align: top;
}

.plan-sheet tr.family-row:nth-child(odd) td {
  background: #fffefa;
}

.plan-sheet tr.family-row:nth-child(even) td {
  background: #fff9ea;
}

.plan-sheet tr.selected-row td {
  background: #effdf3;
  box-shadow: inset 0 0 0 2px rgba(15, 118, 110, 0.38);
}

.plan-sheet tr.drop-target td {
  box-shadow: inset 0 0 0 3px rgba(183, 91, 54, 0.34);
}

.bet-group-row td {
  background: #eaf7ec;
  border-top: 1.2px solid var(--line);
  border-bottom: 1.2px solid var(--line);
  font-weight: 600;
}

.bet-group-row[data-priority="P1"] td { background: #eaf2ff; }
.bet-group-row[data-priority="P2"] td { background: #fff3c9; }
.bet-group-row[data-priority="P3"] td { background: #eee9ff; }

.bet-group {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.bet-group-main {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: baseline;
}

.bet-title {
  font-size: 1rem;
  letter-spacing: -0.03em;
}

.mini-meta {
  color: var(--muted);
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.sheet-cell {
  min-height: 1.35rem;
  border: 1px solid transparent;
  border-radius: 7px;
  padding: 0.06rem 0.14rem;
  white-space: normal;
  word-break: break-word;
}

.sheet-select {
  width: 100%;
  min-height: 1.55rem;
  border: 1px solid transparent;
  border-radius: 7px;
  background: transparent;
  color: var(--ink);
  padding: 0.03rem 1.2rem 0.03rem 0.14rem;
}

.sheet-select:focus {
  background: rgba(15, 118, 110, 0.07);
}

.sheet-cell.active-cell {
  border-color: var(--accent);
  background: rgba(15, 118, 110, 0.07);
}

.sheet-link {
  display: inline-flex;
  max-width: 100%;
  align-items: center;
  gap: 0.25rem;
  color: #0b6795;
  text-decoration: underline;
  text-decoration-thickness: 0.08em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.aux-link {
  margin-left: 0.35rem;
}

.refs-list {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.refs-list .sheet-link {
  max-width: 7.8rem;
}

.muted-value {
  color: #a59b8e;
  font-style: italic;
}

.sheet-cell:empty::before {
  content: attr(data-placeholder);
  color: #a59b8e;
}

.drag-handle {
  width: 1.55rem;
  min-width: 1.55rem;
  padding: 0.14rem 0;
  border-width: 1px;
  border-radius: 8px;
  box-shadow: none;
  cursor: grab;
}

.row-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.row-actions button {
  padding: 0.2rem 0.36rem;
  border-width: 1px;
  border-radius: 8px;
  box-shadow: none;
  font-size: 0.72rem;
}

.w-drag { width: 2.2rem; }
.w-issue { width: 4.1rem; }
.w-slug { width: 6.5rem; }
.w-priority { width: 3.3rem; }
.w-title { width: 16rem; }
.w-state { width: 5.4rem; }
.w-role { width: 6.8rem; }
.w-action { width: 15rem; }
.w-links { width: 8.8rem; }
.w-artifact { width: 5.8rem; }
.w-actions { width: 6.6rem; }

.json-drawer summary {
  cursor: pointer;
  font-size: 1.05rem;
  font-weight: 650;
}

.json-toolbar {
  margin: 14px 0;
}

#registry-editor {
  width: 100%;
  min-height: 420px;
  padding: 14px;
  background: #171d1a;
  color: #edf7ef;
  font: 13px/1.55 "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
}

.validation-log {
  min-height: 2.6rem;
  white-space: pre-wrap;
  color: var(--accent);
  font: 13px/1.45 "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
}

.validation-log.error { color: var(--danger); }

.artifact-links {
  display: grid;
  gap: 6px;
  color: var(--muted);
}

.inspector-panel {
  display: none;
  position: sticky;
  bottom: 18px;
  z-index: 5;
  margin-top: 20px;
  padding: 20px;
  background: #fffefa;
}

.inspector-panel.active {
  display: block;
}

.inspector-toolbar {
  margin: 8px 0 14px;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 12px;
}

label {
  display: grid;
  gap: 5px;
  color: var(--muted);
  font-weight: 650;
}

label.wide { grid-column: 1 / -1; }

label input, label select, label textarea {
  width: 100%;
  padding: 0.65rem 0.72rem;
}

textarea { min-height: 84px; resize: vertical; }

@media (max-width: 760px) {
  .hero, main { padding-left: 16px; padding-right: 16px; }
  .section-heading { display: block; }
  .button-row, .json-toolbar { margin-top: 12px; }
  .hero h1 { font-size: clamp(2.5rem, 16vw, 4.4rem); }
}
"""


_APP_JS = """\
const STATUS_VALUES = new Set([
  "active",
  "candidate",
  "parked",
  "blocked",
  "shipped",
  "rejected",
  "superseded",
  "closed",
]);
const ROLE_VALUES = new Set([
  "ideas-lane",
  "wip",
  "evidence",
  "proof",
  "follow-up",
  "blocked",
  "shipped",
]);
const LINK_KIND_VALUES = new Set([
  "artifact",
  "board",
  "doc",
  "evidence",
  "issue",
  "project",
  "pr",
  "tracker",
]);
const UNLOCK_STATE_VALUES = new Set(["locked", "emerging", "unlocked"]);
const IDENTIFIER_RE = /^[a-z][a-z0-9_-]*$/;
const PRIORITY_RE = /^P([0-9]+)$/;
const APP_ID = window.BIGBETS_APP_ID || "bigbets-portfolio";
const REGISTRY_SCHEMA_VERSION = window.BIGBETS_REGISTRY_SCHEMA_VERSION || "bigbets.registry.v1";
const ARTIFACT_SCHEMA_VERSION = window.BIGBETS_ARTIFACT_SCHEMA_VERSION || "bigbets.artifacts.v2";
const SITE_SCHEMA_VERSION = window.BIGBETS_SITE_SCHEMA_VERSION || "bigbets.site.v2";
const GENERATOR = window.BIGBETS_GENERATOR || { name: "bigbets", version: "unknown" };

const BOARD = {
  cardWidth: 350,
  cardHeight: 156,
  cardGap: 46,
  waveGap: 116,
  marginX: 78,
  marginY: 132,
};

const localStorageAdapter = {
  name: "Browser localStorage",
  async load({ appId }) {
    const text = window.localStorage.getItem(storageKey(appId));
    if (!text) return null;
    return { ...JSON.parse(text), sourceLabel: this.name };
  },
  async save({ appId, registry }) {
    const savedAt = new Date().toISOString();
    window.localStorage.setItem(storageKey(appId), JSON.stringify({ registry, savedAt }));
    return { savedAt, sourceLabel: this.name };
  },
  async listSnapshots({ appId }) {
    return readLocalSnapshots(appId).map(({ registry, ...summary }) => summary);
  },
  async saveSnapshot({ appId, registry, name }) {
    const createdAt = new Date().toISOString();
    const id = `snapshot-${createdAt.replace(/[^0-9TZ]/g, "")}`;
    const snapshot = { id, name: name || `Plan ${createdAt.slice(0, 10)}`, createdAt, registry };
    const snapshots = readLocalSnapshots(appId).filter((item) => item.id !== id);
    snapshots.unshift(snapshot);
    window.localStorage.setItem(snapshotKey(appId), JSON.stringify(snapshots));
    return { id, name: snapshot.name, createdAt };
  },
  async loadSnapshot({ appId, snapshotId }) {
    const snapshot = readLocalSnapshots(appId).find((item) => item.id === snapshotId);
    if (!snapshot) throw new Error(`Unknown snapshot: ${snapshotId}`);
    return snapshot;
  },
  async writeArtifacts({ appId, artifacts }) {
    const savedAt = new Date().toISOString();
    window.localStorage.setItem(`${storageKey(appId)}:artifacts`, JSON.stringify({ artifacts, savedAt }));
    return { links: [], message: "Artifacts saved to browser localStorage. Use the download buttons for portable files." };
  },
};

const adapter = window.bigbetsStorageAdapter || localStorageAdapter;
const state = {
  seed: null,
  registry: null,
  rendered: null,
  selection: null,
  inspectorOpen: false,
  snapshots: [],
  dragRow: null,
  activeCell: null,
  suppressBoardClick: false,
};
const $ = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", init);

async function init() {
  wireButtons();
  state.seed = await fetch("registry.seed.json").then((response) => response.json());
  const loaded = await callAdapter("load", { appId: APP_ID }, null);
  const registry = loaded?.registry || state.seed;
  $("storage-status").textContent = loaded?.sourceLabel || adapter.name || "Seed file";
  $("saved-status").textContent = formatTimestamp(loaded?.savedAt || loaded?.saved_at);
  const artifactButton = $("write-artifacts-button");
  if (artifactButton && (adapter === localStorageAdapter || /localStorage/i.test(adapter.name || ""))) {
    artifactButton.textContent = "Cache artifacts locally";
    artifactButton.title = "Stores generated artifacts in this browser. Use download buttons for portable files.";
  }
  setRegistry(registry, "Loaded registry.");
  await refreshSnapshots();
}

function wireButtons() {
  on("validate-button", "click", () => loadFromJsonEditor("Validated JSON."));
  on("save-button", "click", saveRegistry);
  on("reset-button", "click", () => {
    state.selection = null;
    setRegistry(state.seed, "Reset to seed registry.");
  });
  on("write-artifacts-button", "click", writeArtifacts);
  on("snapshot-button", "click", saveSnapshot);
  on("restore-snapshot-button", "click", restoreSnapshot);
  on("add-bet-button", "click", addBet);
  on("add-family-button", "click", () => addFamily());
  on("table-add-family-button", "click", () => addFamily());
  on("inspector-close-button", "click", clearSelection);
  on("inspector-apply-button", "click", applyInspector);
  on("inspector-delete-button", "click", deleteSelection);
  on("inspector-duplicate-button", "click", duplicateSelection);

  document.querySelectorAll("[data-export]").forEach((button) => {
    button.addEventListener("click", () => exportArtifact(button.dataset.export));
  });
}

function on(id, eventName, handler) {
  const element = $(id);
  if (element) element.addEventListener(eventName, handler);
}

async function callAdapter(method, payload, fallbackValue) {
  const receiver = typeof adapter[method] === "function" ? adapter : localStorageAdapter;
  if (typeof receiver[method] !== "function") return fallbackValue;
  try {
    return await receiver[method](payload);
  } catch (error) {
    console.warn(`Storage adapter ${method} failed`, error);
    if (receiver === localStorageAdapter) throw error;
    return await localStorageAdapter[method](payload);
  }
}

function setRegistry(payload, message) {
  const { registry } = validateRegistry(payload);
  state.registry = registry;
  $("registry-editor").value = JSON.stringify(registry, null, 2);
  renderAll(message || "Valid registry.");
}

function loadFromJsonEditor(message) {
  try {
    setRegistry(JSON.parse($("registry-editor").value), message);
    return true;
  } catch (error) {
    markInvalid(error.message);
    return false;
  }
}

async function saveRegistry() {
  if (!loadFromJsonEditor("Validated before save.")) return;
  const result = await callAdapter("save", { appId: APP_ID, registry: state.registry }, null);
  $("storage-status").textContent = result?.sourceLabel || adapter.name || "Saved";
  $("saved-status").textContent = formatTimestamp(result?.savedAt || result?.saved_at || new Date().toISOString());
  setLog("Saved canonical registry.", false);
}

async function saveSnapshot() {
  if (!loadFromJsonEditor("Validated before snapshot.")) return;
  const name = $("snapshot-name").value.trim();
  const result = await callAdapter("saveSnapshot", { appId: APP_ID, registry: state.registry, name }, null);
  $("snapshot-status").textContent = result?.name ? `Snapshot: ${result.name}` : "Snapshot saved";
  $("snapshot-name").value = "";
  await refreshSnapshots(result?.id);
  setLog("Saved dated plan snapshot.", false);
}

async function restoreSnapshot() {
  const snapshotId = $("snapshot-select").value;
  if (!snapshotId) return;
  const snapshot = await callAdapter("loadSnapshot", { appId: APP_ID, snapshotId }, null);
  if (!snapshot?.registry) {
    markInvalid("Selected snapshot did not include a registry.");
    return;
  }
  state.selection = null;
  setRegistry(snapshot.registry, `Restored snapshot ${snapshot.name || snapshot.id}.`);
  $("snapshot-status").textContent = `Restored: ${snapshot.name || snapshot.id}`;
}

async function refreshSnapshots(selectedId = null) {
  const snapshots = await callAdapter("listSnapshots", { appId: APP_ID }, []);
  state.snapshots = Array.isArray(snapshots) ? snapshots : [];
  const options = state.snapshots
    .map((snapshot) => `<option value="${escapeAttr(snapshot.id)}">${escapeHtml(snapshot.name || snapshot.id)} - ${escapeHtml(formatDate(snapshot.createdAt || snapshot.created_at))}</option>`)
    .join("");
  $("snapshot-select").innerHTML = options || '<option value="">No snapshots yet</option>';
  if (selectedId) $("snapshot-select").value = selectedId;
  $("snapshot-status").textContent = state.snapshots.length ? `${state.snapshots.length} snapshots` : "No snapshots yet";
}

function renderAll(message) {
  try {
    state.rendered = renderArtifacts(state.registry);
    $("validation-status").textContent = "Valid";
    setLog(message || "Valid registry.", false);
    $("graph").innerHTML = state.rendered.svg;
    renderPortfolioLinks();
    wireGraphCards();
    renderPlanTable();
    renderInspector();
  } catch (error) {
    markInvalid(error.message);
  }
}

function renderPortfolioLinks() {
  const target = $("portfolio-links");
  if (!target) return;
  const links = state.registry?.metadata?.links || [];
  target.innerHTML = links
    .map((link) => `<a href="${escapeAttr(link.url)}" target="_blank" rel="noopener">${escapeHtml(link.label)}</a>`)
    .join("");
}

function markInvalid(message) {
  $("validation-status").textContent = "Invalid";
  setLog(message, true);
}

function selectBet(id, openInspector = false) {
  state.selection = { kind: "bet", id };
  state.inspectorOpen = openInspector;
  renderInspector();
  highlightSelection();
}

function selectFamily(issue, openInspector = false) {
  state.selection = { kind: "family", issue: Number(issue) };
  state.inspectorOpen = openInspector;
  renderInspector();
  highlightSelection();
}

function clearSelection() {
  state.selection = null;
  state.inspectorOpen = false;
  renderInspector();
  highlightSelection();
}

function openSelection() {
  if (!state.selection) return;
  state.inspectorOpen = true;
  renderInspector();
}

function wireGraphCards() {
  const svg = $("graph").querySelector("svg");
  if (!svg) return;
  svg.querySelectorAll("[data-bet-id]").forEach((node) => {
    node.addEventListener("click", () => {
      if (state.suppressBoardClick) {
        state.suppressBoardClick = false;
        return;
      }
      selectBet(node.dataset.betId, false);
    });
    node.addEventListener("dblclick", () => selectBet(node.dataset.betId, true));
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectBet(node.dataset.betId, true);
      }
    });
    const handle = node.querySelector("[data-drag-handle]");
    if (handle) {
      handle.addEventListener("pointerdown", (event) => startBoardDrag(event, svg, node));
    }
  });
}

function startBoardDrag(event, svg, node) {
  if (event.button !== 0) return;
  event.stopPropagation();
  event.preventDefault();
  const start = svgPoint(svg, event);
  const origin = {
    x: Number(node.dataset.x || 0),
    y: Number(node.dataset.y || 0),
  };
  let moved = false;
  let currentSlot = boardSlotFromPosition(origin.x, origin.y);
  node.setPointerCapture(event.pointerId);
  node.style.transition = "none";
  node.classList.add("dragging-card");
  const move = (moveEvent) => {
    const point = svgPoint(svg, moveEvent);
    const dx = point.x - start.x;
    const dy = point.y - start.y;
    if (Math.max(Math.abs(dx), Math.abs(dy)) > 3) moved = true;
    node.setAttribute("transform", `translate(${dx} ${dy})`);
    currentSlot = boardSlotFromPosition(origin.x + dx, origin.y + dy);
    updateBoardDropPreview(svg, currentSlot);
  };
  const up = (upEvent) => {
    node.releasePointerCapture(event.pointerId);
    node.removeEventListener("pointermove", move);
    node.removeEventListener("pointerup", up);
    node.removeAttribute("transform");
    node.style.transition = "";
    node.classList.remove("dragging-card");
    clearBoardDropPreview(svg);
    if (!moved) {
      selectBet(node.dataset.betId, false);
      return;
    }
    state.suppressBoardClick = true;
    const point = svgPoint(svg, upEvent);
    currentSlot = boardSlotFromPosition(origin.x + point.x - start.x, origin.y + point.y - start.y);
    moveBetToWaveRank(node.dataset.betId, currentSlot.wave, currentSlot.rank);
    state.selection = { kind: "bet", id: node.dataset.betId };
    state.inspectorOpen = false;
    commitRegistry("Moved bet on dependency board.");
  };
  node.addEventListener("pointermove", move);
  node.addEventListener("pointerup", up);
}

function svgPoint(svg, event) {
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(svg.getScreenCTM().inverse());
}

function boardSlotFromPosition(x, y) {
  return {
    wave: Math.max(1, Math.round((y - BOARD.marginY) / (BOARD.cardHeight + BOARD.waveGap)) + 1),
    rank: Math.max(1, Math.round((x - BOARD.marginX) / (BOARD.cardWidth + BOARD.cardGap)) + 1),
  };
}

function boardSlotPosition(slot) {
  return {
    x: BOARD.marginX + (slot.rank - 1) * (BOARD.cardWidth + BOARD.cardGap),
    y: BOARD.marginY + (slot.wave - 1) * (BOARD.cardHeight + BOARD.waveGap),
  };
}

function updateBoardDropPreview(svg, slot) {
  const position = boardSlotPosition(slot);
  let preview = svg.querySelector("#board-drop-preview");
  if (!preview) {
    preview = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    preview.id = "board-drop-preview";
    preview.setAttribute("class", "board-drop-preview");
    preview.setAttribute("rx", "27");
    svg.appendChild(preview);
  }
  preview.setAttribute("x", position.x);
  preview.setAttribute("y", position.y);
  preview.setAttribute("width", BOARD.cardWidth);
  preview.setAttribute("height", BOARD.cardHeight);
}

function clearBoardDropPreview(svg) {
  svg.querySelector("#board-drop-preview")?.remove();
}

function moveBetToWaveRank(id, wave, rank) {
  const bet = findBet(id);
  bet.wave = wave;
  bet.priority = priorityForWave(wave);
  const waveBets = state.registry.big_bets
    .filter((item) => item.wave === wave && item.id !== id)
    .sort(bigBetSort);
  waveBets.splice(Math.max(0, rank - 1), 0, bet);
  waveBets.forEach((item, index) => {
    item.rank = index + 1;
  });
  normalizeBetRanks();
}

function renderPlanTable() {
  const rows = [];
  const familiesByBet = familiesByBigBet(state.registry);
  state.registry.big_bets.sort(bigBetSort).forEach((bet) => {
    const families = familiesByBet.get(bet.id) || [];
    rows.push(betGroupRow(bet, families.length));
    families.forEach((family) => rows.push(familyRow(family)));
  });
  $("plan-table-body").innerHTML = rows.join("");
  wirePlanTable();
  highlightSelection();
}

function betGroupRow(bet, familyCount) {
  return `
    <tr class="bet-group-row" data-kind="bet" data-id="${escapeAttr(bet.id)}" data-priority="${escapeAttr(bet.priority)}">
      <td><button type="button" class="drag-handle" draggable="true" data-drag-kind="bet" data-drag-id="${escapeAttr(bet.id)}" title="Drag bet">::</button></td>
      <td colspan="10">
        <div class="bet-group">
          <div class="bet-group-main">
            <span class="mini-meta">${escapeHtml(bet.priority)} / ${escapeHtml(bet.status)} / ${escapeHtml(bet.unlock_state)}</span>
            <span class="bet-title">${escapeHtml(bet.title)}</span>
            <span class="mini-meta">${familyCount} idea families</span>
          </div>
          <div class="row-actions">
            <button type="button" data-row-action="focus">Edit bet</button>
            <button type="button" data-row-action="child">Add family</button>
          </div>
        </div>
      </td>
    </tr>
  `;
}

function familyRow(family) {
  return `
    <tr class="family-row" data-kind="family" data-issue="${family.issue}" data-big-bet="${escapeAttr(family.big_bet)}">
      <td><button type="button" class="drag-handle" draggable="true" data-drag-kind="family" data-drag-id="${family.issue}" title="Drag family">::</button></td>
      ${issueCell(family)}
      ${cell("slug", family.slug || "", "w-slug", "short_slug")}
      ${selectCell("priority", family.priority, "w-priority", ["P0", "P1", "P2", "P3"])}
      ${cell("title", family.title, "w-title", "Idea family title")}
      ${selectCell("status", family.status, "w-state", [...STATUS_VALUES])}
      ${selectCell("role", family.role || "", "w-role", ["", ...ROLE_VALUES])}
      ${cell("next_action", family.next_action || "", "w-action", "next action")}
      ${linksCell(family)}
      ${linkCell("artifact", family.artifact || "", "w-artifact", "none")}
      <td class="row-actions w-actions">
        <button type="button" data-row-action="focus">Edit</button>
        <button type="button" data-row-action="hydrate">Autofill</button>
        <button type="button" data-row-action="delete">Remove</button>
      </td>
    </tr>
  `;
}

function cell(field, value, className, placeholder) {
  return `<td class="${className}"><div class="sheet-cell" contenteditable="true" data-cell data-field="${escapeAttr(field)}" data-placeholder="${escapeAttr(placeholder)}">${escapeHtml(value)}</div></td>`;
}

function selectCell(field, value, className, options) {
  const optionHtml = options
    .map((option) => `<option value="${escapeAttr(option)}"${String(option) === String(value || "") ? " selected" : ""}>${escapeHtml(option || "-")}</option>`)
    .join("");
  return `<td class="${className}"><select class="sheet-select" data-cell data-field="${escapeAttr(field)}">${optionHtml}</select></td>`;
}

function issueCell(family) {
  const label = `#${family.issue}`;
  const issue = family.url
    ? `<a class="sheet-link" href="${escapeAttr(family.url)}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`
    : escapeHtml(label);
  return `<td class="w-issue"><div class="sheet-cell" tabindex="0" data-cell data-field="issue" data-value="${family.issue}" data-placeholder="Issue">${issue}</div></td>`;
}

function linksCell(family) {
  const links = (family.links || []).filter((link) => link.url);
  const content = links.length
    ? `<span class="refs-list">${links.map((link) => `<a class="sheet-link" href="${escapeAttr(link.url)}" target="_blank" rel="noopener">${escapeHtml(shortLinkLabel(link))}</a>`).join("")}</span>`
    : `<span class="muted-value">none</span>`;
  return `<td class="w-links"><div class="sheet-cell" tabindex="0" data-cell data-field="links" data-value="${escapeAttr(linksText(family.links || []))}" data-placeholder="refs">${content}</div></td>`;
}

function linkCell(field, value, className, label) {
  const text = value ? linkLabel(value, label) : "";
  const content = value
    ? `<a class="sheet-link" href="${escapeAttr(value)}" target="_blank" rel="noopener">${escapeHtml(text)}</a>`
    : `<span class="muted-value">${escapeHtml(label)}</span>`;
  return `<td class="${className}"><div class="sheet-cell" tabindex="0" data-cell data-field="${escapeAttr(field)}" data-value="${escapeAttr(value)}" data-placeholder="${escapeAttr(label)}">${content}</div></td>`;
}

function wirePlanTable() {
  $("plan-table-body").querySelectorAll("tr").forEach((row) => {
    row.addEventListener("dragover", (event) => {
      if (!state.dragRow) return;
      event.preventDefault();
      row.classList.add("drop-target");
    });
    row.addEventListener("dragleave", () => row.classList.remove("drop-target"));
    row.addEventListener("drop", (event) => {
      event.preventDefault();
      handleSheetDrop(row);
    });
    row.querySelectorAll("[data-row-action]").forEach((button) => {
      button.addEventListener("click", () => handleRowAction(row, button.dataset.rowAction));
    });
  });
  $("plan-table-body").querySelectorAll("[data-cell]").forEach((cellNode) => {
    cellNode.addEventListener("blur", () => {
      cellNode.classList.remove("active-cell");
      if (cellNode.isContentEditable) applyTableRow(cellNode.closest("tr"));
    });
    cellNode.addEventListener("change", () => {
      applyTableRow(cellNode.closest("tr"));
    });
    cellNode.addEventListener("keydown", handleCellKeydown);
    cellNode.addEventListener("focus", () => {
      document.querySelectorAll(".active-cell").forEach((node) => node.classList.remove("active-cell"));
      cellNode.classList.add("active-cell");
      selectFamily(cellNode.closest("tr").dataset.issue, false);
    });
  });
  $("plan-table-body").querySelectorAll("[data-drag-kind]").forEach((handle) => {
    handle.addEventListener("dragstart", (event) => {
      state.dragRow = { kind: handle.dataset.dragKind, id: handle.dataset.dragId };
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", JSON.stringify(state.dragRow));
    });
    handle.addEventListener("dragend", () => {
      state.dragRow = null;
      document.querySelectorAll(".drop-target").forEach((node) => node.classList.remove("drop-target"));
    });
  });
}

function handleCellKeydown(event) {
  if (
    event.currentTarget.tagName === "SELECT" &&
    ["ArrowLeft", "ArrowRight", "ArrowDown", "ArrowUp", "Enter"].includes(event.key) &&
    !event.metaKey &&
    !event.ctrlKey
  ) {
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    openSelection();
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "i") {
    event.preventDefault();
    addFamily(event.currentTarget.closest("tr").dataset.bigBet);
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key === "Backspace") {
    event.preventDefault();
    state.selection = { kind: "family", issue: Number(event.currentTarget.closest("tr").dataset.issue) };
    state.inspectorOpen = false;
    deleteSelection();
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.shiftKey && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
    event.preventDefault();
    const family = findFamily(event.currentTarget.closest("tr").dataset.issue);
    moveFamilyToBetRank(family.issue, family.big_bet, Math.max(1, (family.rank || 1) + (event.key === "ArrowUp" ? -1 : 1)));
    state.selection = { kind: "family", issue: family.issue };
    state.inspectorOpen = false;
    commitRegistry("Reordered idea family row.");
    return;
  }
  if (event.key === "Enter") {
    event.preventDefault();
    if (event.currentTarget.isContentEditable) event.currentTarget.blur();
    else openSelection();
    return;
  }
  if (event.key === "Tab") {
    event.preventDefault();
    focusRelativeCell(event.currentTarget, event.shiftKey ? -1 : 1);
    return;
  }
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    focusRelativeCell(event.currentTarget, event.key === "ArrowRight" ? 1 : -1);
    return;
  }
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    focusVerticalCell(event.currentTarget, event.key === "ArrowDown" ? 1 : -1);
  }
}

function focusRelativeCell(current, delta) {
  const cells = [...document.querySelectorAll("[data-cell]")];
  const index = cells.indexOf(current);
  const target = cells[index + delta];
  if (target) focusCell(target);
}

function focusVerticalCell(current, delta) {
  const field = current.dataset.field;
  const row = current.closest("tr");
  const rows = [...document.querySelectorAll("tr.family-row")];
  const index = rows.indexOf(row);
  const target = rows[index + delta]?.querySelector(`[data-field="${field}"]`);
  if (target) focusCell(target);
}

function focusCell(cellNode) {
  cellNode.focus();
  if (cellNode.tagName === "SELECT") return;
  const range = document.createRange();
  range.selectNodeContents(cellNode);
  range.collapse(false);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
}

function handleSheetDrop(targetRow) {
  if (!state.dragRow) return;
  const payload = state.dragRow;
  document.querySelectorAll(".drop-target").forEach((node) => node.classList.remove("drop-target"));
  if (payload.kind === "family") {
    const targetBet = targetRow.dataset.kind === "bet" ? targetRow.dataset.id : targetRow.dataset.bigBet;
    const targetRank = targetRow.dataset.kind === "family" ? rowIndexInBet(targetRow) + 1 : 1;
    moveFamilyToBetRank(Number(payload.id), targetBet, targetRank);
    state.selection = { kind: "family", issue: Number(payload.id) };
    commitRegistry("Moved idea family row.");
    return;
  }
  if (payload.kind === "bet") {
    const targetBet = targetRow.dataset.kind === "bet" ? targetRow.dataset.id : targetRow.dataset.bigBet;
    if (!targetBet) return;
    const target = findBet(targetBet);
    moveBetToWaveRank(payload.id, target.wave, target.rank || 1);
    state.selection = { kind: "bet", id: payload.id };
    commitRegistry("Moved big bet row.");
  }
}

function rowIndexInBet(row) {
  const rows = [...document.querySelectorAll(`tr.family-row[data-big-bet="${cssEscape(row.dataset.bigBet)}"]`)];
  return rows.indexOf(row);
}

function moveFamilyToBetRank(issue, bigBetId, rank) {
  const family = findFamily(issue);
  family.big_bet = bigBetId;
  const siblings = state.registry.idea_families
    .filter((item) => item.big_bet === bigBetId && item.issue !== issue)
    .sort(ideaFamilySort);
  siblings.splice(Math.max(0, rank - 1), 0, family);
  siblings.forEach((item, index) => {
    item.rank = index + 1;
  });
  normalizeFamilyRanks();
}

function handleRowAction(row, action) {
  const kind = row.dataset.kind;
  if (action === "focus") {
    if (kind === "bet") selectBet(row.dataset.id, true);
    if (kind === "family") selectFamily(row.dataset.issue, true);
    return;
  }
  if (action === "child") return addFamily(row.dataset.id);
  if (action === "hydrate") {
    applyTableRow(row);
    if (kind === "family") selectFamily(row.dataset.issue, true);
    return hydrateIssueFromSelection();
  }
  if (action === "delete") {
    if (kind === "bet") state.selection = { kind: "bet", id: row.dataset.id };
    if (kind === "family") state.selection = { kind: "family", issue: Number(row.dataset.issue) };
    return deleteSelection();
  }
}

function applyTableRow(row) {
  if (!row || row.dataset.kind !== "family") return;
  try {
    const previousIssue = Number(row.dataset.issue);
    const existing = findFamily(previousIssue);
    const value = (name) => cellValue(row, name);
    const updated = {
      ...existing,
      issue: positiveInt(Number(value("issue")), "family.issue"),
      slug: optionalIdentifier(value("slug"), "family.slug"),
      title: requiredString(value("title"), "family.title"),
      priority: priority(requiredString(value("priority"), "family.priority"), "family.priority"),
      rank: optionalPositiveInt(value("rank"), "family.rank") || existing.rank,
      status: status(requiredString(value("status"), "family.status"), "family.status"),
      role: optionalString(value("role")),
      next_action: optionalString(value("next_action")),
      artifact: optionalString(value("artifact")),
      url: optionalString(value("url")) || existing.url || null,
      links: existing.links || [],
    };
    state.registry.idea_families = state.registry.idea_families.filter((family) => family.issue !== previousIssue && family.issue !== updated.issue);
    state.registry.idea_families.push(updated);
    state.selection = { kind: "family", issue: updated.issue };
    commitRegistry("Applied sheet edit.");
  } catch (error) {
    markInvalid(error.message);
  }
}

function cellValue(row, name) {
  const node = row.querySelector(`[data-field="${name}"]`);
  if (!node) return "";
  if (node.tagName === "SELECT") return node.value.trim();
  if (node.dataset.value != null) return node.dataset.value.trim();
  return node.textContent?.trim() || "";
}

function renderInspector() {
  const panel = $("inspector-panel");
  if (!state.selection || !state.inspectorOpen) {
    panel.classList.remove("active");
    return;
  }
  const item = state.selection.kind === "bet" ? findBet(state.selection.id) : findFamily(state.selection.issue);
  if (!item) {
    clearSelection();
    return;
  }
  panel.classList.add("active");
  $("inspector-kicker").textContent = state.selection.kind === "bet" ? "Big bet" : "Idea family";
  $("inspector-title").textContent = state.selection.kind === "bet" ? item.title : `#${item.issue}: ${item.title}`;
  $("inspector-hint").textContent = state.selection.kind === "bet"
    ? "Edit the full bet contract, including narrative, risks, and dependency edges."
    : "Edit the durable issue-family lane. The sheet row remains the compact ranking view.";
  $("inspector-toolbar").innerHTML = state.selection.kind === "bet" ? betToolbar(item) : familyToolbar();
  $("inspector-fields").innerHTML = state.selection.kind === "bet" ? betFields(item) : familyFields(item);
  wireInspectorToolbar();
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function betToolbar(bet) {
  return `
    <button type="button" data-inspector-action="rank-up">Move left</button>
    <button type="button" data-inspector-action="rank-down">Move right</button>
    <button type="button" data-inspector-action="wave-earlier">Earlier priority</button>
    <button type="button" data-inspector-action="wave-later">Later priority</button>
    <button type="button" data-inspector-action="add-family">Add family to ${escapeHtml(bet.id)}</button>
  `;
}

function familyToolbar() {
  return `
    <button type="button" data-inspector-action="rank-up">Move up</button>
    <button type="button" data-inspector-action="rank-down">Move down</button>
    <button type="button" data-inspector-action="hydrate">Autofill from links</button>
  `;
}

function wireInspectorToolbar() {
  $("inspector-toolbar").querySelectorAll("[data-inspector-action]").forEach((button) => {
    button.addEventListener("click", () => handleInspectorAction(button.dataset.inspectorAction));
  });
}

function handleInspectorAction(action) {
  if (!state.selection) return;
  if (state.selection.kind === "bet" && (action === "rank-up" || action === "rank-down")) {
    const bet = findBet(state.selection.id);
    moveBetToWaveRank(bet.id, bet.wave, Math.max(1, (bet.rank || 1) + (action === "rank-up" ? -1 : 1)));
    commitRegistry("Adjusted bet order.");
    return;
  }
  if (state.selection.kind === "family" && (action === "rank-up" || action === "rank-down")) {
    const family = findFamily(state.selection.issue);
    moveFamilyToBetRank(family.issue, family.big_bet, Math.max(1, (family.rank || 1) + (action === "rank-up" ? -1 : 1)));
    commitRegistry("Adjusted family order.");
    return;
  }
  if (state.selection.kind === "bet" && (action === "wave-earlier" || action === "wave-later")) {
    const bet = findBet(state.selection.id);
    moveBetToWaveRank(bet.id, Math.max(1, bet.wave + (action === "wave-earlier" ? -1 : 1)), bet.rank || 1);
    commitRegistry("Adjusted bet layer.");
    return;
  }
  if (state.selection.kind === "bet" && action === "add-family") return addFamily(state.selection.id);
  if (state.selection.kind === "family" && action === "hydrate") return hydrateIssueFromSelection();
}

function betFields(bet) {
  const betIds = state.registry.big_bets.map((item) => item.id);
  return [
    field("id", "ID", bet.id),
    field("title", "Title", bet.title),
    field("priority", "Priority", bet.priority, "select", ["P0", "P1", "P2", "P3"]),
    field("rank", "Position", bet.rank || "", "number"),
    field("status", "Status", bet.status, "select", [...STATUS_VALUES]),
    field("unlock_state", "Unlock state", bet.unlock_state, "select", [...UNLOCK_STATE_VALUES]),
    field("unlock_evidence", "Unlock evidence", bet.unlock_evidence || "", "textarea wide"),
    field("confidence", "Confidence", bet.confidence || "", "text", ["high", "medium_high", "medium", "low"]),
    field("narrative", "Narrative", bet.narrative, "textarea wide"),
    field("near_term_win", "Near-term win", bet.near_term_win, "textarea wide"),
    field("long_term_unlock", "Long-term unlock", bet.long_term_unlock, "textarea wide"),
    field("next_action", "Next action", bet.next_action || "", "textarea wide"),
    field("risk", "Risk", bet.risk || "", "textarea wide"),
    field("depends_on", "Depends on IDs", bet.depends_on.join(", "), "text", betIds),
    field("unlocks", "Unlocks IDs", bet.unlocks.join(", "), "text", betIds),
    field("edge_labels", "Edge labels", edgeLabelsText(bet.edge_labels || []), "textarea wide"),
  ].join("");
}

function familyFields(family) {
  const artifacts = uniqueOptionValues(state.registry.idea_families.map((item) => item.artifact));
  const urls = uniqueOptionValues(state.registry.idea_families.map((item) => item.url));
  const titles = uniqueOptionValues(state.registry.idea_families.map((item) => item.title));
  return [
    field("issue", "Issue", family.issue, "number"),
    field("slug", "Slug", family.slug || "", "text", uniqueOptionValues(state.registry.idea_families.map((item) => item.slug))),
    field("title", "Title", family.title, "text", titles),
    field("big_bet", "Big bet", family.big_bet, "select", state.registry.big_bets.map((bet) => bet.id)),
    field("priority", "Lane priority", family.priority, "select", ["P0", "P1", "P2", "P3"]),
    field("rank", "Position inside bet", family.rank || "", "number"),
    field("status", "Status", family.status, "select", [...STATUS_VALUES]),
    field("role", "Type", family.role || "", "select", ["", ...ROLE_VALUES]),
    field("next_action", "Next action", family.next_action || "", "textarea wide"),
    field("artifact", "Seed URL", family.artifact || "", "url wide", artifacts),
    field("url", "Issue/PR URL", family.url || "", "url wide", urls),
    field("links", "Refs", linksText(family.links || []), "textarea wide"),
  ].join("");
}

function field(name, label, value, kind = "text", options = []) {
  const className = kind.includes("wide") ? "wide" : "";
  const normalizedKind = kind.split(" ")[0];
  if (normalizedKind === "textarea") {
    return `<label class="${className}">${escapeHtml(label)} <textarea data-inspect-field="${escapeAttr(name)}">${escapeHtml(value)}</textarea></label>`;
  }
  if (normalizedKind === "select") {
    const optionHtml = options.map((option) => `<option value="${escapeAttr(option)}"${String(option) === String(value) ? " selected" : ""}>${escapeHtml(option)}</option>`).join("");
    return `<label class="${className}">${escapeHtml(label)} <select data-inspect-field="${escapeAttr(name)}">${optionHtml}</select></label>`;
  }
  const inputType = ["number", "text", "url"].includes(normalizedKind) ? normalizedKind : "text";
  const optionValues = uniqueOptionValues(options);
  const listId = optionValues.length ? `bigbets-options-${escapeAttr(name)}` : "";
  const list = optionValues.length
    ? `<datalist id="${listId}">${optionValues.map((option) => `<option value="${escapeAttr(option)}"></option>`).join("")}</datalist>`
    : "";
  const listAttr = listId ? ` list="${listId}"` : "";
  return `<label class="${className}">${escapeHtml(label)} <input data-inspect-field="${escapeAttr(name)}" type="${inputType}" value="${escapeAttr(value)}"${listAttr}>${list}</label>`;
}

function uniqueOptionValues(values) {
  return [...new Set((values || []).map((value) => String(value || "").trim()).filter(Boolean))]
    .sort((a, b) => a.localeCompare(b));
}

function linksText(links) {
  return (links || [])
    .map((link) => [link.kind || "link", link.label, link.url].filter(Boolean).join(" | "))
    .join("\\n");
}

function edgeLabelsText(edgeLabels) {
  return (edgeLabels || [])
    .map((edge) => `${edge.target}: ${edge.label}`)
    .join("\\n");
}

function parseLinksText(value, label) {
  return String(value || "")
    .split("\\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const parts = line.split("|").map((part) => part.trim()).filter(Boolean);
      if (parts.length === 3) return { kind: optionalString(parts[0]), label: requiredString(parts[1], `${label}[${index}].label`), url: requiredString(parts[2], `${label}[${index}].url`) };
      if (parts.length === 2) return { kind: null, label: requiredString(parts[0], `${label}[${index}].label`), url: requiredString(parts[1], `${label}[${index}].url`) };
      throw new Error(`${label}[${index}] must use "kind | label | url" or "label | url".`);
    });
}

function parseEdgeLabelsText(value, label) {
  return String(value || "")
    .split("\\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const separator = line.indexOf(":");
      if (separator === -1) throw new Error(`${label}[${index}] must use "target: label".`);
      return {
        target: identifier(line.slice(0, separator).trim(), `${label}[${index}].target`),
        label: requiredString(line.slice(separator + 1), `${label}[${index}].label`),
      };
    });
}

function applyInspector() {
  if (!state.selection) return;
  try {
    const value = (name) => $inspect(name)?.value || "";
    if (state.selection.kind === "bet") {
      const previousId = state.selection.id;
      const updated = {
        id: identifier(requiredString(value("id"), "bet.id"), "bet.id"),
        title: requiredString(value("title"), "bet.title"),
        priority: priority(requiredString(value("priority"), "bet.priority"), "bet.priority"),
        rank: optionalPositiveInt(value("rank"), "bet.rank"),
        wave: priorityRank(priority(requiredString(value("priority"), "bet.priority"), "bet.priority")) + 1,
        status: status(requiredString(value("status"), "bet.status"), "bet.status"),
        unlock_state: unlockState(requiredString(value("unlock_state"), "bet.unlock_state"), "bet.unlock_state"),
        unlock_evidence: optionalString(value("unlock_evidence")),
        narrative: requiredString(value("narrative"), "bet.narrative"),
        near_term_win: requiredString(value("near_term_win"), "bet.near_term_win"),
        long_term_unlock: requiredString(value("long_term_unlock"), "bet.long_term_unlock"),
        next_action: optionalString(value("next_action")),
        confidence: optionalString(value("confidence")),
        risk: optionalString(value("risk")),
        depends_on: splitIdentifiers(value("depends_on"), "bet.depends_on"),
        unlocks: splitIdentifiers(value("unlocks"), "bet.unlocks"),
        edge_labels: parseEdgeLabelsText(value("edge_labels"), "bet.edge_labels"),
      };
      state.registry.big_bets = state.registry.big_bets.filter((bet) => bet.id !== previousId && bet.id !== updated.id);
      state.registry.big_bets.push(updated);
      state.registry.idea_families = state.registry.idea_families.map((family) =>
        family.big_bet === previousId ? { ...family, big_bet: updated.id } : family
      );
      state.selection = { kind: "bet", id: updated.id };
    } else {
      const previousIssue = state.selection.issue;
      const updated = {
        issue: positiveInt(Number(value("issue")), "family.issue"),
        slug: optionalIdentifier(value("slug"), "family.slug"),
        title: requiredString(value("title"), "family.title"),
        big_bet: identifier(requiredString(value("big_bet"), "family.big_bet"), "family.big_bet"),
        priority: priority(requiredString(value("priority"), "family.priority"), "family.priority"),
        rank: optionalPositiveInt(value("rank"), "family.rank"),
        status: status(requiredString(value("status"), "family.status"), "family.status"),
        role: optionalString(value("role")),
        next_action: optionalString(value("next_action")),
        artifact: optionalString(value("artifact")),
        url: optionalString(value("url")),
        links: parseLinksText(value("links"), "family.links"),
      };
      state.registry.idea_families = state.registry.idea_families.filter((family) => family.issue !== previousIssue && family.issue !== updated.issue);
      state.registry.idea_families.push(updated);
      state.selection = { kind: "family", issue: updated.issue };
    }
    commitRegistry("Applied inspector changes.");
  } catch (error) {
    markInvalid(error.message);
  }
}

function $inspect(name) {
  return document.querySelector(`[data-inspect-field="${name}"]`);
}

async function hydrateIssueFromSelection() {
  if (!state.selection || state.selection.kind !== "family") return;
  const family = findFamily(state.selection.issue);
  const currentArtifact = $inspect("artifact")?.value || family.artifact || "";
  const parsed = parseIssueUrl($inspect("url")?.value || family.url || "");
  let data = parsed;
  const resolver = adapter.fetchIssue || window.bigbetsIssueResolver;
  if (typeof resolver === "function") {
    data = { ...parsed, ...(await resolver({ url: parsed.url, issue: parsed.issue })) };
  }
  const inferred = inferFamilyFields({ ...family, artifact: currentArtifact, url: data.url || parsed.url });
  if (data.issue && $inspect("issue")) $inspect("issue").value = String(data.issue);
  if (data.url && $inspect("url")) $inspect("url").value = data.url;
  if ($inspect("title") && (!isDefaultTitle($inspect("title").value) || data.title || inferred.title)) {
    $inspect("title").value = data.title || inferred.title || $inspect("title").value;
  }
  if ($inspect("role") && !$inspect("role").value && inferred.role) $inspect("role").value = inferred.role;
  if ($inspect("artifact") && inferred.artifact) $inspect("artifact").value = inferred.artifact;
  if ($inspect("url") && !($inspect("url").value || "").trim() && inferred.url) $inspect("url").value = inferred.url;
  setLog("Autofilled fields from issue/artifact links. Apply changes to persist.", false);
}

function parseIssueUrl(url) {
  const match = String(url).match(/(?:issues|pull)\\/(\\d+)(?:\\b|$)/);
  return {
    issue: match ? Number(match[1]) : null,
    url: url || null,
    title: null,
  };
}

function inferFamilyFields(family) {
  const artifact = family.artifact || "";
  const url = family.url || "";
  return {
    artifact,
    url,
    title: isDefaultTitle(family.title) ? titleFromArtifact(artifact) : null,
    role: family.role || roleFromLinks(url, artifact),
  };
}

function isDefaultTitle(value) {
  return !String(value || "").trim() || String(value).trim() === "New idea family";
}

function titleFromArtifact(value) {
  const leaf = String(value || "").split("/").filter(Boolean).at(-1) || "";
  if (!leaf) return null;
  return leaf
    .replace(/\\.autoclanker\\.ideas\\.json$/i, "")
    .replace(/\\.ideas\\.json$/i, "")
    .replace(/\\.json$/i, "")
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\\b\\w/g, (char) => char.toUpperCase());
}

function roleFromLinks(url, artifact) {
  if (/\\.ideas\\.json(?:\\?|#|$)/i.test(artifact)) return "ideas-lane";
  if (/\\/pull\\/\\d+(?:\\b|$)/.test(String(url))) return "wip";
  return null;
}

function addBet() {
  const nextId = uniqueId("new_big_bet", new Set(state.registry.big_bets.map((bet) => bet.id)));
  const nextWave = Math.max(1, ...state.registry.big_bets.map((bet) => bet.wave || 1));
  const nextRank = nextRankFor(state.registry.big_bets.filter((bet) => bet.wave === nextWave));
  state.registry.big_bets.push({
    id: nextId,
    title: "New big bet",
    priority: priorityForWave(nextWave),
    rank: nextRank,
    wave: nextWave,
    status: "candidate",
    unlock_state: "emerging",
    unlock_evidence: null,
    narrative: "Describe why this bet matters.",
    near_term_win: "Define one measurable near-term win.",
    long_term_unlock: "Define what this unlocks later.",
    next_action: "Identify the next concrete action.",
    confidence: null,
    risk: null,
    depends_on: [],
    unlocks: [],
    edge_labels: [],
  });
  state.selection = { kind: "bet", id: nextId };
  state.inspectorOpen = true;
  commitRegistry("Added new big bet.");
}

function addFamily(bigBetId = null) {
  const existing = new Set(state.registry.idea_families.map((family) => family.issue));
  let issue = 1;
  while (existing.has(issue)) issue += 1;
  const targetBet = bigBetId || (state.selection?.kind === "bet" ? state.selection.id : null) || state.registry.big_bets[0]?.id || "";
  state.registry.idea_families.push({
    issue,
    slug: null,
    title: "New idea family",
    big_bet: targetBet,
    priority: "P2",
    rank: nextRankFor(state.registry.idea_families.filter((family) => family.big_bet === targetBet)),
    status: "candidate",
    role: null,
    next_action: "Define the next exploration step.",
    artifact: null,
    url: null,
    links: [],
  });
  state.selection = { kind: "family", issue };
  state.inspectorOpen = true;
  commitRegistry("Added new idea family.");
}

function duplicateSelection() {
  if (!state.selection) return;
  if (state.selection.kind === "bet") {
    const bet = findBet(state.selection.id);
    const id = uniqueId(`${bet.id}_copy`, new Set(state.registry.big_bets.map((item) => item.id)));
    state.registry.big_bets.push({ ...bet, id, title: `${bet.title} copy`, rank: nextRankFor(state.registry.big_bets.filter((item) => item.wave === bet.wave)), depends_on: [...bet.depends_on], unlocks: [...bet.unlocks], edge_labels: [...(bet.edge_labels || [])] });
    state.selection = { kind: "bet", id };
    state.inspectorOpen = true;
  } else {
    const family = findFamily(state.selection.issue);
    const existing = new Set(state.registry.idea_families.map((item) => item.issue));
    let issue = family.issue + 1;
    while (existing.has(issue)) issue += 1;
    state.registry.idea_families.push({ ...family, issue, slug: null, title: `${family.title} copy`, rank: nextRankFor(state.registry.idea_families.filter((item) => item.big_bet === family.big_bet)), links: [...(family.links || [])] });
    state.selection = { kind: "family", issue };
    state.inspectorOpen = true;
  }
  commitRegistry("Duplicated selected item.");
}

function deleteSelection() {
  if (!state.selection) return;
  if (state.selection.kind === "bet") {
    const id = state.selection.id;
    state.registry.big_bets = state.registry.big_bets.filter((bet) => bet.id !== id);
  } else {
    const issue = state.selection.issue;
    state.registry.idea_families = state.registry.idea_families.filter((family) => family.issue !== issue);
  }
  state.selection = null;
  state.inspectorOpen = false;
  commitRegistry("Deleted selected item.");
}

function commitRegistry(message) {
  try {
    const { registry } = validateRegistry(state.registry);
    state.registry = registry;
    $("registry-editor").value = JSON.stringify(registry, null, 2);
    renderAll(message);
  } catch (error) {
    $("registry-editor").value = JSON.stringify(state.registry, null, 2);
    markInvalid(error.message);
  }
}

function validateRegistry(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("Registry must be a JSON object.");
  }
  if (payload.schema_version && payload.schema_version !== REGISTRY_SCHEMA_VERSION) {
    throw new Error(`Unsupported schema_version ${payload.schema_version}; expected ${REGISTRY_SCHEMA_VERSION}.`);
  }
  const metadata = objectOrDefault(payload.metadata, "metadata");
  const registry = {
    schema_version: REGISTRY_SCHEMA_VERSION,
    metadata: {
      title: optionalString(metadata.title) || "Big Bets Portfolio",
      description: optionalString(metadata.description),
      updated_at: optionalString(metadata.updated_at),
      links: parseLinks(metadata.links, "metadata.links"),
      max_p0_big_bets: positiveInt(metadata.max_p0_big_bets, "metadata.max_p0_big_bets", 3),
    },
    big_bets: arrayRequired(payload.big_bets, "big_bets").map(parseBigBet),
    idea_families: arrayRequired(payload.idea_families, "idea_families").map(parseIdeaFamily),
  };
  registry.big_bets.sort(bigBetSort);
  registry.idea_families.sort(ideaFamilySort);
  validateSemantics(registry);
  return { registry };
}

function parseBigBet(item, index) {
  const label = `big_bets[${index}]`;
  const value = objectRequired(item, label);
  return {
    id: identifier(requiredString(value.id, `${label}.id`), `${label}.id`),
    title: requiredString(value.title, `${label}.title`),
    priority: priority(requiredString(value.priority, `${label}.priority`), `${label}.priority`),
    rank: optionalPositiveInt(value.rank, `${label}.rank`),
    wave: positiveInt(value.wave, `${label}.wave`),
    status: status(requiredString(value.status, `${label}.status`), `${label}.status`),
    unlock_state: unlockState(optionalString(value.unlock_state) || defaultUnlockState(value), `${label}.unlock_state`),
    unlock_evidence: optionalString(value.unlock_evidence),
    narrative: requiredString(value.narrative, `${label}.narrative`),
    near_term_win: requiredString(value.near_term_win, `${label}.near_term_win`),
    long_term_unlock: requiredString(value.long_term_unlock, `${label}.long_term_unlock`),
    next_action: optionalString(value.next_action),
    confidence: optionalString(value.confidence),
    risk: optionalString(value.risk),
    depends_on: optionalIdentifierArray(value.depends_on, `${label}.depends_on`),
    unlocks: optionalIdentifierArray(value.unlocks, `${label}.unlocks`),
    edge_labels: parseEdgeLabels(value.edge_labels, `${label}.edge_labels`),
  };
}

function parseIdeaFamily(item, index) {
  const label = `idea_families[${index}]`;
  const value = objectRequired(item, label);
  return {
    issue: positiveInt(value.issue, `${label}.issue`),
    slug: optionalIdentifier(value.slug, `${label}.slug`),
    title: requiredString(value.title, `${label}.title`),
    big_bet: identifier(requiredString(value.big_bet, `${label}.big_bet`), `${label}.big_bet`),
    priority: priority(requiredString(value.priority, `${label}.priority`), `${label}.priority`),
    rank: optionalPositiveInt(value.rank, `${label}.rank`),
    status: status(requiredString(value.status, `${label}.status`), `${label}.status`),
    role: optionalString(value.role),
    next_action: optionalString(value.next_action),
    artifact: optionalString(value.artifact),
    url: optionalString(value.url),
    links: parseLinks(value.links, `${label}.links`),
  };
}

function validateSemantics(registry) {
  const betIds = registry.big_bets.map((bet) => bet.id);
  const idSet = new Set(betIds);
  const duplicateBets = duplicates(betIds);
  if (duplicateBets.length) throw new Error(`Duplicate big_bets ids: ${duplicateBets.join(", ")}`);
  const duplicateIssues = duplicates(registry.idea_families.map((family) => family.issue));
  if (duplicateIssues.length) throw new Error(`Duplicate idea family issues: ${duplicateIssues.join(", ")}`);
  const unknownFamilyBets = [...new Set(registry.idea_families.map((family) => family.big_bet).filter((id) => !idSet.has(id)))];
  if (unknownFamilyBets.length) throw new Error(`Idea families reference unknown big bets: ${unknownFamilyBets.join(", ")}`);
  const badEdges = [];
  registry.big_bets.forEach((bet) => {
    [...bet.depends_on, ...bet.unlocks].forEach((target) => {
      if (!idSet.has(target)) badEdges.push(target);
      if (target === bet.id) throw new Error(`Big bet ${bet.id} links to itself.`);
    });
    const expectedPriority = priorityForWave(bet.wave);
    if (bet.priority !== expectedPriority) {
      throw new Error(`${bet.id} has ${bet.priority}, but wave ${bet.wave} requires ${expectedPriority}.`);
    }
    if (bet.unlock_state === "unlocked" && !bet.unlock_evidence) {
      throw new Error(`${bet.id} is unlocked and must set unlock_evidence.`);
    }
    bet.edge_labels.forEach((edge) => {
      if (![...bet.depends_on, ...bet.unlocks].includes(edge.target)) {
        throw new Error(`${bet.id} labels ${edge.target}, but no matching edge exists.`);
      }
    });
    if (priorityRank(bet.priority) <= 1 && !bet.next_action) {
      throw new Error(`${bet.id} is ${bet.priority} and must set next_action.`);
    }
  });
  registry.idea_families.forEach((family) => {
    if (family.role && !ROLE_VALUES.has(family.role)) {
      throw new Error(`#${family.issue} has unsupported role ${family.role}. Use one of: ${[...ROLE_VALUES].join(", ")}.`);
    }
    family.links.forEach((link) => {
      if (link.kind && !LINK_KIND_VALUES.has(link.kind)) {
        throw new Error(`#${family.issue} has unsupported link kind ${link.kind}. Use one of: ${[...LINK_KIND_VALUES].join(", ")}.`);
      }
    });
  });
  registry.metadata.links.forEach((link) => {
    if (link.kind && !LINK_KIND_VALUES.has(link.kind)) {
      throw new Error(`metadata.links has unsupported link kind ${link.kind}. Use one of: ${[...LINK_KIND_VALUES].join(", ")}.`);
    }
  });
  if (badEdges.length) throw new Error(`Edges reference unknown big bets: ${[...new Set(badEdges)].join(", ")}`);
  const p0Count = registry.big_bets.filter((bet) => bet.priority === "P0").length;
  if (p0Count > registry.metadata.max_p0_big_bets) {
    throw new Error(`Registry has ${p0Count} P0 big bets; max is ${registry.metadata.max_p0_big_bets}.`);
  }
  const familiesByBet = familiesByBigBet(registry);
  const emptyActive = registry.big_bets
    .filter((bet) => !["candidate", "parked", "superseded", "closed"].includes(bet.status))
    .filter((bet) => !(familiesByBet.get(bet.id) || []).length)
    .map((bet) => bet.id);
  if (emptyActive.length) throw new Error(`Active big bets need idea families: ${emptyActive.join(", ")}`);
}

function renderArtifacts(registry) {
  const edges = normalizedEdges(registry);
  const familiesByBet = familiesByBigBet(registry);
  const bigBets = registry.big_bets.map((bet) => {
    const families = familiesByBet.get(bet.id) || [];
    return { ...bet, idea_family_issues: families.map((family) => family.issue), idea_family_count: families.length };
  });
  const normalized = {
    schema_version: REGISTRY_SCHEMA_VERSION,
    artifact_schema_version: ARTIFACT_SCHEMA_VERSION,
    generator: GENERATOR,
    metadata: registry.metadata,
    summary: {
      big_bet_count: registry.big_bets.length,
      idea_family_count: registry.idea_families.length,
      edge_count: edges.length,
      p0_big_bet_count: registry.big_bets.filter((bet) => bet.priority === "P0").length,
    },
    big_bets: bigBets,
    idea_families: registry.idea_families,
    edges,
  };
  const mermaid = renderMermaid(registry, edges, familiesByBet);
  const excalidraw = renderExcalidraw(registry, edges, familiesByBet);
  return {
    normalized,
    registryJson: JSON.stringify(registry, null, 2) + "\\n",
    normalizedJson: JSON.stringify(normalized, null, 2) + "\\n",
    artifactMetadataJson: JSON.stringify(artifactMetadata(), null, 2) + "\\n",
    csv: renderCsv(registry, familiesByBet),
    markdown: renderMarkdown(registry, mermaid, familiesByBet),
    mermaid,
    excalidraw,
    svg: renderSvg(registry, edges, familiesByBet),
  };
}

function renderCsv(registry, familiesByBet) {
  const rows = [["kind", "big_bet_priority", "big_bet_order", "wave", "big_bet_id", "big_bet_title", "big_bet_status", "unlock_state", "unlock_evidence", "lane_priority", "lane_order", "issue", "idea_family_slug", "idea_family_title", "idea_family_status", "role", "next_action", "artifact", "url", "links", "schema_version", "generator_version"]];
  registry.big_bets.forEach((bet) => {
    const families = familiesByBet.get(bet.id) || [];
    rows.push(["bet", bet.priority, bet.rank || "", bet.wave, bet.id, bet.title, bet.status, bet.unlock_state, bet.unlock_evidence || "", "", "", "", "", "", "", "", bet.next_action || "", "", "", "", REGISTRY_SCHEMA_VERSION, GENERATOR.version || "unknown"]);
    families.forEach((family) => rows.push(["family", bet.priority, bet.rank || "", bet.wave, bet.id, bet.title, bet.status, bet.unlock_state, bet.unlock_evidence || "", family.priority, family.rank || "", family.issue, family.slug || "", family.title, family.status, family.role || "", family.next_action || bet.next_action || "", family.artifact || "", family.url || "", linksText(family.links || []), REGISTRY_SCHEMA_VERSION, GENERATOR.version || "unknown"]));
  });
  return rows.map((row) => row.map(csvCell).join(",")).join("\\n") + "\\n";
}

function renderMarkdown(registry, mermaid, familiesByBet) {
  const lines = [
    `# ${registry.metadata.title}`,
    "",
    `Generated by \\`bigbets ${GENERATOR.version || "unknown"}\\`.`,
    "",
    `Registry schema: \\`${REGISTRY_SCHEMA_VERSION}\\`.`,
    "",
  ];
  if (registry.metadata.description) lines.push(registry.metadata.description, "");
  if (registry.metadata.updated_at) lines.push(`Updated: \\`${registry.metadata.updated_at}\\``, "");
  if (registry.metadata.links?.length) {
    lines.push("## Links", "");
    registry.metadata.links.forEach((link) => {
      const prefix = link.kind ? `${link.kind}: ` : "";
      lines.push(`- [${prefix}${link.label}](${link.url})`);
    });
    lines.push("");
  }
  lines.push("## Priority Queue", "");
  lines.push("| Priority | Big bet | Status | Idea families | Next action |");
  lines.push("| --- | --- | --- | --- | --- |");
  registry.big_bets.forEach((bet) => {
    const familyLinks = (familiesByBet.get(bet.id) || []).map(markdownIssueLink).join(", ") || "-";
    lines.push(`| ${bet.priority} | ${escapeMarkdownTable(bet.title)} | ${bet.status} | ${familyLinks} | ${escapeMarkdownTable(bet.next_action || "-")} |`);
  });
  lines.push("", "## Big Bets", "");
  registry.big_bets.forEach((bet) => {
    lines.push(`### ${bet.priority}: ${bet.title}`, "");
    lines.push(bet.narrative, "");
    lines.push(`- **Near-term win:** ${bet.near_term_win}`);
    lines.push(`- **Long-term unlock:** ${bet.long_term_unlock}`);
    lines.push(`- **Status:** ${bet.status}`);
    lines.push(`- **Unlock state:** ${bet.unlock_state}`);
    lines.push(`- **Unlock evidence:** ${bet.unlock_evidence || "-"}`);
    lines.push(`- **Next action:** ${bet.next_action || "-"}`);
    lines.push(`- **Confidence:** ${bet.confidence || "-"}`);
    lines.push(`- **Risk:** ${bet.risk || "-"}`, "");
  });
  lines.push("## Graph", "", "```mermaid", mermaid.trim(), "```", "");
  return lines.join("\\n");
}

function renderMermaid(registry, edges, familiesByBet) {
  const lines = [`%% schema_version=${REGISTRY_SCHEMA_VERSION} generator=bigbets@${GENERATOR.version || "unknown"}`, "flowchart TD"];
  waves(registry).forEach(([wave, bets]) => {
    lines.push(`  subgraph wave_${wave}[${priorityForWave(wave)}]`);
    bets.forEach((bet) => {
      const issueLabel = familyChipLabel(familiesByBet.get(bet.id) || []);
      lines.push(`    ${nodeId(bet.id)}["${escapeMermaid(`${bet.priority} / ${bet.title}\\\\n${bet.unlock_state}\\\\n${issueLabel}`)}"]`);
    });
    lines.push("  end");
  });
  edges.forEach((edge) => {
    const label = edge.label ? `|${escapeMermaid(edge.label)}|` : "";
    lines.push(`  ${nodeId(edge.from)} -->${label} ${nodeId(edge.to)}`);
  });
  return lines.join("\\n") + "\\n";
}

function boardLayout(registry) {
  const grouped = waves(registry);
  const maxCards = Math.max(1, ...grouped.map(([, bets]) => bets.length));
  const width = Math.max(1040, BOARD.marginX * 2 + maxCards * BOARD.cardWidth + Math.max(0, maxCards - 1) * BOARD.cardGap);
  const height = Math.max(520, BOARD.marginY * 2 + grouped.length * BOARD.cardHeight + Math.max(0, grouped.length - 1) * BOARD.waveGap);
  const positions = new Map();
  grouped.forEach(([wave, bets], waveIndex) => {
    const y = BOARD.marginY + waveIndex * (BOARD.cardHeight + BOARD.waveGap);
    bets.forEach((bet, cardIndex) => positions.set(bet.id, {
      x: BOARD.marginX + cardIndex * (BOARD.cardWidth + BOARD.cardGap),
      y,
      wave,
      cardWidth: BOARD.cardWidth,
      cardHeight: BOARD.cardHeight,
    }));
  });
  return { grouped, positions, width, height };
}

function renderSvg(registry, edges, familiesByBet) {
  const layout = boardLayout(registry);
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${layout.width}" height="${layout.height}" viewBox="0 0 ${layout.width} ${layout.height}" role="img" aria-label="${escapeAttr(registry.metadata.title)}">`,
    `<metadata>${escapeHtml(JSON.stringify(artifactMetadata()))}</metadata>`,
    "<defs>",
    '<pattern id="grid" width="56" height="56" patternUnits="userSpaceOnUse"><path d="M 56 0 L 0 0 0 56" fill="none" stroke="#ece5d8" stroke-width="0.9"/></pattern>',
    '<filter id="shadow" x="-8%" y="-8%" width="116%" height="132%"><feDropShadow dx="0" dy="8" stdDeviation="6" flood-color="#1e1e1e" flood-opacity="0.105"/></filter>',
    '<marker id="arrow" viewBox="0 0 10 10" refX="8.7" refY="5" markerWidth="6.2" markerHeight="6.2" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#1e1e1e"/></marker>',
    "</defs>",
    '<rect width="100%" height="100%" fill="#fdfbf4"/>',
    '<circle cx="190" cy="135" r="230" fill="#dbeafe" opacity="0.22"/>',
    `<circle cx="${layout.width - 170}" cy="145" r="250" fill="#d8f3dc" opacity="0.21"/>`,
    '<rect width="100%" height="100%" fill="url(#grid)" opacity="0.66"/>',
    `<text x="${BOARD.marginX}" y="52" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="31" font-weight="400" fill="#1e1e1e">${escapeHtml(registry.metadata.title)}</text>`,
  ];
  layout.grouped.forEach(([wave, bets]) => {
    const first = layout.positions.get(bets[0].id);
    if (first) parts.push(`<text x="${BOARD.marginX}" y="${first.y - 22}" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="18" font-weight="400" fill="#4b5563">${priorityForWave(wave)}</text>`);
  });
  edges.forEach((edge) => {
    const source = layout.positions.get(edge.from);
    const target = layout.positions.get(edge.to);
    if (!source || !target) return;
    const sx = source.x + BOARD.cardWidth / 2;
    const sy = source.y + BOARD.cardHeight;
    const tx = target.x + BOARD.cardWidth / 2;
    const ty = target.y;
    const bend = Math.max(46, Math.abs(ty - sy) / 2);
    const dash = edge.kind === "depends_on" ? ' stroke-dasharray="8 8"' : "";
    parts.push(...svgEdge(edge.from, edge.to, sx, sy, tx, ty, bend, dash, edge.label));
  });
  registry.big_bets.forEach((bet) => {
    const position = layout.positions.get(bet.id);
    if (!position) return;
    parts.push(svgCard(bet, familiesByBet.get(bet.id) || [], position));
  });
  parts.push("</svg>");
  return parts.join("\\n");
}

function svgCard(bet, families, position) {
  const color = priorityColor(bet.priority);
  const fill = priorityFill(bet.priority);
  const titleLines = wrapText(bet.title, 33, 3);
  const issueLabel = familyChipLabel(families) || "No mapped idea families";
  const parts = [
    `<g data-bet-id="${escapeAttr(bet.id)}" data-x="${position.x}" data-y="${position.y}" tabindex="0" role="button" aria-label="${escapeAttr(bet.title)}">`,
    `<title>Double-click or press Enter to edit. Drag the dotted handle to move.</title>`,
    `<path d="${roundedRectPath(position.x, position.y, BOARD.cardWidth, BOARD.cardHeight, 27)}" fill="${fill}" stroke="none" filter="url(#shadow)"/>`,
    ...roughRectPaths(position.x, position.y, BOARD.cardWidth, BOARD.cardHeight, 27, stableSeed(bet.id)),
    `<path d="${roughRectPath(position.x + 13, position.y + 13, BOARD.cardWidth - 26, BOARD.cardHeight - 26, 20, stableSeed(`inner:${bet.id}`))}" fill="none" stroke="${color}" stroke-width="1.05" stroke-linecap="round" stroke-linejoin="round" opacity="0.34"/>`,
    `<text x="${position.x + 23}" y="${position.y + 32}" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="12" font-weight="400" fill="${color}">${escapeHtml(bet.priority)} / ${escapeHtml(bet.status)} / ${escapeHtml(bet.unlock_state)}</text>`,
    `<g class="card-drag-handle" data-drag-handle="true" aria-label="Drag ${escapeAttr(bet.title)}">`,
    `<rect x="${position.x + BOARD.cardWidth - 54}" y="${position.y + 18}" width="31" height="24" rx="10" fill="#fffefa" stroke="${color}" stroke-width="1" opacity="0.78"/>`,
    `<circle cx="${position.x + BOARD.cardWidth - 44}" cy="${position.y + 27}" r="1.7" fill="${color}"/>`,
    `<circle cx="${position.x + BOARD.cardWidth - 34}" cy="${position.y + 27}" r="1.7" fill="${color}"/>`,
    `<circle cx="${position.x + BOARD.cardWidth - 44}" cy="${position.y + 35}" r="1.7" fill="${color}"/>`,
    `<circle cx="${position.x + BOARD.cardWidth - 34}" cy="${position.y + 35}" r="1.7" fill="${color}"/>`,
    `</g>`,
  ];
  titleLines.forEach((line, index) => {
    parts.push(`<text x="${position.x + 23}" y="${position.y + 61 + index * 19}" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="16" font-weight="400" fill="#1e1e1e">${escapeHtml(line)}</text>`);
  });
  parts.push(`<text x="${position.x + 23}" y="${position.y + BOARD.cardHeight - 20}" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="12" fill="#4b5563">${escapeHtml(issueLabel)}</text>`);
  parts.push(`<g class="card-action-hint" opacity="0">`);
  parts.push(`<rect x="${position.x + BOARD.cardWidth - 102}" y="${position.y + 20}" width="39" height="21" rx="9" fill="#fffefa" stroke="${color}" stroke-width="1" opacity="0.82"/>`);
  parts.push(`<text x="${position.x + BOARD.cardWidth - 82}" y="${position.y + 34}" text-anchor="middle" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="10.5" fill="${color}">Edit</text>`);
  parts.push(`</g>`);
  parts.push("</g>");
  return parts.join("\\n");
}

function svgEdge(source, target, sx, sy, tx, ty, bend, dash, label) {
  const seed = stableSeed(`edge:${source}:${target}`);
  const parts = [
    `<path d="${edgePath(sx, sy, tx, ty, bend, seed)}" fill="none" stroke="#1e1e1e" stroke-width="1.42" stroke-linecap="round" stroke-linejoin="round"${dash} marker-end="url(#arrow)"/>`,
    `<path d="${edgePath(sx, sy, tx, ty, bend, seed + 17)}" fill="none" stroke="#1e1e1e" stroke-width="0.72" stroke-linecap="round" stroke-linejoin="round" opacity="0.28"${dash}/>`,
  ];
  if (label) {
    const labelRatio = 0.38 + (seed % 29) / 100;
    const lx = sx + (tx - sx) * labelRatio + jitter(seed, 33, 34);
    const ly = sy + (ty - sy) * labelRatio - 8 + jitter(seed, 34, 20);
    parts.push(`<rect x="${fmt(lx - 64)}" y="${fmt(ly - 15)}" width="128" height="22" rx="11" fill="#fdfbf4" opacity="0.82"/>`);
    parts.push(`<text x="${fmt(lx)}" y="${fmt(ly)}" text-anchor="middle" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="11" fill="#5d6c84">${escapeHtml(label)}</text>`);
  }
  return parts;
}

function edgePath(sx, sy, tx, ty, bend, seed) {
  return `M ${fmt(sx + jitter(seed, 1))} ${fmt(sy + jitter(seed, 2))} C ${fmt(sx + jitter(seed, 3))} ${fmt(sy + bend + jitter(seed, 4))}, ${fmt(tx + jitter(seed, 5))} ${fmt(ty - bend + jitter(seed, 6))}, ${fmt(tx + jitter(seed, 7))} ${fmt(ty + jitter(seed, 8))}`;
}

function roughRectPaths(x, y, width, height, radius, seed) {
  return [
    `<path class="card-outline" d="${roughRectPath(x, y, width, height, radius, seed)}" fill="none" stroke="#1e1e1e" stroke-width="1.62" stroke-linecap="round" stroke-linejoin="round"/>`,
    `<path d="${roughRectPath(x + 1, y - 1, width - 2, height + 1, radius, seed + 31)}" fill="none" stroke="#1e1e1e" stroke-width="0.75" stroke-linecap="round" stroke-linejoin="round" opacity="0.34"/>`,
  ];
}

function roughRectPath(x, y, width, height, radius, seed) {
  const points = [
    [x + radius, y],
    [x + width - radius, y],
    [x + width, y + radius],
    [x + width, y + height - radius],
    [x + width - radius, y + height],
    [x + radius, y + height],
    [x, y + height - radius],
    [x, y + radius],
  ].map(([px, py], index) => [px + jitter(seed, index * 2), py + jitter(seed, index * 2 + 1)]);
  return `M ${fmt(points[0][0])} ${fmt(points[0][1])} L ${fmt(points[1][0])} ${fmt(points[1][1])} Q ${fmt(x + width + jitter(seed, 20))} ${fmt(y + jitter(seed, 21))} ${fmt(points[2][0])} ${fmt(points[2][1])} L ${fmt(points[3][0])} ${fmt(points[3][1])} Q ${fmt(x + width + jitter(seed, 22))} ${fmt(y + height + jitter(seed, 23))} ${fmt(points[4][0])} ${fmt(points[4][1])} L ${fmt(points[5][0])} ${fmt(points[5][1])} Q ${fmt(x + jitter(seed, 24))} ${fmt(y + height + jitter(seed, 25))} ${fmt(points[6][0])} ${fmt(points[6][1])} L ${fmt(points[7][0])} ${fmt(points[7][1])} Q ${fmt(x + jitter(seed, 26))} ${fmt(y + jitter(seed, 27))} ${fmt(points[0][0])} ${fmt(points[0][1])} Z`;
}

function roundedRectPath(x, y, width, height, radius) {
  return `M ${x + radius} ${y} H ${x + width - radius} Q ${x + width} ${y} ${x + width} ${y + radius} V ${y + height - radius} Q ${x + width} ${y + height} ${x + width - radius} ${y + height} H ${x + radius} Q ${x} ${y + height} ${x} ${y + height - radius} V ${y + radius} Q ${x} ${y} ${x + radius} ${y} Z`;
}

function jitter(seed, salt, amount = 0.9) {
  let value = (seed ^ Math.imul(salt, 0x9e3779b1)) >>> 0;
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb352d) >>> 0;
  value ^= value >>> 15;
  return (((value % 1000) / 999) - 0.5) * amount * 2;
}

function fmt(value) {
  return Number(value).toFixed(1);
}

function renderExcalidraw(registry, edges, familiesByBet) {
  const layout = boardLayout(registry);
  const elements = [];
  registry.big_bets.forEach((bet) => {
    const position = layout.positions.get(bet.id);
    if (!position) return;
    const rectangleId = stableId(`rect:${bet.id}`);
    const textId = stableId(`text:${bet.id}`);
    elements.push(excalidrawBase(rectangleId, "rectangle", position.x, position.y, BOARD.cardWidth, BOARD.cardHeight, {
      strokeColor: "#1e1e1e",
      backgroundColor: priorityFill(bet.priority),
      roundness: { type: 3 },
      boundElements: [{ type: "text", id: textId }],
      roughness: 2,
    }));
    const issues = familyChipLabel(familiesByBet.get(bet.id) || []);
    const text = `${bet.priority} / ${bet.title}\\n${bet.status} / ${bet.unlock_state}${issues ? `\\n${issues}` : ""}`;
    elements.push({
      ...excalidrawBase(textId, "text", position.x + 22, position.y + 22, BOARD.cardWidth - 44, 90, {
        strokeColor: "#1e1e1e",
        backgroundColor: "transparent",
        roundness: null,
        boundElements: null,
      }),
      fontSize: 16,
      fontFamily: 5,
      text,
      rawText: text,
      textAlign: "left",
      verticalAlign: "top",
      containerId: rectangleId,
      originalText: text,
      lineHeight: 1.25,
    });
  });
  edges.forEach((edge) => {
    const source = layout.positions.get(edge.from);
    const target = layout.positions.get(edge.to);
    if (!source || !target) return;
    const sx = source.x + BOARD.cardWidth / 2;
    const sy = source.y + BOARD.cardHeight;
    const tx = target.x + BOARD.cardWidth / 2;
    const ty = target.y;
    elements.push({
      ...excalidrawBase(stableId(`arrow:${edge.from}:${edge.to}`), "arrow", sx, sy, tx - sx, ty - sy, {
        strokeColor: edge.kind === "unlocks" ? "#1e1e1e" : "#5d6c84",
        backgroundColor: "transparent",
        strokeStyle: edge.kind === "unlocks" ? "solid" : "dashed",
        roundness: { type: 2 },
        roughness: 2,
      }),
      points: [[0, 0], [tx - sx, ty - sy]],
      lastCommittedPoint: null,
      startBinding: null,
      endBinding: null,
      startArrowhead: null,
      endArrowhead: "arrow",
    });
    if (edge.label) {
      const text = edge.label;
      const labelSeed = stableSeed(`edge-label:${edge.from}:${edge.to}`);
      const labelRatio = 0.38 + (labelSeed % 29) / 100;
      elements.push({
        ...excalidrawBase(stableId(`edge-label:${edge.from}:${edge.to}`), "text", sx + (tx - sx) * labelRatio - 70 + jitter(labelSeed, 1, 34), sy + (ty - sy) * labelRatio - 24 + jitter(labelSeed, 2, 20), 140, 24, {
          strokeColor: "#5d6c84",
          backgroundColor: "transparent",
          roundness: null,
          boundElements: null,
        }),
        fontSize: 13,
        fontFamily: 5,
        text,
        rawText: text,
        textAlign: "center",
        verticalAlign: "middle",
        containerId: null,
        originalText: text,
        lineHeight: 1.25,
      });
    }
  });
  return JSON.stringify({
    type: "excalidraw",
    version: 2,
    source: `bigbets ${GENERATOR.version || "unknown"}`,
    elements,
    appState: {
      viewBackgroundColor: "#fdfbf4",
      gridSize: 56,
      currentItemFontFamily: 5,
    },
    files: {},
  }, null, 2) + "\\n";
}

function excalidrawBase(id, type, x, y, width, height, overrides = {}) {
  return {
    id,
    type,
    x,
    y,
    width,
    height,
    angle: 0,
    strokeColor: "#1e1e1e",
    backgroundColor: "transparent",
    fillStyle: "solid",
    strokeWidth: 1,
    strokeStyle: "solid",
    roughness: 1,
    opacity: 100,
    groupIds: [],
    frameId: null,
    roundness: null,
    seed: stableSeed(id),
    version: 1,
    versionNonce: stableSeed(`nonce:${id}`),
    isDeleted: false,
    boundElements: null,
    updated: 1,
    link: null,
    locked: false,
    ...overrides,
  };
}

async function writeArtifacts() {
  if (!loadFromJsonEditor("Validated before writing artifacts.")) return;
  $("artifact-status").textContent = "Publishing artifacts...";
  const result = await callAdapter("writeArtifacts", { appId: APP_ID, artifacts: artifactMap() }, null);
  const links = (result?.links || []).map((link) => `<a href="${escapeAttr(link.href)}">${escapeHtml(link.label || link.href)}</a>`);
  $("artifact-links").innerHTML = links.length ? `<strong>Artifacts:</strong> ${links.join(" / ")}` : escapeHtml(result?.message || "Artifacts written.");
  $("artifact-status").textContent = links.length ? `Published ${links.length} artifacts` : (result?.message || "Artifacts cached");
  setLog(result?.message || "Generated artifacts written.", false);
}

function exportArtifact(kind) {
  if (!loadFromJsonEditor("Validated before export.")) return;
  const files = artifactMap();
  const selected = {
    metadata: ["big_bets.artifact_metadata.json", files["big_bets.artifact_metadata.json"], "application/json"],
    json: ["big_bets.input.registry.json", files["big_bets.input.registry.json"], "application/json"],
    "normalized-json": ["big_bets.registry.json", files["big_bets.registry.json"], "application/json"],
    csv: ["big_bets.rankings.csv", files["big_bets.rankings.csv"], "text/csv"],
    markdown: ["big_bets.md", files["big_bets.md"], "text/markdown"],
    mermaid: ["big_bets.mmd", files["big_bets.mmd"], "text/plain"],
    excalidraw: ["big_bets.excalidraw", files["big_bets.excalidraw"], "application/json"],
    svg: ["big_bets.svg", files["big_bets.svg"], "image/svg+xml"],
  }[kind];
  if (selected) download(...selected);
}

function artifactMap() {
  return {
    "big_bets.input.registry.json": state.rendered.registryJson,
    "big_bets.artifact_metadata.json": state.rendered.artifactMetadataJson,
    "big_bets.registry.json": state.rendered.normalizedJson,
    "big_bets.rankings.csv": state.rendered.csv,
    "big_bets.md": state.rendered.markdown,
    "big_bets.mmd": state.rendered.mermaid,
    "big_bets.excalidraw": state.rendered.excalidraw,
    "big_bets.svg": state.rendered.svg,
  };
}

function artifactMetadata() {
  return {
    schema_version: REGISTRY_SCHEMA_VERSION,
    artifact_schema_version: ARTIFACT_SCHEMA_VERSION,
    site_schema_version: SITE_SCHEMA_VERSION,
    generator: GENERATOR,
  };
}

function familiesByBigBet(registry) {
  const map = new Map();
  registry.idea_families.forEach((family) => {
    if (!map.has(family.big_bet)) map.set(family.big_bet, []);
    map.get(family.big_bet).push(family);
  });
  map.forEach((families) => families.sort(ideaFamilySort));
  return map;
}

function normalizedEdges(registry) {
  const edges = new Map();
  const labels = new Map();
  registry.big_bets.forEach((bet) => {
    (bet.edge_labels || []).forEach((edge) => {
      labels.set(`${bet.id}\\u0000${edge.target}`, edge.label);
      labels.set(`${edge.target}\\u0000${bet.id}`, edge.label);
    });
    bet.unlocks.forEach((target) => edges.set(`${bet.id}\\u0000${target}`, { from: bet.id, to: target, kind: "unlocks" }));
    bet.depends_on.forEach((source) => {
      const key = `${source}\\u0000${bet.id}`;
      if (!edges.has(key)) edges.set(key, { from: source, to: bet.id, kind: "depends_on" });
    });
  });
  return [...edges.entries()]
    .map(([key, edge]) => {
      const label = labels.get(key);
      return label ? { ...edge, label } : edge;
    })
    .sort((a, b) => `${a.from}:${a.to}`.localeCompare(`${b.from}:${b.to}`));
}

function waves(registry) {
  const grouped = new Map();
  registry.big_bets.forEach((bet) => {
    if (!grouped.has(bet.wave)) grouped.set(bet.wave, []);
    grouped.get(bet.wave).push(bet);
  });
  return [...grouped.entries()].sort((a, b) => a[0] - b[0]).map(([wave, bets]) => [wave, bets.sort(bigBetSort)]);
}

function normalizeBetRanks() {
  waves(state.registry).forEach(([, bets]) => {
    bets.forEach((bet, index) => {
      bet.rank = index + 1;
      bet.priority = priorityForWave(bet.wave);
    });
  });
}

function normalizeFamilyRanks() {
  const grouped = familiesByBigBet(state.registry);
  grouped.forEach((families) => {
    families.forEach((family, index) => {
      family.rank = index + 1;
    });
  });
}

function findBet(id) {
  const bet = state.registry.big_bets.find((item) => item.id === id);
  if (!bet) throw new Error(`Unknown big bet: ${id}`);
  return bet;
}

function findFamily(issue) {
  const family = state.registry.idea_families.find((item) => item.issue === Number(issue));
  if (!family) throw new Error(`Unknown idea family: ${issue}`);
  return family;
}

function objectOrDefault(value, label) {
  if (value == null) return {};
  return objectRequired(value, label);
}

function objectRequired(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object.`);
  return value;
}

function arrayRequired(value, label) {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
  return value;
}

function requiredString(value, label) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} must be a non-empty string.`);
  return value.trim();
}

function optionalString(value) {
  if (value == null) return null;
  if (typeof value !== "string") return String(value).trim() || null;
  return value.trim() || null;
}

function positiveInt(value, label, defaultValue = null) {
  if (value == null && defaultValue != null) return defaultValue;
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer.`);
  return value;
}

function optionalPositiveInt(value, label) {
  if (value == null || value === "") return null;
  return positiveInt(Number(value), label);
}

function identifier(value, label) {
  if (!IDENTIFIER_RE.test(value)) throw new Error(`${label} must match ${IDENTIFIER_RE}.`);
  return value;
}

function priority(value, label) {
  if (!PRIORITY_RE.test(value)) throw new Error(`${label} must look like P0, P1, ...`);
  return value;
}

function status(value, label) {
  if (!STATUS_VALUES.has(value)) throw new Error(`${label} must be one of: ${[...STATUS_VALUES].join(", ")}.`);
  return value;
}

function unlockState(value, label) {
  if (!UNLOCK_STATE_VALUES.has(value)) throw new Error(`${label} must be one of: ${[...UNLOCK_STATE_VALUES].join(", ")}.`);
  return value;
}

function defaultUnlockState(value) {
  return Array.isArray(value.depends_on) && value.depends_on.length ? "locked" : "emerging";
}

function optionalIdentifierArray(value, label) {
  if (value == null) return [];
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
  return value.map((item) => identifier(requiredString(item, label), label));
}

function optionalIdentifier(value, label) {
  const text = optionalString(value);
  return text ? identifier(text, label) : null;
}

function parseLinks(value, label) {
  if (value == null) return [];
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
  return value.map((item, index) => {
    const link = objectRequired(item, `${label}[${index}]`);
    return {
      label: requiredString(link.label, `${label}[${index}].label`),
      url: requiredString(link.url, `${label}[${index}].url`),
      kind: optionalString(link.kind),
    };
  });
}

function parseEdgeLabels(value, label) {
  if (value == null) return [];
  if (!Array.isArray(value) && typeof value === "object") {
    return Object.entries(value).map(([target, edgeLabel]) => ({
      target: identifier(target, `${label}.target`),
      label: requiredString(edgeLabel, `${label}.${target}`),
    }));
  }
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
  return value.map((item, index) => {
    const edge = objectRequired(item, `${label}[${index}]`);
    return {
      target: identifier(requiredString(edge.target, `${label}[${index}].target`), `${label}[${index}].target`),
      label: requiredString(edge.label, `${label}[${index}].label`),
    };
  });
}

function splitIdentifiers(value, label) {
  return String(value || "").split(",").map((item) => item.trim()).filter(Boolean).map((item) => identifier(item, label));
}

function uniqueId(base, existing) {
  let candidate = base;
  let index = 2;
  while (existing.has(candidate)) {
    candidate = `${base}_${index}`;
    index += 1;
  }
  return candidate;
}

function nextRankFor(items) {
  return Math.max(0, ...items.map((item) => Number(item.rank || 0))) + 1;
}

function duplicates(items) {
  const seen = new Set();
  const duplicateSet = new Set();
  items.forEach((item) => {
    if (seen.has(item)) duplicateSet.add(item);
    seen.add(item);
  });
  return [...duplicateSet].sort();
}

function bigBetSort(a, b) {
  return a.wave - b.wave || rankValue(a) - rankValue(b) || a.title.localeCompare(b.title);
}

function ideaFamilySort(a, b) {
  return priorityRank(a.priority) - priorityRank(b.priority) || rankValue(a) - rankValue(b) || a.issue - b.issue;
}

function rankValue(item) {
  return item.rank || 9999;
}

function priorityRank(value) {
  return Number(value.match(PRIORITY_RE)?.[1] || 999);
}

function priorityForWave(wave) {
  return `P${Math.max(0, Number(wave || 1) - 1)}`;
}

function priorityColor(value) {
  return { P0: "#0f766e", P1: "#1d4e89", P2: "#b75b36", P3: "#7c3aed" }[value] || "#5d6c84";
}

function priorityFill(value) {
  return { P0: "#d8f3dc", P1: "#dbeafe", P2: "#fff0bf", P3: "#eee6ff" }[value] || "#eef2f7";
}

function nodeId(value) {
  return `bb_${value.replace(/[^A-Za-z0-9_]/g, "_")}`;
}

function markdownIssueLink(family) {
  const label = familyLabel(family);
  return family.url ? `[${label}](${family.url})` : label;
}

function familyLabel(family) {
  return family.slug || `#${family.issue}`;
}

function familyChipLabel(families) {
  const labels = families.slice(0, 3).map((family) => truncateLabel(familyLabel(family), 18));
  if (families.length > 3) labels.push(`+${families.length - 3}`);
  return labels.join(", ");
}

function truncateLabel(value, maxLength) {
  const text = String(value || "");
  if (text.length <= maxLength) return text;
  return `${text.slice(0, Math.max(0, maxLength - 3))}...`;
}

function linkLabel(value, fallback) {
  const text = String(value || "");
  if (!text) return fallback;
  const issue = text.match(/(?:issues|pull)\\/(\\d+)(?:\\b|$)/);
  if (issue) return `#${issue[1]}`;
  const file = text.split("/").filter(Boolean).at(-1) || fallback;
  if (file.endsWith(".ideas.json")) return "seed";
  if (file.endsWith(".json")) return file.replace(/.*?([a-z0-9_-]+\\.json)$/i, "$1");
  if (fallback === "none") return "human-authored";
  return fallback || file;
}

function shortLinkLabel(link) {
  const label = String(link.label || "").trim();
  if (label) return truncateLabel(label, 18);
  const fallback = link.kind || "ref";
  return truncateLabel(linkLabel(link.url, fallback), 18);
}

function wrapText(text, width, maxLines) {
  const words = text.split(/\\s+/);
  const lines = [];
  let current = "";
  for (const word of words) {
    const candidate = `${current} ${word}`.trim();
    if (candidate.length <= width) {
      current = candidate;
      continue;
    }
    if (current) lines.push(current);
    current = word;
    if (lines.length === maxLines) break;
  }
  if (current && lines.length < maxLines) lines.push(current);
  if (lines.length === maxLines && words.join(" ").length > lines.join(" ").length) {
    lines[lines.length - 1] = `${lines[lines.length - 1].replace(/[.,;:]$/, "")}...`;
  }
  return lines.length ? lines : [text.slice(0, width)];
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll('"', "&quot;");
}

function escapeMermaid(value) {
  return String(value).replaceAll('"', '\\\\"');
}

function escapeMarkdownTable(value) {
  return String(value).replaceAll("|", "\\\\|").replaceAll("\\n", " ");
}

function cssEscape(value) {
  return String(value).replaceAll("\\\\", "\\\\\\\\").replaceAll('"', '\\\\"');
}

function setLog(message, isError) {
  $("validation-log").textContent = message;
  $("validation-log").classList.toggle("error", Boolean(isError));
}

function download(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function highlightSelection() {
  document.querySelectorAll(".selected-row").forEach((row) => row.classList.remove("selected-row"));
  document.querySelectorAll(".selected-card").forEach((node) => node.classList.remove("selected-card"));
  if (!state.selection) return;
  const selector = state.selection.kind === "bet"
    ? `tr[data-kind="bet"][data-id="${cssEscape(state.selection.id)}"]`
    : `tr[data-kind="family"][data-issue="${state.selection.issue}"]`;
  document.querySelector(selector)?.classList.add("selected-row");
  if (state.selection.kind === "bet") {
    document.querySelector(`[data-bet-id="${cssEscape(state.selection.id)}"]`)?.classList.add("selected-card");
  }
}

function stableSeed(value) {
  let hash = 2166136261;
  for (const char of String(value)) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash);
}

function stableId(value) {
  return `bb_${stableSeed(value).toString(16).padStart(8, "0")}`;
}

function formatTimestamp(value) {
  if (!value) return "Not saved yet";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatDate(value) {
  if (!value) return "unknown date";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toISOString().slice(0, 10);
}

function readLocalSnapshots(appId) {
  const text = window.localStorage.getItem(snapshotKey(appId));
  if (!text) return [];
  try {
    const snapshots = JSON.parse(text);
    return Array.isArray(snapshots) ? snapshots : [];
  } catch {
    return [];
  }
}

function storageKey(appId) {
  return `bigbets:${appId}:registry`;
}

function snapshotKey(appId) {
  return `bigbets:${appId}:snapshots`;
}
"""


__all__ = [
    "StaticSiteScaffold",
    "StorageAdapter",
    "list_storage_adapters",
    "site_result_payload",
    "write_static_site",
]
