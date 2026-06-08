const issueSeedSchemaVersion = "autoclanker.issue-seed.v1";
const runContractSchemaVersion = "autoclanker.run-contract.v1";

const sampleSeed = {
  title: "Explore lower-cost pipeline execution",
  goal: "Find lower-cost data pipeline execution plans by using benchmark and evidence artifacts to compare concrete lanes under a fixed eval contract.",
  target_repo: "owner/repo",
  label: "agent-loop-idea-family",
  era_id: "era_pipeline_optimization_v1",
  session_id: "pipeline_optimization_seed",
  run_intensity: "mega",
  canonicalization_mode: "deterministic",
  ideas: [
    "Batch repeated source reads into one planned load.",
    "Move persisted data shape closer to the runtime processing layout.",
    "Avoid repeated decode and shape-conversion work on fields not used by downstream stages.",
  ],
  constraints: [
    "Keep the eval surface fixed during lane comparison.",
    "Use evidence artifacts as search inputs, not as proof by themselves.",
    "Record proposal or blocker state before stopping.",
  ],
  artifacts: [
    {
      name: "benchmark-snapshot",
      kind: "benchmark_snapshot",
      url: "https://example.invalid/artifacts/benchmark-snapshot.json",
      expected_path: "artifacts/benchmark-snapshot.json",
    },
    {
      name: "evidence-graph",
      kind: "clankergraph",
      url: "https://example.invalid/artifacts/evidence.clankergraph.json",
      expected_path: "graphs/evidence.clankergraph.json",
    },
  ],
  evidence_notes:
    "Start by reading the benchmark snapshot, clankergraph evidence, profiler notes, and prior run summaries. Convert those into explicit candidate lanes before measuring.",
  big_bet: "minimum_cost_pipeline",
  priority: "P0",
  next_action:
    "Build the smallest measured minimum-cost pipeline ledger before implementation.",
};

const requestRenderingSeed = {
  title: "Explore lower-cost request rendering",
  goal: "Reduce request rendering cost while preserving response output and correctness under a fixed eval contract.",
  target_repo: "owner/repo",
  label: "agent-loop-idea-family",
  era_id: "era_request_rendering_v1",
  session_id: "request_rendering_seed",
  run_intensity: "mega",
  canonicalization_mode: "deterministic",
  ideas: [
    "Batch repeated render-time data lookups across components in one planned load.",
    "Precompute a compact view model in the shape consumed by the renderer.",
    "Skip decode and transformation work for fields not referenced by the selected route.",
  ],
  constraints: [
    "Preserve byte-equivalent or explicitly approved response output changes.",
    "Keep route, cache, and error-handling semantics unchanged unless a candidate explicitly measures that change.",
    "Use render benchmarks and trace evidence as search inputs, not as proof by themselves.",
  ],
  artifacts: [
    {
      name: "render-benchmark-snapshot",
      kind: "benchmark_snapshot",
      url: "https://example.invalid/artifacts/render-benchmark-snapshot.json",
      expected_path: "artifacts/render-benchmark-snapshot.json",
    },
    {
      name: "render-evidence-graph",
      kind: "clankergraph",
      url: "https://example.invalid/artifacts/render-evidence.clankergraph.json",
      expected_path: "graphs/render-evidence.clankergraph.json",
    },
  ],
  evidence_notes:
    "Start by reading the render benchmark snapshot, trace summaries, clankergraph evidence, profiler notes, and prior run summaries. Convert repeated work, data-shape mismatch, and unused-field signals into explicit candidate lanes before measuring.",
  big_bet: "minimum_cost_request_rendering",
  priority: "P0",
  next_action: "Build the smallest measured request-rendering cost ledger before implementation.",
};

const storageKey = "autoclanker.issue-seeder.demo.v1";
const $ = (id) => document.getElementById(id);

function asArray(value) {
  if (Array.isArray(value)) return value.filter((item) => String(item || "").trim());
  if (typeof value === "string") {
    return value
      .split(/\n+/)
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return [];
}

function slug(value) {
  return String(value || "optimization-seed")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);
}

function fenced(language, body) {
  return `\`\`\`${language}\n${String(body).trim()}\n\`\`\``;
}

function jsonString(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function escapeMarkdownCell(value) {
  return String(value).replaceAll("|", "\\|");
}

function shellArg(value) {
  const raw = String(value);
  if (/^[A-Za-z0-9_./:=+-]+$/.test(raw)) return raw;
  return `'${raw.replaceAll("'", "'\\''")}'`;
}

function artifactsFor(seed) {
  const artifactObjects = Array.isArray(seed.artifacts)
    ? seed.artifacts.flatMap((artifact, index) => {
        if (typeof artifact === "string") {
          const url = artifact.trim();
          return url ? [{ name: `artifact-${index + 1}`, kind: "artifact", url }] : [];
        }
        if (!artifact || typeof artifact !== "object") return [];
        return [
          {
            name: String(artifact.name || `artifact-${index + 1}`),
            kind: String(artifact.kind || "artifact"),
            url: String(artifact.url || ""),
            ...(artifact.expected_path
              ? { expected_path: String(artifact.expected_path) }
              : {}),
            ...(artifact.digest ? { digest: String(artifact.digest) } : {}),
          },
        ].filter((item) => item.url);
      })
    : [];
  return [
    ...artifactObjects,
    ...(seed.benchmark_snapshot
      ? [
          {
            name: "benchmark-snapshot",
            kind: "benchmark_snapshot",
            url: String(seed.benchmark_snapshot),
          },
        ]
      : []),
    ...(seed.corpus_snapshot
      ? [
          {
            name: "corpus-snapshot",
            kind: "corpus_snapshot",
            url: String(seed.corpus_snapshot),
          },
        ]
      : []),
  ];
}

function buildBundle(seed) {
  const title = String(seed.title || "Optimization seed").trim();
  const goal = String(seed.goal || "Improve the target metric without reducing correctness.").trim();
  const targetRepo = String(seed.target_repo || seed.repo || "owner/repo").trim();
  const label = String(seed.label || "agent-loop-idea-family").trim();
  const intensity = String(seed.run_intensity || seed.intensity || "mega").trim();
  const canonicalizationMode = String(seed.canonicalization_mode || "deterministic").trim();
  const ideas = asArray(seed.ideas || seed.lanes);
  const effectiveIdeas = ideas.length ? ideas : ["Infer the first candidate lane from evidence."];
  const constraints = asArray(seed.constraints);
  const artifacts = artifactsFor(seed);
  const seedSlug = slug(title || goal);
  const sessionContext = {
    ...(seed.era_id ? { era_id: String(seed.era_id) } : {}),
    ...(seed.session_id ? { session_id: String(seed.session_id) } : {}),
  };
  const autoclankerIdeas = {
    ...(Object.keys(sessionContext).length ? { session_context: sessionContext } : {}),
    ideas: effectiveIdeas.map((idea) => ({
      idea,
      confidence: 2,
      evidence_sources: artifacts.length ? ["benchmark"] : ["intuition"],
    })),
  };
  const artifactManifest = {
    schema_version: issueSeedSchemaVersion,
    seed_slug: seedSlug,
    target_repo: targetRepo,
    title,
    issue_label: label,
    artifacts,
    expected_workspace_files: [
      "autoclanker.ideas.json",
      "artifact-manifest.json",
      "run-contract.json",
      "lane-ledger.md",
      "pi.prompt.txt",
      "headless-command.sh",
      "host-adapter-contract.md",
    ],
    ...(seed.issue_number ? { issue_number: Number(seed.issue_number) } : {}),
    ...(seed.big_bet ? { big_bet: String(seed.big_bet) } : {}),
    ...(seed.priority ? { priority: String(seed.priority) } : {}),
  };
  const runContract = {
    schema_version: runContractSchemaVersion,
    goal,
    run_intensity: intensity,
    acceptance_gates: [
      "fixed_eval_surface_preserved",
      "evidence_artifacts_ingested_before_candidate_selection",
      "multi_lane_measurements_bound_to_candidate_identity",
      "posterior_fit_or_explicit_blocker_recorded",
      "proposal_or_blocker_recorded_before_exit",
    ],
    evidence_policy:
      "Artifact references are exploratory input signals. Promotion requires measured eval evidence under the locked eval contract.",
    promotion_policy:
      "Promote independently reviewable proposals only after measured keep/reject/blocker decisions are recorded for relevant lanes.",
    stop_conditions: [
      "one_or_more_proposals_ready",
      "all_plausible_lanes_rejected_or_blocked",
      "true_hard_blocker_recorded",
    ],
    evidence_intake_checklist: [
      "materialize_or_read_artifact_manifest_references",
      "summarize_clankergraph_benchmark_corpus_and_prior_run_signals",
      "turn_evidence_into_explicit_candidate_lanes_before_measurement",
      "record_lane_ledger_state_before_first_eval",
      "bind_every_eval_result_to_candidate_identity",
    ],
    required_session_outputs: [
      "updated lane-ledger.md",
      "posterior or explicit blocker summary",
      "measured keep/reject/blocker decision per attempted lane",
      "draft proposal links for independently reviewable candidates",
    ],
    ...(constraints.length ? { constraints } : {}),
    ...(seed.adapter_config_path
      ? { adapter_config_path: String(seed.adapter_config_path) }
      : {}),
    ...(seed.canonicalization_mode
      ? { canonicalization_mode: String(seed.canonicalization_mode) }
      : {}),
    ...(seed.canonicalization_model
      ? { canonicalization_model: String(seed.canonicalization_model) }
      : {}),
  };
  const laneRows = effectiveIdeas.map(
    (idea, index) =>
      `| lane-${index + 1} | active | ${escapeMarkdownCell(idea)} | pending first measurement |`,
  );
  const laneLedger = `# Lane Ledger

| Lane | Status | Hypothesis | Latest decision |
| --- | --- | --- | --- |
${laneRows.join("\n")}

Next action: ${seed.next_action || "Run the strongest evidence-backed lane first."}

Update this before first measurement, after each keep/reject/blocker decision,
and before stopping.
`;
  const piPrompt = `Use pi-autoclanker to run this seeded optimization issue.

Goal: ${goal}

Read autoclanker.ideas.json, artifact-manifest.json, run-contract.json,
lane-ledger.md, clankergraph files, benchmark snapshots, and corpus snapshots
before choosing candidate lanes. Start or resume with /autoclanker run
${intensity === "deep" || intensity === "mega" ? "--overnight " : ""}--ideas-input autoclanker.ideas.json. The slash command prepares or resumes the workspace;
read and execute its returned handoffPrompt.

During execution, keep candidate lanes explicit, preserve the fixed eval
surface, bind every multi-lane eval ingest to the measured candidate, use
evidence artifacts as search inputs, fit/suggest between measurements, and
continue until each valuable lane has a measured keep/reject/blocker decision
or an independently reviewable proposal is ready.`;
  const adapterArg = seed.adapter_config_path
    ? ` \\\n  --adapter-config ${shellArg(seed.adapter_config_path)}`
    : "";
  const applyAdapterArg = seed.adapter_config_path
    ? ` \\\n#   --adapter-config ${shellArg(seed.adapter_config_path)}`
    : "";
  const modelArg = seed.canonicalization_model
    ? ` \\\n  --canonicalization-model ${shellArg(seed.canonicalization_model)}`
    : "";
  const headlessCommand = `AUTOCLANKER_SESSION_ID="\${AUTOCLANKER_SESSION_ID:-${seed.session_id || "seeded-run"}}"
AUTOCLANKER_ERA_ID="\${AUTOCLANKER_ERA_ID:-${seed.era_id || "seeded-era"}}"

autoclanker session init \\
  --beliefs-input autoclanker.ideas.json \\
  --session-id "$AUTOCLANKER_SESSION_ID" \\
  --era-id "$AUTOCLANKER_ERA_ID" \\
  --canonicalization-mode ${canonicalizationMode}${modelArg}${adapterArg} > autoclanker.init.json

# Then apply the preview digest from autoclanker.init.json before evals:
# autoclanker session apply-beliefs --session-id "$AUTOCLANKER_SESSION_ID" \\
#   --preview-digest "<preview_digest>"${applyAdapterArg}

pi-autoclanker command run ${intensity === "deep" || intensity === "mega" ? "--overnight " : ""}\\
  --workspace "$PWD" \\
  --ideas-input autoclanker.ideas.json > autoclanker.run.json

# Give the supervising agent the handoffPrompt from autoclanker.run.json.
# The command prepares or resumes state; it is not the full autonomous run.`;
  const hostAdapter = `# Optional Host Adapter Contract

The default issue seeder is static and local-first. A hosted deployment may add
an adapter, but must keep secrets out of browser-local seed JSON.

Recommended optional host methods:

- \`loadSeeds({ appId })\`
- \`saveSeed({ appId, seed })\`
- \`uploadArtifact({ appId, name, contentType, body })\`
- \`fetchIssue({ repo, issue, url })\`
- \`createIssue({ repo, title, body, labels })\`
- \`canonicalizeIdeas({ seed, artifacts })\`

LLM provider keys belong behind canonicalizeIdeas or the CLI environment,
not in the static browser app. Artifact uploads should return immutable URLs or
object keys that can be copied into the generated issue body.`;
  const artifactLines = artifacts.length
    ? artifacts
        .map((artifact) => `- ${artifact.name}: ${artifact.url} (${artifact.kind || "artifact"})`)
        .join("\n")
    : "- No external artifacts declared yet.";
  const laneLines = ideas.length
    ? ideas.map((idea) => `- active: ${idea}`).join("\n")
    : "- Seed lanes should be inferred during intake.";
  const constraintLines = constraints.length
    ? constraints.map((constraint) => `- ${constraint}`).join("\n")
    : "- Preserve correctness, rollback safety, and the locked eval surface.";
  const bigbetsBlock =
    seed.big_bet || seed.priority || seed.next_action
      ? `
<!-- bigbets:idea-family
slug: ${seedSlug}
big_bet: ${seed.big_bet || "unmapped"}
priority: ${seed.priority || "P1"}
status: active
role: ideas-lane
artifact: autoclanker.ideas.json
next_action: ${seed.next_action || "Run the strongest evidence-backed lane first."}
-->
`
      : "";
  const issueBody = `## Start Here

This issue seeds a long-running optimization exploration for:

${goal}

Use the artifact references and run contract below as the current source of
truth. Start or resume the workspace, execute the returned autoclanker handoff
prompt, compare multiple lanes when practical, and post draft proposals or
blockers as they become independently reviewable.

<details>
<summary>Local Pi Kickoff</summary>

${fenced("text", piPrompt)}

</details>

<details>
<summary>Headless CLI Kickoff</summary>

${fenced("bash", headlessCommand)}

</details>

<details>
<summary>Seed Artifacts</summary>

${artifactLines}

${fenced("json", jsonString(artifactManifest))}

</details>

<details>
<summary>Seed Lanes</summary>

${laneLines}

</details>

<details>
<summary>Constraints</summary>

${constraintLines}

</details>

<details>
<summary>Run Contract</summary>

${fenced("json", jsonString(runContract))}

</details>

<details>
<summary>Lane Ledger</summary>

${fenced("markdown", laneLedger)}

</details>

<details>
<summary>Optional Host Adapter Contract</summary>

${fenced("markdown", hostAdapter)}

</details>

## Evidence Notes

${seed.evidence_notes || "No additional evidence notes yet."}

## Target

- Repo: ${targetRepo}
- Suggested title: ${title}
- Suggested label: ${label}
${bigbetsBlock}`;
  return {
    artifactManifest,
    autoclankerIdeas,
    headlessCommand,
    hostAdapter,
    issueBody,
    laneLedger,
    piPrompt,
    runContract,
  };
}

function render(seed) {
  const bundle = buildBundle(seed);
  $("issueBody").value = bundle.issueBody;
  $("ideasJson").value = jsonString(bundle.autoclankerIdeas);
  $("artifactManifestJson").value = jsonString(bundle.artifactManifest);
  $("runContractJson").value = jsonString(bundle.runContract);
  $("laneLedger").value = bundle.laneLedger;
  $("piPrompt").value = bundle.piPrompt;
  $("headlessCommand").value = bundle.headlessCommand;
  $("hostAdapter").value = bundle.hostAdapter;
  $("status").textContent = "valid";
}

function refresh() {
  try {
    const seed = JSON.parse($("seedInput").value);
    localStorage.setItem(storageKey, JSON.stringify(seed, null, 2));
    render(seed);
  } catch (error) {
    $("status").textContent = "invalid JSON";
  }
}

function reset() {
  useSeed(sampleSeed);
}

function useSeed(seed) {
  $("seedInput").value = `${JSON.stringify(seed, null, 2)}\n`;
  refresh();
}

$("seedInput").value =
  localStorage.getItem(storageKey) || `${JSON.stringify(sampleSeed, null, 2)}\n`;
$("seedInput").addEventListener("input", refresh);
$("formatSeed").addEventListener("click", () => {
  const seed = JSON.parse($("seedInput").value);
  $("seedInput").value = `${JSON.stringify(seed, null, 2)}\n`;
  refresh();
});
$("resetSeed").addEventListener("click", reset);
$("loadPipelineSeed").addEventListener("click", () => useSeed(sampleSeed));
$("loadRequestRenderingSeed").addEventListener("click", () => useSeed(requestRenderingSeed));
$("downloadSeed").addEventListener("click", () => {
  const blob = new Blob([$("seedInput").value], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${slug(JSON.parse($("seedInput").value).title)}.seed.json`;
  link.click();
  URL.revokeObjectURL(url);
});
for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    await navigator.clipboard.writeText($(button.dataset.copy).value);
    const original = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => {
      button.textContent = original;
    }, 900);
  });
}
refresh();
