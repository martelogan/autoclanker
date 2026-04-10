from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from autoclanker.bayes_layer.types import (
    GeneStateRef,
    JsonValue,
    SemanticLevel,
    SurfaceKind,
    ValidationFailure,
)


def _normalize_token(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValidationFailure("Gene and state identifiers must be non-empty strings.")
    return normalized


@dataclass(frozen=True, slots=True)
class GeneDefinition:
    gene_id: str
    states: tuple[str, ...]
    default_state: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    state_descriptions: dict[str, str] | None = None
    state_aliases: dict[str, tuple[str, ...]] | None = None
    surface_kind: SurfaceKind = "runtime_option"
    semantic_level: SemanticLevel = "concrete"
    materializable: bool = True
    code_scopes: tuple[str, ...] = ()
    risk_hints: tuple[str, ...] = ()
    origin: str = "adapter"
    metadata: dict[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class GeneRegistry:
    genes: dict[str, GeneDefinition]

    @classmethod
    def from_serialized_dict(
        cls,
        payload: Mapping[str, object],
    ) -> GeneRegistry:
        mapping: dict[str, tuple[str, ...]] = {}
        defaults: dict[str, str] = {}
        descriptions: dict[str, str] = {}
        aliases: dict[str, tuple[str, ...]] = {}
        state_descriptions: dict[str, dict[str, str]] = {}
        state_aliases: dict[str, dict[str, tuple[str, ...]]] = {}
        surface_kinds: dict[str, SurfaceKind] = {}
        semantic_levels: dict[str, SemanticLevel] = {}
        materializable: dict[str, bool] = {}
        code_scopes: dict[str, tuple[str, ...]] = {}
        risk_hints: dict[str, tuple[str, ...]] = {}
        origins: dict[str, str] = {}
        metadata: dict[str, dict[str, JsonValue]] = {}
        for raw_gene_id, raw_definition in payload.items():
            gene_id = str(raw_gene_id)
            if not isinstance(raw_definition, Mapping):
                raise ValidationFailure(
                    f"Serialized surface entry {gene_id!r} must be an object."
                )
            definition = cast(Mapping[str, object], raw_definition)
            raw_states = definition.get("states")
            if not isinstance(raw_states, list):
                raise ValidationFailure(
                    f"Serialized surface entry {gene_id!r} must define a states list."
                )
            state_items: list[str] = []
            for item in cast(list[object], raw_states):
                if not isinstance(item, str):
                    raise ValidationFailure(
                        f"Serialized surface entry {gene_id!r} contained a non-string state."
                    )
                state_items.append(item)
            mapping[gene_id] = tuple(state_items)
            default_state = definition.get("default_state")
            if not isinstance(default_state, str):
                raise ValidationFailure(
                    f"Serialized surface entry {gene_id!r} must define default_state."
                )
            defaults[gene_id] = default_state
            description = definition.get("description")
            if isinstance(description, str) and description.strip():
                descriptions[gene_id] = description.strip()
            raw_aliases = definition.get("aliases")
            if isinstance(raw_aliases, list):
                alias_items: list[str] = []
                for item in cast(list[object], raw_aliases):
                    if isinstance(item, str) and item.strip():
                        alias_items.append(item.strip())
                aliases[gene_id] = tuple(alias_items)
            raw_state_descriptions = definition.get("state_descriptions")
            if isinstance(raw_state_descriptions, Mapping):
                parsed_state_descriptions: dict[str, str] = {}
                for state_id, text in cast(
                    Mapping[object, object], raw_state_descriptions
                ).items():
                    if isinstance(text, str) and text.strip():
                        parsed_state_descriptions[str(state_id)] = text.strip()
                state_descriptions[gene_id] = parsed_state_descriptions
            raw_state_aliases = definition.get("state_aliases")
            if isinstance(raw_state_aliases, Mapping):
                parsed_state_aliases: dict[str, tuple[str, ...]] = {}
                for state_id, raw_items in cast(
                    Mapping[object, object], raw_state_aliases
                ).items():
                    if isinstance(raw_items, list):
                        state_alias_items: list[str] = []
                        for item in cast(list[object], raw_items):
                            if isinstance(item, str) and item.strip():
                                state_alias_items.append(item.strip())
                        parsed_state_aliases[str(state_id)] = tuple(state_alias_items)
                state_aliases[gene_id] = parsed_state_aliases
            surface_kind = definition.get("surface_kind")
            if isinstance(surface_kind, str):
                surface_kinds[gene_id] = cast(SurfaceKind, surface_kind)
            semantic_level = definition.get("semantic_level")
            if isinstance(semantic_level, str):
                semantic_levels[gene_id] = cast(SemanticLevel, semantic_level)
            materializable_value = definition.get("materializable")
            if isinstance(materializable_value, bool):
                materializable[gene_id] = materializable_value
            raw_code_scopes = definition.get("code_scopes")
            if isinstance(raw_code_scopes, list):
                parsed_code_scopes: list[str] = []
                for item in cast(list[object], raw_code_scopes):
                    if isinstance(item, str) and item.strip():
                        parsed_code_scopes.append(item.strip())
                code_scopes[gene_id] = tuple(parsed_code_scopes)
            raw_risk_hints = definition.get("risk_hints")
            if isinstance(raw_risk_hints, list):
                parsed_risk_hints: list[str] = []
                for item in cast(list[object], raw_risk_hints):
                    if isinstance(item, str) and item.strip():
                        parsed_risk_hints.append(item.strip())
                risk_hints[gene_id] = tuple(parsed_risk_hints)
            origin = definition.get("origin")
            if isinstance(origin, str) and origin.strip():
                origins[gene_id] = origin.strip()
            raw_metadata = definition.get("metadata")
            if isinstance(raw_metadata, Mapping):
                parsed_metadata: dict[str, JsonValue] = {}
                for key, value in cast(
                    Mapping[object, JsonValue], raw_metadata
                ).items():
                    parsed_metadata[str(key)] = value
                metadata[gene_id] = parsed_metadata
        return cls.from_mapping(
            mapping,
            defaults=defaults,
            descriptions=descriptions or None,
            aliases=aliases or None,
            state_descriptions=state_descriptions or None,
            state_aliases=state_aliases or None,
            surface_kinds=surface_kinds or None,
            semantic_levels=semantic_levels or None,
            materializable=materializable or None,
            code_scopes=code_scopes or None,
            risk_hints=risk_hints or None,
            origins=origins or None,
            metadata=metadata or None,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Sequence[str]],
        *,
        defaults: Mapping[str, str] | None = None,
        descriptions: Mapping[str, str] | None = None,
        aliases: Mapping[str, Sequence[str]] | None = None,
        state_descriptions: Mapping[str, Mapping[str, str]] | None = None,
        state_aliases: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
        surface_kinds: Mapping[str, SurfaceKind] | None = None,
        semantic_levels: Mapping[str, SemanticLevel] | None = None,
        materializable: Mapping[str, bool] | None = None,
        code_scopes: Mapping[str, Sequence[str]] | None = None,
        risk_hints: Mapping[str, Sequence[str]] | None = None,
        origins: Mapping[str, str] | None = None,
        metadata: Mapping[str, Mapping[str, JsonValue]] | None = None,
    ) -> GeneRegistry:
        genes: dict[str, GeneDefinition] = {}
        for gene_id, states in mapping.items():
            normalized_gene_id = _normalize_token(gene_id)
            normalized_states = tuple(_normalize_token(state) for state in states)
            if not normalized_states:
                raise ValidationFailure(
                    f"Registry gene {normalized_gene_id!r} must declare at least one state."
                )
            default_state = (
                _normalize_token(defaults[normalized_gene_id])
                if defaults and normalized_gene_id in defaults
                else normalized_states[0]
            )
            if default_state not in normalized_states:
                raise ValidationFailure(
                    f"Default state {default_state!r} is not valid for gene "
                    f"{normalized_gene_id!r}."
                )
            description = (
                _normalize_token(descriptions[normalized_gene_id])
                if descriptions and normalized_gene_id in descriptions
                else ""
            )
            gene_aliases = (
                tuple(_normalize_token(alias) for alias in aliases[normalized_gene_id])
                if aliases and normalized_gene_id in aliases
                else ()
            )
            raw_state_descriptions = (
                state_descriptions[normalized_gene_id]
                if state_descriptions and normalized_gene_id in state_descriptions
                else None
            )
            normalized_state_descriptions = (
                {
                    _normalize_token(state_id): _normalize_token(state_description)
                    for state_id, state_description in raw_state_descriptions.items()
                }
                if raw_state_descriptions is not None
                else None
            )
            raw_state_aliases = (
                state_aliases[normalized_gene_id]
                if state_aliases and normalized_gene_id in state_aliases
                else None
            )
            normalized_state_aliases = (
                {
                    _normalize_token(state_id): tuple(
                        _normalize_token(alias) for alias in alias_items
                    )
                    for state_id, alias_items in raw_state_aliases.items()
                }
                if raw_state_aliases is not None
                else None
            )
            if normalized_state_descriptions is not None:
                unknown_state_ids = set(normalized_state_descriptions) - set(
                    normalized_states
                )
                if unknown_state_ids:
                    joined = ", ".join(sorted(unknown_state_ids))
                    raise ValidationFailure(
                        f"State descriptions referenced unknown states for gene "
                        f"{normalized_gene_id!r}: {joined}"
                    )
            if normalized_state_aliases is not None:
                unknown_state_ids = set(normalized_state_aliases) - set(
                    normalized_states
                )
                if unknown_state_ids:
                    joined = ", ".join(sorted(unknown_state_ids))
                    raise ValidationFailure(
                        f"State aliases referenced unknown states for gene "
                        f"{normalized_gene_id!r}: {joined}"
                    )
            genes[normalized_gene_id] = GeneDefinition(
                gene_id=normalized_gene_id,
                states=normalized_states,
                default_state=default_state,
                description=description,
                aliases=gene_aliases,
                state_descriptions=normalized_state_descriptions,
                state_aliases=normalized_state_aliases,
                surface_kind=(
                    surface_kinds[normalized_gene_id]
                    if surface_kinds and normalized_gene_id in surface_kinds
                    else "runtime_option"
                ),
                semantic_level=(
                    semantic_levels[normalized_gene_id]
                    if semantic_levels and normalized_gene_id in semantic_levels
                    else "concrete"
                ),
                materializable=(
                    materializable[normalized_gene_id]
                    if materializable and normalized_gene_id in materializable
                    else True
                ),
                code_scopes=(
                    tuple(
                        _normalize_token(item)
                        for item in code_scopes[normalized_gene_id]
                    )
                    if code_scopes and normalized_gene_id in code_scopes
                    else ()
                ),
                risk_hints=(
                    tuple(
                        _normalize_token(item)
                        for item in risk_hints[normalized_gene_id]
                    )
                    if risk_hints and normalized_gene_id in risk_hints
                    else ()
                ),
                origin=(
                    _normalize_token(origins[normalized_gene_id])
                    if origins and normalized_gene_id in origins
                    else "adapter"
                ),
                metadata=(
                    dict(metadata[normalized_gene_id])
                    if metadata and normalized_gene_id in metadata
                    else None
                ),
            )
        return cls(genes=genes)

    def known_gene_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.genes))

    def default_genotype(
        self,
        *,
        materializable_only: bool = True,
    ) -> tuple[GeneStateRef, ...]:
        return tuple(
            GeneStateRef(gene_id=gene_id, state_id=definition.default_state)
            for gene_id, definition in sorted(self.genes.items())
            if (not materializable_only) or definition.materializable
        )

    def materializable_gene_ids(self) -> tuple[str, ...]:
        return tuple(
            gene_id
            for gene_id, definition in sorted(self.genes.items())
            if definition.materializable
        )

    def has_ref(self, ref: GeneStateRef) -> bool:
        definition = self.genes.get(ref.gene_id)
        return definition is not None and ref.state_id in definition.states

    def ensure_known_ref(self, ref: GeneStateRef) -> GeneStateRef:
        if not self.has_ref(ref):
            raise ValidationFailure(
                f"Unknown registry reference {ref.gene_id!r}:{ref.state_id!r}."
            )
        return ref

    def canonicalize_ref(self, ref: GeneStateRef) -> GeneStateRef:
        candidate = GeneStateRef(
            gene_id=_normalize_token(ref.gene_id),
            state_id=_normalize_token(ref.state_id),
        )
        return self.ensure_known_ref(candidate)

    def canonicalize_members(
        self,
        members: Sequence[GeneStateRef],
    ) -> tuple[GeneStateRef, ...]:
        normalized = tuple(
            sorted((self.canonicalize_ref(item) for item in members), key=self.sort_key)
        )
        keys = [item.canonical_key for item in normalized]
        if len(keys) != len(set(keys)):
            raise ValidationFailure("Duplicate gene-state references are not allowed.")
        return normalized

    @staticmethod
    def sort_key(ref: GeneStateRef) -> tuple[str, str]:
        return ref.gene_id, ref.state_id

    @staticmethod
    def main_ref(ref: GeneStateRef) -> str:
        return f"main:{ref.gene_id}={ref.state_id}"

    def pair_ref(self, members: Sequence[GeneStateRef]) -> str:
        left, right = self.canonicalize_members(members)
        return f"pair:{left.gene_id}={left.state_id}+{right.gene_id}={right.state_id}"

    def gene_description(self, gene_id: str) -> str:
        return self.genes[gene_id].description

    def gene_aliases(self, gene_id: str) -> tuple[str, ...]:
        return self.genes[gene_id].aliases

    def state_description(self, ref: GeneStateRef) -> str:
        definition = self.genes[ref.gene_id]
        if definition.state_descriptions is None:
            return ""
        return definition.state_descriptions.get(ref.state_id, "")

    def state_aliases(self, ref: GeneStateRef) -> tuple[str, ...]:
        definition = self.genes[ref.gene_id]
        if definition.state_aliases is None:
            return ()
        return definition.state_aliases.get(ref.state_id, ())

    @staticmethod
    def pattern_ref(members: Sequence[GeneStateRef]) -> str:
        ordered = sorted(members, key=GeneRegistry.sort_key)
        return "pattern:" + ",".join(
            f"{member.gene_id}={member.state_id}" for member in ordered
        )

    @staticmethod
    def feasibility_ref(ref: GeneStateRef, failure_mode: str) -> str:
        return f"feasibility:{ref.gene_id}={ref.state_id}#{failure_mode}"

    @staticmethod
    def vram_ref(ref: GeneStateRef) -> str:
        return f"vram:{ref.gene_id}={ref.state_id}"

    def with_overlay(self, overlay: GeneRegistry) -> GeneRegistry:
        merged = dict(self.genes)
        merged.update(overlay.genes)
        return GeneRegistry(genes=merged)

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            gene_id: {
                "states": list(definition.states),
                "default_state": definition.default_state,
                "description": definition.description,
                "aliases": list(definition.aliases),
                "state_descriptions": dict(definition.state_descriptions or {}),
                "state_aliases": {
                    state_id: list(alias_items)
                    for state_id, alias_items in (
                        definition.state_aliases or {}
                    ).items()
                },
                "surface_kind": definition.surface_kind,
                "semantic_level": definition.semantic_level,
                "materializable": definition.materializable,
                "code_scopes": list(definition.code_scopes),
                "risk_hints": list(definition.risk_hints),
                "origin": definition.origin,
                "metadata": dict(definition.metadata or {}),
            }
            for gene_id, definition in sorted(self.genes.items())
        }

    def surface_summary(self) -> dict[str, object]:
        by_kind: dict[str, int] = {}
        by_level: dict[str, int] = {}
        materializable = 0
        for definition in self.genes.values():
            by_kind[definition.surface_kind] = (
                by_kind.get(definition.surface_kind, 0) + 1
            )
            by_level[definition.semantic_level] = (
                by_level.get(definition.semantic_level, 0) + 1
            )
            if definition.materializable:
                materializable += 1
        return {
            "gene_count": len(self.genes),
            "materializable_gene_count": materializable,
            "surface_kind_counts": by_kind,
            "semantic_level_counts": by_level,
        }


def build_fixture_registry() -> GeneRegistry:
    return GeneRegistry.from_mapping(
        {
            "parser.matcher": (
                "matcher_basic",
                "matcher_compiled",
                "matcher_jit",
            ),
            "parser.plan": (
                "plan_default",
                "plan_context_pair",
                "plan_full_scan",
            ),
            "capture.window": ("window_default", "window_wide"),
            "io.chunk": ("chunk_default", "chunk_large"),
            "emit.summary": ("summary_default", "summary_streaming"),
            "search.incident_cluster_pass": (
                "cluster_default",
                "cluster_context_path",
            ),
            "risk.capture_memory_pressure": ("pressure_default", "pressure_high"),
        },
        defaults={
            "parser.matcher": "matcher_basic",
            "parser.plan": "plan_default",
            "capture.window": "window_default",
            "io.chunk": "chunk_default",
            "emit.summary": "summary_default",
            "search.incident_cluster_pass": "cluster_default",
            "risk.capture_memory_pressure": "pressure_default",
        },
        descriptions={
            "parser.matcher": "How the parser matches tokens inside each log line.",
            "parser.plan": "How much cross-line incident context the parser reconstructs.",
            "capture.window": "How many neighboring log lines stay in memory.",
            "io.chunk": "How many lines the parser reads in one batch.",
            "emit.summary": "How the operator-facing incident summary is emitted.",
            "search.incident_cluster_pass": "Higher-level search family for reconstructing adjacent incident clusters.",
            "risk.capture_memory_pressure": "Higher-level risk family for memory pressure caused by wider capture scopes.",
        },
        aliases={
            "parser.matcher": ("matching", "parser matching", "regex mode"),
            "parser.plan": ("context plan", "incident plan", "grouping strategy"),
            "capture.window": ("context window", "capture buffer", "memory window"),
            "io.chunk": ("batch size", "chunk size", "read chunk"),
            "emit.summary": ("summary mode", "output mode", "incident output"),
            "search.incident_cluster_pass": (
                "incident cluster pass",
                "clustered incident search",
            ),
            "risk.capture_memory_pressure": (
                "memory pressure",
                "capture pressure",
                "buffer pressure",
            ),
        },
        state_descriptions={
            "parser.matcher": {
                "matcher_basic": "Simple token splitting for each log line.",
                "matcher_compiled": "Compiled regex matching for repeated log formats.",
                "matcher_jit": "Aggressive regex plan that can be faster but less stable.",
            },
            "parser.plan": {
                "plan_default": "Treat each warning/error line independently.",
                "plan_context_pair": "Pair each incident with its nearest context line.",
                "plan_full_scan": "Scan a wider context neighborhood around every incident.",
            },
            "capture.window": {
                "window_default": "Keep a small context window in memory.",
                "window_wide": "Keep a wide context window in memory.",
            },
            "io.chunk": {
                "chunk_default": "Read a small chunk of log lines at a time.",
                "chunk_large": "Read a larger chunk of log lines at a time.",
            },
            "emit.summary": {
                "summary_default": "Emit a standard end-of-run summary.",
                "summary_streaming": "Stream summary lines as incidents are found.",
            },
            "search.incident_cluster_pass": {
                "cluster_default": "Do not bias toward an incident-cluster search family.",
                "cluster_context_path": "Bias toward the context-pair reconstruction path for adjacent incidents.",
            },
            "risk.capture_memory_pressure": {
                "pressure_default": "Assume ordinary memory pressure.",
                "pressure_high": "Treat wide capture scopes as a likely memory-pressure risk.",
            },
        },
        state_aliases={
            "parser.matcher": {
                "matcher_basic": ("basic parser", "token split", "simple matching"),
                "matcher_compiled": (
                    "compiled regex",
                    "regex",
                    "compiled matcher",
                    "precompiled parser",
                ),
                "matcher_jit": ("jit regex", "aggressive matcher", "fast regex"),
            },
            "parser.plan": {
                "plan_default": (
                    "default plan",
                    "single-line incidents",
                    "plain grouping",
                ),
                "plan_context_pair": (
                    "context pair",
                    "pair errors with context",
                    "nearby context",
                    "context-aware grouping",
                ),
                "plan_full_scan": (
                    "full scan",
                    "scan whole neighborhood",
                    "wide context scan",
                ),
            },
            "capture.window": {
                "window_default": ("small window", "default window", "small buffer"),
                "window_wide": (
                    "wide window",
                    "large buffer",
                    "capture more context",
                    "bigger memory window",
                ),
            },
            "io.chunk": {
                "chunk_default": ("small chunk", "default chunk", "smaller batches"),
                "chunk_large": ("large chunk", "bigger batches", "larger read chunk"),
            },
            "emit.summary": {
                "summary_default": ("normal summary", "final summary"),
                "summary_streaming": (
                    "streaming summary",
                    "emit while parsing",
                    "stream output",
                ),
            },
            "search.incident_cluster_pass": {
                "cluster_default": ("default cluster path",),
                "cluster_context_path": (
                    "incident cluster path",
                    "cluster context path",
                ),
            },
            "risk.capture_memory_pressure": {
                "pressure_default": ("normal memory pressure",),
                "pressure_high": ("high memory pressure", "memory risk"),
            },
        },
        surface_kinds={
            "parser.matcher": "runtime_option",
            "parser.plan": "runtime_option",
            "capture.window": "runtime_option",
            "io.chunk": "runtime_option",
            "emit.summary": "runtime_option",
            "search.incident_cluster_pass": "search_angle",
            "risk.capture_memory_pressure": "risk_family",
        },
        semantic_levels={
            "parser.matcher": "concrete",
            "parser.plan": "concrete",
            "capture.window": "concrete",
            "io.chunk": "concrete",
            "emit.summary": "concrete",
            "search.incident_cluster_pass": "strategy",
            "risk.capture_memory_pressure": "risk",
        },
        materializable={
            "parser.matcher": True,
            "parser.plan": True,
            "capture.window": True,
            "io.chunk": True,
            "emit.summary": True,
            "search.incident_cluster_pass": False,
            "risk.capture_memory_pressure": False,
        },
        code_scopes={
            "parser.matcher": ("log parser", "matching pipeline"),
            "parser.plan": ("incident grouping", "parser planner"),
            "capture.window": ("incident capture", "log buffering"),
            "io.chunk": ("log reader", "batch parser"),
            "emit.summary": ("operator summary", "incident reporting"),
            "search.incident_cluster_pass": ("parser.plan", "emit.summary"),
            "risk.capture_memory_pressure": ("capture.window", "io.chunk"),
        },
        risk_hints={
            "parser.matcher": ("metric_instability",),
            "parser.plan": ("runtime_fail", "metric_instability"),
            "capture.window": ("oom",),
            "io.chunk": ("oom",),
            "emit.summary": ("metric_instability",),
            "search.incident_cluster_pass": ("metric_instability",),
            "risk.capture_memory_pressure": ("oom",),
        },
        metadata={
            "search.incident_cluster_pass": {
                "state_rules": {
                    "cluster_context_path": {
                        "implied_by_all": ["main:parser.plan=plan_context_pair"]
                    }
                }
            },
            "risk.capture_memory_pressure": {
                "state_rules": {
                    "pressure_high": {
                        "implied_by_any": [
                            "main:capture.window=window_wide",
                            "main:io.chunk=chunk_large",
                        ]
                    }
                }
            },
        },
    )
