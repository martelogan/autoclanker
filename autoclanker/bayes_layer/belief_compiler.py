from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import cast

from autoclanker.bayes_layer.belief_io import validate_payload_against_schema
from autoclanker.bayes_layer.config import BayesLayerConfig, load_bayes_layer_config
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    Belief,
    BeliefPreview,
    CompiledPriorBundle,
    CompiledPriorItem,
    CompiledPriorPreview,
    ConstraintBelief,
    EraState,
    ExpertPriorBelief,
    ExpertTarget,
    FeasibilityTarget,
    GraphDirectiveBelief,
    IdeaBelief,
    MainEffectTarget,
    PairEffectTarget,
    PreferenceBelief,
    PriorDecay,
    PriorSpec,
    ProposalBelief,
    RelationBelief,
    ValidatedBeliefBatch,
    ValidationFailure,
    to_json_value,
)


@dataclass(frozen=True, slots=True)
class _BeliefCompilation:
    preview: BeliefPreview
    prior_specs: tuple[PriorSpec, ...]


def _variance_scale(config: BayesLayerConfig, confidence_level: int) -> float:
    return config.beliefs.confidence_level_to_variance_scale[confidence_level]


def _effect_mean(config: BayesLayerConfig, effect_strength: int) -> float:
    return config.beliefs.effect_strength_to_prior_mean[effect_strength]


def _relation_mean(config: BayesLayerConfig, strength: int) -> float:
    return config.beliefs.relation_strength_to_pair_mean[strength]


def _default_decay(
    *,
    config: BayesLayerConfig,
    target_kind: str,
) -> PriorDecay:
    cross_era_transfer = config.beliefs.prior_decay.cross_era_main_effect_transfer
    if target_kind in {"pair_effect", "linkage_hint"}:
        cross_era_transfer = config.beliefs.prior_decay.cross_era_pair_effect_transfer
    return PriorDecay(
        per_eval_multiplier=config.beliefs.prior_decay.per_eval_multiplier,
        cross_era_transfer=cross_era_transfer,
    )


def _risk_mean(severity: int) -> float:
    return severity * 0.6


def _specs_for_items(
    belief_id: str,
    items: Sequence[CompiledPriorItem],
) -> tuple[PriorSpec, ...]:
    return tuple(PriorSpec(source_belief_id=belief_id, item=item) for item in items)


def _target_ref_for_expert(
    registry: GeneRegistry,
    target: ExpertTarget,
) -> tuple[str, str]:
    if isinstance(target, FeasibilityTarget):
        ref = registry.canonicalize_ref(target.gene)
        return (
            target.target_kind,
            registry.feasibility_ref(ref, target.failure_mode),
        )
    if isinstance(target, PairEffectTarget):
        members = registry.canonicalize_members(target.members)
        return target.target_kind, registry.pair_ref(members)
    if isinstance(target, MainEffectTarget):
        gene = registry.canonicalize_ref(target.gene)
        return target.target_kind, registry.main_ref(gene)
    gene = registry.canonicalize_ref(target.gene)
    return target.target_kind, registry.vram_ref(gene)
    raise ValidationFailure(f"Unsupported expert target type: {type(target)!r}")


def _compile_proposal(
    belief: ProposalBelief,
) -> _BeliefCompilation:
    warnings = [
        "proposal_text is preserved as metadata and requires manual canonicalization."
    ]
    metadata = None if belief.context is None else belief.context.metadata
    if metadata is not None:
        suggested_options = metadata.get("suggested_options")
        if isinstance(suggested_options, list) and suggested_options:
            joined = ", ".join(str(item) for item in suggested_options[:3])
            warnings.append(f"Suggested options from registry matching: {joined}.")
    preview = BeliefPreview(
        belief_id=belief.id,
        compile_status="metadata_only",
        compiled_items=(),
        warnings=tuple(warnings),
        influence_summary=(),
    )
    return _BeliefCompilation(preview=preview, prior_specs=())


def _compile_idea(
    belief: IdeaBelief,
    *,
    registry: GeneRegistry,
    config: BayesLayerConfig,
) -> _BeliefCompilation:
    gene = registry.canonicalize_ref(belief.gene)
    scale = _variance_scale(config, belief.confidence_level)
    items: list[CompiledPriorItem] = [
        CompiledPriorItem(
            target_kind="main_effect",
            target_ref=registry.main_ref(gene),
            prior_family="normal",
            mean=_effect_mean(config, belief.effect_strength),
            scale=scale,
            decay=_default_decay(config=config, target_kind="main_effect"),
            notes=("compiled from idea belief",),
        )
    ]
    if belief.complexity_delta:
        items.append(
            CompiledPriorItem(
                target_kind="vram_effect",
                target_ref=registry.vram_ref(gene),
                prior_family="normal",
                mean=0.25 * float(belief.complexity_delta),
                scale=scale,
                decay=_default_decay(config=config, target_kind="vram_effect"),
                notes=("compiled from complexity_delta hint",),
            )
        )
    if belief.risk is not None:
        for failure_mode, severity in belief.risk.nonzero_items():
            items.append(
                CompiledPriorItem(
                    target_kind="feasibility_logit",
                    target_ref=registry.feasibility_ref(gene, failure_mode),
                    prior_family="logit_normal",
                    mean=_risk_mean(severity),
                    scale=scale,
                    decay=_default_decay(
                        config=config, target_kind="feasibility_logit"
                    ),
                    notes=("compiled from risk hint",),
                )
            )
    warnings: list[str] = []
    metadata = None if belief.context is None else belief.context.metadata
    if metadata is not None:
        canonicalization_mode = metadata.get("canonicalization_mode")
        if canonicalization_mode == "heuristic_single":
            warnings.append(
                "Auto-canonicalized from a high-level idea using registry descriptions."
            )
    preview = BeliefPreview(
        belief_id=belief.id,
        compile_status="compiled",
        compiled_items=tuple(items),
        warnings=tuple(warnings),
        influence_summary=tuple(
            f"{belief.id} influences {item.target_ref}" for item in items
        ),
    )
    return _BeliefCompilation(
        preview=preview, prior_specs=_specs_for_items(belief.id, items)
    )


def _compile_relation(
    belief: RelationBelief,
    *,
    registry: GeneRegistry,
    config: BayesLayerConfig,
) -> _BeliefCompilation:
    members = registry.canonicalize_members(belief.members)
    base_mean = _relation_mean(config, belief.strength)
    sign = 1.0 if belief.relation in {"synergy", "dependency"} else -1.0
    joint_bonus = 0.0
    if belief.joint_effect_strength is not None:
        joint_bonus = 0.5 * _effect_mean(config, belief.joint_effect_strength)
    scale = _variance_scale(config, belief.confidence_level)
    items: list[CompiledPriorItem] = []
    warnings: list[str] = []
    for left, right in combinations(members, 2):
        items.append(
            CompiledPriorItem(
                target_kind="pair_effect",
                target_ref=registry.pair_ref((left, right)),
                prior_family="normal",
                mean=(sign * base_mean) + joint_bonus,
                scale=scale,
                decay=_default_decay(config=config, target_kind="pair_effect"),
                notes=(f"compiled from relation:{belief.relation}",),
            )
        )
    if len(members) > 2:
        warnings.append("Expanded higher-order relation into pairwise priors.")
    metadata = None if belief.context is None else belief.context.metadata
    if metadata is not None:
        canonicalization_mode = metadata.get("canonicalization_mode")
        if canonicalization_mode == "heuristic_relation":
            warnings.append(
                "Auto-canonicalized a high-level relation using registry descriptions."
            )
    if belief.relation == "dependency":
        items.append(
            CompiledPriorItem(
                target_kind="screening_hint",
                target_ref=registry.pattern_ref(members),
                prior_family="screening_hint",
                mean=float(belief.strength),
                scale=scale,
                decay=_default_decay(config=config, target_kind="screening_hint"),
                notes=("favor joint materialization",),
            )
        )
    if belief.relation == "exclusion":
        items.append(
            CompiledPriorItem(
                target_kind="mask",
                target_ref=registry.pattern_ref(members),
                prior_family="hard_mask",
                mean=1.0,
                scale=0.01,
                notes=("exclusion relation compiled into hard mask",),
            )
        )
    preview = BeliefPreview(
        belief_id=belief.id,
        compile_status="compiled",
        compiled_items=tuple(items),
        warnings=tuple(warnings),
        influence_summary=tuple(
            f"{belief.id} influences {item.target_ref}" for item in items
        ),
    )
    return _BeliefCompilation(
        preview=preview, prior_specs=_specs_for_items(belief.id, items)
    )


def _compile_preference(
    belief: PreferenceBelief,
    *,
    registry: GeneRegistry,
    config: BayesLayerConfig,
) -> _BeliefCompilation:
    left = registry.canonicalize_members(belief.left_pattern.members)
    right = registry.canonicalize_members(belief.right_pattern.members)
    if left == right:
        raise ValidationFailure(
            "Preference patterns collapsed to the same canonical genotype."
        )
    direction = {"left": 1.0, "right": -1.0, "tie": 0.0}[belief.preference]
    item = CompiledPriorItem(
        target_kind="preference_pseudo_observation",
        target_ref=f"{registry.pattern_ref(left)}>{registry.pattern_ref(right)}",
        prior_family="paired_preference",
        mean=direction * float(belief.strength),
        scale=_variance_scale(config, belief.confidence_level),
        notes=("compiled from pairwise preference",),
    )
    preview = BeliefPreview(
        belief_id=belief.id,
        compile_status="compiled",
        compiled_items=(item,),
        warnings=(),
        influence_summary=(f"{belief.id} influences {item.target_ref}",),
    )
    return _BeliefCompilation(
        preview=preview, prior_specs=_specs_for_items(belief.id, (item,))
    )


def _compile_constraint(
    belief: ConstraintBelief,
    *,
    registry: GeneRegistry,
    config: BayesLayerConfig,
) -> _BeliefCompilation:
    scope = registry.canonicalize_members(belief.scope)
    scale = _variance_scale(config, belief.confidence_level)
    items: list[CompiledPriorItem] = []
    pattern_ref = registry.pattern_ref(scope)
    if belief.constraint_type == "hard_exclude":
        items.append(
            CompiledPriorItem(
                target_kind="mask",
                target_ref=pattern_ref,
                prior_family="hard_mask",
                mean=1.0,
                scale=0.01,
                notes=("compiled from hard_exclude constraint",),
            )
        )
    elif belief.constraint_type == "soft_avoid":
        items.append(
            CompiledPriorItem(
                target_kind="screening_hint",
                target_ref=pattern_ref,
                prior_family="screening_hint",
                mean=-float(belief.severity),
                scale=scale,
                decay=_default_decay(config=config, target_kind="screening_hint"),
                notes=("compiled from soft_avoid constraint",),
            )
        )
    elif belief.constraint_type == "require":
        items.append(
            CompiledPriorItem(
                target_kind="screening_hint",
                target_ref=pattern_ref,
                prior_family="screening_hint",
                mean=float(belief.severity),
                scale=scale,
                decay=_default_decay(config=config, target_kind="screening_hint"),
                notes=("compiled from require constraint",),
            )
        )
    else:
        for ref in scope:
            items.append(
                CompiledPriorItem(
                    target_kind="vram_effect",
                    target_ref=registry.vram_ref(ref),
                    prior_family="normal",
                    mean=0.35 * float(belief.severity),
                    scale=scale,
                    decay=_default_decay(config=config, target_kind="vram_effect"),
                    notes=("compiled from budget_cap constraint",),
                )
            )
    preview = BeliefPreview(
        belief_id=belief.id,
        compile_status="compiled",
        compiled_items=tuple(items),
        warnings=(),
    )
    return _BeliefCompilation(
        preview=preview, prior_specs=_specs_for_items(belief.id, items)
    )


def _compile_expert_prior(
    belief: ExpertPriorBelief,
    *,
    registry: GeneRegistry,
    config: BayesLayerConfig,
) -> _BeliefCompilation:
    target_kind, target_ref = _target_ref_for_expert(registry, belief.target)
    decay = _default_decay(config=config, target_kind=target_kind)
    if belief.decay_override is not None:
        decay = PriorDecay(
            per_eval_multiplier=(
                belief.decay_override.per_eval_multiplier
                if belief.decay_override.per_eval_multiplier is not None
                else decay.per_eval_multiplier
            ),
            cross_era_transfer=(
                belief.decay_override.cross_era_transfer
                if belief.decay_override.cross_era_transfer is not None
                else decay.cross_era_transfer
            ),
        )
    notes = ["compiled from expert prior"]
    if belief.observation_weight is not None:
        notes.append(f"observation_weight={belief.observation_weight:g}")
    item = CompiledPriorItem(
        target_kind=target_kind,
        target_ref=target_ref,
        prior_family=belief.prior_family,
        mean=belief.mean,
        scale=belief.scale,
        observation_weight=belief.observation_weight,
        decay=decay,
        notes=tuple(notes),
    )
    preview = BeliefPreview(
        belief_id=belief.id,
        compile_status="compiled",
        compiled_items=(item,),
        warnings=(),
    )
    return _BeliefCompilation(
        preview=preview, prior_specs=_specs_for_items(belief.id, (item,))
    )


def _compile_graph_directive(
    belief: GraphDirectiveBelief,
    *,
    registry: GeneRegistry,
    config: BayesLayerConfig,
) -> _BeliefCompilation:
    members = registry.canonicalize_members(belief.members)
    target_kind = "screening_hint"
    sign = 1.0
    if belief.directive == "screen_exclude":
        sign = -1.0
    if belief.directive == "linkage_positive":
        target_kind = "linkage_hint"
    if belief.directive == "linkage_negative":
        target_kind = "linkage_hint"
        sign = -1.0
    item = CompiledPriorItem(
        target_kind=target_kind,
        target_ref=registry.pair_ref(members),
        prior_family="graph_directive",
        mean=sign * float(belief.strength),
        scale=_variance_scale(config, belief.confidence_level),
        decay=_default_decay(config=config, target_kind=target_kind),
        notes=(f"compiled from graph directive:{belief.directive}",),
    )
    preview = BeliefPreview(
        belief_id=belief.id,
        compile_status="compiled",
        compiled_items=(item,),
        warnings=(),
    )
    return _BeliefCompilation(
        preview=preview, prior_specs=_specs_for_items(belief.id, (item,))
    )


def _compile_single_belief(
    belief: Belief,
    *,
    registry: GeneRegistry,
    config: BayesLayerConfig,
) -> _BeliefCompilation:
    if isinstance(belief, ProposalBelief):
        return _compile_proposal(belief)
    if isinstance(belief, IdeaBelief):
        return _compile_idea(belief, registry=registry, config=config)
    if isinstance(belief, RelationBelief):
        return _compile_relation(belief, registry=registry, config=config)
    if isinstance(belief, PreferenceBelief):
        return _compile_preference(belief, registry=registry, config=config)
    if isinstance(belief, ConstraintBelief):
        return _compile_constraint(belief, registry=registry, config=config)
    if isinstance(belief, ExpertPriorBelief):
        return _compile_expert_prior(belief, registry=registry, config=config)
    return _compile_graph_directive(belief, registry=registry, config=config)


def _compile_all(
    *,
    beliefs: Sequence[Belief],
    registry: GeneRegistry,
    config: BayesLayerConfig,
) -> tuple[_BeliefCompilation, ...]:
    results: list[_BeliefCompilation] = []
    for belief in beliefs:
        try:
            results.append(
                _compile_single_belief(belief, registry=registry, config=config)
            )
        except ValidationFailure as exc:
            results.append(
                _BeliefCompilation(
                    preview=BeliefPreview(
                        belief_id=belief.id,
                        compile_status="rejected",
                        compiled_items=(),
                        warnings=(str(exc),),
                    ),
                    prior_specs=(),
                )
            )
    return tuple(results)


def preview_compiled_beliefs(
    beliefs: ValidatedBeliefBatch,
    registry: GeneRegistry,
    era_state: EraState,
    *,
    config: BayesLayerConfig | None = None,
) -> CompiledPriorPreview:
    active_config = config or load_bayes_layer_config()
    compilations = _compile_all(
        beliefs=beliefs.beliefs,
        registry=registry,
        config=active_config,
    )
    preview = CompiledPriorPreview(
        era_id=era_state.era_id,
        belief_previews=tuple(compilation.preview for compilation in compilations),
    )
    validate_payload_against_schema(
        cast(dict[str, object], to_json_value(preview)),
        "compiled_prior_preview.schema.json",
    )
    return preview


def compile_beliefs(
    beliefs: ValidatedBeliefBatch,
    registry: GeneRegistry,
    era_state: EraState,
    *,
    config: BayesLayerConfig | None = None,
) -> CompiledPriorBundle:
    active_config = config or load_bayes_layer_config()
    compilations = _compile_all(
        beliefs=beliefs.beliefs,
        registry=registry,
        config=active_config,
    )
    all_specs = tuple(
        spec for compilation in compilations for spec in compilation.prior_specs
    )

    def filter_specs(target_kind: str) -> tuple[PriorSpec, ...]:
        return tuple(spec for spec in all_specs if spec.item.target_kind == target_kind)

    return CompiledPriorBundle(
        era_id=era_state.era_id,
        main_effect_priors=filter_specs("main_effect"),
        pair_priors=filter_specs("pair_effect"),
        feasibility_priors=filter_specs("feasibility_logit"),
        vram_priors=filter_specs("vram_effect"),
        hard_masks=filter_specs("mask"),
        preference_observations=filter_specs("preference_pseudo_observation"),
        candidate_generation_hints=filter_specs("screening_hint"),
        linkage_hints=filter_specs("linkage_hint"),
        belief_previews=tuple(compilation.preview for compilation in compilations),
    )
