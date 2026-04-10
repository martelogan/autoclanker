from __future__ import annotations

from autoclanker.bayes_layer.canonicalization import (
    CanonicalizationModel,
    CanonicalizationRequest,
    CanonicalizationSuggestion,
)
from autoclanker.bayes_layer.types import (
    ExpertPriorBelief,
    GeneStateRef,
    GraphDirectiveBelief,
    MainEffectTarget,
)


class AdvancedSkillCanonicalizationModel(CanonicalizationModel):
    name = "advanced-skill-stub"

    def canonicalize(
        self,
        request: CanonicalizationRequest,
    ) -> tuple[CanonicalizationSuggestion, ...]:
        suggestions: list[CanonicalizationSuggestion] = []
        for idea in request.ideas:
            lowered = idea.rationale.lower()
            if "interaction screen" in lowered or "keep together" in lowered:
                suggestions.append(
                    CanonicalizationSuggestion(
                        belief_id=idea.belief_id,
                        belief=GraphDirectiveBelief(
                            id=idea.belief_id,
                            confidence_level=idea.confidence_level,
                            evidence_sources=idea.evidence_sources,
                            rationale=idea.rationale,
                            members=(
                                GeneStateRef(
                                    gene_id="parser.matcher",
                                    state_id="matcher_compiled",
                                ),
                                GeneStateRef(
                                    gene_id="parser.plan",
                                    state_id="plan_context_pair",
                                ),
                            ),
                            directive="screen_include",
                            strength=2,
                        ),
                        source="llm",
                        summary="Promoted the rough idea into a graph directive.",
                        matched_evidence=("interaction screen", "keep together"),
                        confidence_score=0.79,
                    )
                )
                continue
            if "prior" in lowered or "compiled matching" in lowered:
                suggestions.append(
                    CanonicalizationSuggestion(
                        belief_id=idea.belief_id,
                        belief=ExpertPriorBelief(
                            id=idea.belief_id,
                            confidence_level=idea.confidence_level,
                            evidence_sources=idea.evidence_sources,
                            rationale=idea.rationale,
                            target=MainEffectTarget(
                                target_kind="main_effect",
                                gene=GeneStateRef(
                                    gene_id="parser.matcher",
                                    state_id="matcher_compiled",
                                ),
                            ),
                            prior_family="normal",
                            mean=0.45,
                            scale=0.35,
                        ),
                        source="llm",
                        summary="Promoted the rough idea into a conservative expert prior.",
                        matched_evidence=("compiled matching", "prior"),
                        confidence_score=0.83,
                    )
                )
                continue
            suggestions.append(
                CanonicalizationSuggestion(
                    belief_id=idea.belief_id,
                    belief=None,
                    source="llm",
                    summary="The advanced skill stub could not promote this idea confidently.",
                    needs_review=True,
                    confidence_score=0.22,
                )
            )
        return tuple(suggestions)


def build_autoclanker_canonicalization_model() -> CanonicalizationModel:
    return AdvancedSkillCanonicalizationModel()
