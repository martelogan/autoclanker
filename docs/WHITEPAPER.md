# WHITEPAPER.md


## CLI-first implementation note

The research-backed architecture in this pack is intended to ship first as a **library plus thin CLI**, not as a dashboard or interactive GUI. That choice is about implementation reliability and future adapterability rather than about the Bayesian model itself.

## A Bayesian Guidance Layer for Autoresearch-Style Evaluation Loops with Structured Human Beliefs

## Executive summary

This project adds a Bayesian guidance layer on top of an expensive black-box evaluation loop such as autoresearch or cEvolve. The layer is **not** a replacement for the outer loop. Instead, it provides:

1. a structured way for humans and LLMs to express beliefs,
2. a compiler from those beliefs into explicit priors,
3. an era-local statistical model over utility and feasibility,
4. a posterior interaction graph over ideas / genes / states,
5. a query policy that asks humans only high-value questions,
6. a principled commit policy that is more robust than `best observed score wins`.

## Core position

The best-supported architecture is a **hybrid** [1], [2], [3], [4]:

- LLMs and humans provide ideas, prior beliefs, rationales, pairwise preferences, and targeted corrections.
- A statistical Bayesian layer performs uncertainty-aware scoring, feasibility modeling, information-seeking, and commit logic.
- The outer evolutionary loop remains in place.

This is more defensible than `let the LLM do everything` for three reasons.

1. Recent hybrid BO work repeatedly uses LLMs for priors, initialization, preference shaping, or early exploration while retaining a separate surrogate for calibrated uncertainty [1], [3], [4].
2. Human-in-the-loop BO literature strongly favors prior forms that are intuitive to users and robust when those priors are wrong [5], [6], [7].
3. RLHF and BO-from-human-feedback research both support targeted pairwise / preference querying rather than asking users for dense numeric supervision everywhere [8].

## Why not a pure natural-language belief card?

A purely free-form card is too lossy and too ambiguous for implementation. It leaves unclear:

- which idea or gene is being referenced,
- whether `higher learning rate` means x1.1, x1.5, or x2.0,
- whether a `risk` is compile-time, runtime, OOM, timeout, or metric instability,
- whether a synergy is a soft preference, a dependency, or an exclusion.

The right interface is therefore **dual-lane and typed** [5], [6], [7], [9]:

- typed and schema-validated fields for anything the optimizer must treat deterministically,
- optional natural-language rationale fields for nuance and evidence,
- pairwise preference inputs for cases where comparison is easier than absolute scoring,
- optional free-form idea proposals that are compiled into structured candidate genes before use,
- optional expert direct priors for users who want declarative control over optimizer features.

## Why the dual-lane API is a strong fit

The literature on prior elicitation repeatedly finds that experts are better at expressing beliefs about observable outcomes, comparisons, and structured scenarios than about raw model parameters [9], [10]. ColaBO and PrBO show that BO benefits when priors are supplied in more intuitive forms over promising regions or likely optimizer structure instead of only over abstract function priors [5], [6]. DynMeanBO adds a key operational pattern: prior influence should decay over time so the system can recover if expert guidance is misleading [7].

For natural-language interaction specifically, LILO shows that language can be valuable when it is **compiled** into quantitative utility information while preserving BO’s uncertainty machinery, rather than replacing BO entirely [2]. LLINBO makes a similar point from the reliability side: LLM-only optimization lacks explicit surrogate modeling and calibrated uncertainty, so a hybrid statistical layer remains important [3].

## Resulting UX principles

The human-facing system should [2], [5], [6], [7], [8], [9], [10]:

- anchor all structured fields to canonical IDs and bounded enums,
- ask for observable outcomes or pairwise comparisons when possible,
- allow free-form rationale, but never free-form control fields,
- support runtime interventions,
- ask only a few high-value questions per era,
- show the compiled structured belief back to the human before using it,
- offer an expert lane for direct priors without forcing it on normal users.

## Research-grounded design summary

The chosen implementation direction is therefore:

- a typed belief API with both ergonomic and expert lanes,
- an explicit canonicalization boundary where free text must become typed beliefs or remain metadata-only,
- compiled prior preview / round-trip before execution,
- era-local Bayesian utility and feasibility models,
- decayed human priors,
- bounded active querying,
- hybrid LLM + BO behavior instead of LLM-only control,
- provider-backed canonicalization that remains inspectable rather than hidden,
- repo-native, testable implementation inside the project repository.

That combination is intentionally conservative in the right places: it borrows robust patterns from prior-guided BO, user-guided BO, and prior elicitation, while using newer LLM-in-the-loop results as design pressure rather than as the sole foundation [1]–[10].

Implementation note:

- in this repo, billed real-model-provider validation is kept separate from the required
  self-contained gate so the Bayesian engine can stay reproducible and testable even when
  model credentials are absent;
- the expected common advanced-authoring workflow is rough ideas → canonicalization →
  preview → assistant-guided refinement into typed `expert_prior` or `graph_directive`
  declarations.

## References

[1] T. Liu, N. Astorga, N. Seedat, and M. van der Schaar, “Large Language Models to Enhance Bayesian Optimization,” in *International Conference on Learning Representations (ICLR)*, 2024.

[2] K. Kobalczyk, Z. J. Lin, B. Letham, Z. Zhao, M. Balandat, and E. Bakshy, “LILO: Bayesian Optimization with Interactive Natural Language Feedback,” *OpenReview*, submitted to ICLR 2026, 2025.

[3] C.-Y. Chang, M. Azvar, C. Okwudire, and R. Al Kontar, “LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization,” *OpenReview*, submitted to ICLR 2026, 2025.

[4] Y. Zeng, N. Maus, H. T. Jones, J. Tao, F. Wan, M. Der Torossian Torres, C. de la Fuente-Nunez, R. Marcus, O. Bastani, and J. R. Gardner, “Scaling Multi-Task Bayesian Optimization with Large Language Models,” in *International Conference on Learning Representations (ICLR)*, 2026.

[5] C. Hvarfner, F. Hutter, and L. Nardi, “A General Framework for User-Guided Bayesian Optimization,” in *International Conference on Learning Representations (ICLR)*, 2024.

[6] A. Souza, M. Lindauer, F. Hutter, and L. Nardi, “Prior-guided Bayesian Optimization,” in *International Conference on Learning Representations (ICLR)*, 2021.

[7] C. Qu, M. Liu, J. Lan, S. Dong, and Z. Liu, “Incorporating Expert Priors into Bayesian Optimization via Dynamic Mean Decay,” in *International Conference on Learning Representations (ICLR)*, 2026.

[8] K. Ji, J. He, and Q. Gu, “Reinforcement Learning from Human Feedback with Active Queries,” *arXiv preprint arXiv:2402.09401*, 2024.

[9] P. Mikkola, O. A. Martin, S. Chandramouli, M. Hartmann, O. Abril Pla, O. Thomas, H. Pesonen, J. Corander, A. Vehtari, S. Kaski, P.-C. Bürkner, and A. Klami, “Prior Knowledge Elicitation: The Past, Present, and Future,” *Bayesian Analysis*, vol. 19, no. 4, pp. 1135–1168, 2024.

[10] F. Bockting and P.-C. Bürkner, “elicito: A Python Package for Expert Prior Elicitation,” *arXiv preprint arXiv:2506.16830*, 2025.
