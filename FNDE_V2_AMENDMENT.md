# FNDE v2 Mapping Amendment

**Frozen before v2 telemetry**

## Why a second mapping attempt is justified

FNDE v1 stopped correctly because its head communities were unstable. That run
used only one generated position from each greedy and sampled continuation and
pooled six very different task families into one dependence estimate.

V2 changes the microscope, not the passing score. It does not reinterpret or
replace the v1 `MECHANISM_NOT_ACHIEVED` result.

## Frozen changes

- Use a fresh `network_map_v2` corpus of 240 prompts not present in any v1 or
  Phase 2 split.
- Balance exactly 40 prompts across constrained, factual, narrative,
  analogical, code, and control tasks.
- Include 60 paraphrase pairs and 30 unrelated-prompt negative-control pairs.
- Generate two fixed greedy tokens and two fixed sampled tokens per prompt.
  This yields 960 eligible generated-position observations, four times v1.
- Observe all 36 layers and all 1,152 attention heads.
- Preserve full 2,560-dimensional residual contributions and 16-bin relative
  attention patterns. No random feature projection is permitted.
- Construct one CKA graph per task family, then average the six equally weighted
  adjacency matrices. This prevents the largest-variance task family from
  defining the universal map.
- Use four deterministic, category-stratified prompt folds. Select graph density
  and community count using folds 0 and 2 only. Evaluate the fixed choice once
  using folds 1 and 3.
- Candidate densities are `[0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
  0.075, 0.10, 0.15, 0.20]`; community counts are three through eight.
- Selection is lexicographic: tuning ARI, retained-node count, lower density,
  lower community count.

## Gates that do not change

- held-out adjusted Rand index must be at least `0.70`;
- median bootstrap assignment probability must be at least `0.80`;
- paraphrase stability must exceed unrelated-pair stability;
- communities must not be explained solely by layer or shared-KV family;
- at least three stable communities must exist;
- at least two layers must contain heads from multiple communities;
- modularity must exceed the degree-preserving null's 95th percentile.

The held-out fold is not eligible for candidate selection. If the selected
candidate misses any gate, FNDE stops again. The threshold will not be lowered.

If all mapping gates pass, community assignments and the donor graph are frozen
before causal screening. Passing the map does not authorize CCAD by itself:
exact causal eligibility and every later gate in
[NETWORK_DESEGREGATION_SPEC.md](NETWORK_DESEGREGATION_SPEC.md) still apply.
