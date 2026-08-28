# Dynamic Connectivity Flattening Experiment

**Short name:** DCF

**Protocol version:** 0.1

**Status:** Frozen before DCF telemetry

**Primary subject:** `mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`

## The new question

The two FNDE maps showed that static attention-head communities are not a
stable enough substrate. DCF drops communities entirely and asks a narrower,
more direct question:

> If we transiently flatten the model's ordinary head-to-head attention
> correlations while it generates, can we create a controlled, reversible
> desynchronized regime without merely damaging its output?

This is closer to the motivating observation that psilocybin changes patterns
of synchronization and functional connectivity across scales. It is still a
computational analogy, not a biological simulation.

## What counts as a component and a relationship

One component is one query head in one Transformer layer. At generated token
`t`, its state is the standardized pre-softmax attention-logit row over all
available key positions:

```text
u[l,h,t,k] = (s[l,h,t,k] - mean_k(s)) / max(std_k(s), epsilon)
```

The relationship matrix for a layer is the regularized correlation matrix
`C[l]` of these standardized rows, accumulated over prompts, generated tokens,
and key positions. Unlike FNDE, DCF does not cluster `C` or assign semantic
identities to heads.

Only autoregressive decode steps with one query and at least eight unmasked keys
are eligible. Prompt-prefill attention is observed but never intervened on.

## Fresh mapping data

Before telemetry, create and hash `data/phase3/dynamic_map.jsonl` with:

- 96 fresh prompts, 16 in each of constrained, factual, narrative, analogical,
  code, and open-association task families;
- no exact or near duplicate from any earlier LSPE or FNDE split;
- one frozen greedy continuation of 24 tokens per prompt;
- deterministic prompt-family-stratified assignment to four folds;
- paraphrase twins, if present, kept in the same fold.

Folds 0 and 2 fit every transform and select eligible layer windows. Folds 1
and 3 remain untouched until exactly one candidate is frozen.

## Mapping statistic

For every layer, estimate a shrinkage correlation matrix on tuning folds. The
shrinkage target is identity and the coefficient is selected analytically from
tuning observations; it is never selected using held-out outcomes.

Mapping feasibility requires all of the following:

- every retained layer has at least 1,000 eligible head/key observations;
- the tuning and held-out correlation matrices have upper-triangle Pearson
  correlation at least `0.70` in at least 24 of 36 layers;
- at least one contiguous eight-layer window has median matrix correlation at
  least `0.75` and no layer below `0.60`;
- median absolute off-diagonal correlation in that window exceeds `0.02`, so
  there is meaningful synchrony to flatten;
- zero-intervention observation leaves logits within absolute tolerance
  `1e-5` and preserves greedy top-1 tokens exactly.

The eligible window is selected lexicographically using tuning folds only:
highest median split correlation, then highest median absolute off-diagonal
correlation, then earliest start layer. Windows are exactly eight contiguous
layers. Failure stops DCF before intervention calibration.

## Intervention

For a fitted positive-definite correlation matrix
`C = V diag(lambda) V^T`, define:

```text
T(alpha) = V diag(lambda^(-alpha / 2)) V^T
U' = T(alpha) U
```

`alpha=0` is sham and `alpha=1` is full whitening. After mixing across heads,
each head's attention-logit row is independently restored to its original mean
and standard deviation before the causal mask is restored and softmax runs.
The transform is fixed for the entire run and never learns from generated
outcomes.

This operation may reduce ordinary synchrony, but it is not assumed to do so.
The measured mechanism gate decides.

## Mandatory kernel invariants

- `alpha=0` is identical to the wrapped sham within `1e-6`;
- wrapping and unwrapping restores the original attention modules;
- no masked or future key becomes reachable;
- every finite output row preserves its pre-intervention mean and standard
  deviation within `1e-5`;
- the transform is symmetric, finite, deterministic, and content-hashed;
- no model parameter or cache value is modified in place;
- cached and uncached sham logits agree with their corresponding ordinary
  paths within the frozen runtime tolerance;
- non-finite input, output, eigenvalue, or telemetry is a hard failure.

## Calibration candidates

Test `alpha` values `[0.05, 0.10, 0.20, 0.35, 0.50, 0.70]` on the tuning folds
only. Calibration uses frozen teacher-forced 24-token continuations so every
candidate sees identical histories.

For each candidate record:

- median change in absolute off-diagonal correlation;
- median change in correlation-matrix effective rank;
- matrix distance from sham;
- attention KL and JS divergence;
- output KL and JS divergence;
- top-1 agreement and top-10 overlap;
- entropy, invalid-mask, zero-variance, and non-finite counts.

Select the lowest `alpha` satisfying the tuning mechanism band:

- absolute off-diagonal correlation falls by at least 15%;
- effective rank rises by at least 10%;
- median teacher-forced output KL is between `0.005` and `0.08` nats;
- top-1 agreement is at least 80%;
- no invariant violation occurs.

No candidate means `NO_ELIGIBLE_DOSE`; thresholds are not relaxed.

## Untouched mechanism confirmation

The single frozen layer window and dose are evaluated once on folds 1 and 3.
DCF reaches behavioral generation only if prompt-clustered bootstrap intervals
support all of:

- at least 15% reduction in absolute off-diagonal correlation, with the upper
  95% bound at or below `-10%`;
- at least 10% increase in effective rank, with the lower 95% bound above 5%;
- median output KL at most `0.08` nats;
- top-1 agreement at least 80%;
- zero numerical, mask, cache, or weight-integrity violations.

This is the primary scientific gate. Interesting text cannot override it.

## Controls after the mechanism gate

Every later prompt must run paired under:

| Condition | Purpose |
| --- | --- |
| `baseline` | Ordinary model |
| `sham` | Wrapped kernel with `alpha=0` |
| `dcf` | Frozen covariance-flattening transform |
| `random_basis` | Same eigenvalues in a frozen random orthogonal basis |
| `attn_white` | Independent attention-logit noise matched on output KL |
| `temp_match` | Output temperature matched on sampling entropy |

`random_basis` and `attn_white` are independently calibrated to DCF's output
KL. A control that cannot be matched invalidates the behavioral candidate.

## Behavioral pilot and confirmation

If and only if the untouched mechanism gate passes, use fresh pilot and
confirmation splits with the existing validity, competence, degeneration, and
semantic-diversity machinery. Add relational novelty scoring before making a
creativity claim.

Primary contrast: DCF minus entropy-matched temperature on valid semantic
diversity. DCF must also retain competence better than KL-matched attention
noise. Prompt is the clustered unit and every condition is paired within
prompt. A positive claim requires a prompt-clustered 95% interval above zero
and every competence/degeneration non-inferiority gate.

## Stop rules

Stop and preserve artifacts if mapping feasibility, sham equivalence,
intervention invariants, dose calibration, untouched mechanism confirmation,
control matching, or data/source/model hashes fail. Later stages are forbidden
after a failed gate.

## Required artifacts

- immutable data lock with cross-project leakage audit;
- fitted tuning correlation matrices and transform hashes;
- tuning-only window and dose decision;
- untouched-fold mechanism report;
- per-token mechanism and output-divergence telemetry;
- source, model, data, environment, and artifact checksums;
- machine-readable terminal status and plain-language report;
- exact commands for resume, verification, and report regeneration.

## Scientific lineage

The biological motivation is the loss of network segregation and widespread
desynchronization reported by [Siegel et al., Nature
2024](https://www.nature.com/articles/s41586-024-07624-5). Exact and attribution
patching motivate causal follow-up if DCF achieves its mechanism gate; see
[Kramár et al., AtP* 2024](https://arxiv.org/abs/2403.00745). Neither source
implies that Transformer attention heads are brain regions or that whitening
attention logits implements a drug.
