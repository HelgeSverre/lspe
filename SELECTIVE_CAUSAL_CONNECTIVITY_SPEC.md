# Selective Causal Connectivity Flattening

**Short name:** SCCF

**Protocol version:** 0.1

**Status:** Frozen before SCCF telemetry

**Primary subject:** `mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`

## The question

Global Dynamic Connectivity Flattening (DCF) changed the intended internal
mechanism, but it reached behaviorally meaningful output divergence only after
next-token agreement fell below its competence gate. SCCF tests the most direct
follow-up:

> Did global DCF fail because it flattened useful and fragile connectivity
> modes indiscriminately, and is there a subset that moves association-heavy
> processing while sparing ordinary competence?

SCCF is a causal selectivity experiment. It is not licensed to make a
creativity claim from calibration text.

## Frozen inputs

SCCF reuses only frozen DCF assets:

- the 96-prompt `data/phase3/dynamic_map.jsonl` corpus;
- the fixed 24-token teacher-forced continuations;
- tuning correlation matrices fitted on folds 0 and 2;
- the DCF-selected layers 15 through 22;
- baseline logits from the completed DCF calibration.

Folds 1 and 3 remain untouched until one mask and dose are frozen. All mode
screening, ranking, mask size selection, and dose selection use folds 0 and 2.

## Connectivity modes

For each selected layer, decompose its frozen tuning correlation matrix:

```text
C[l] = V[l] diag(lambda[l]) V[l]^T
```

A mode is identified by `(layer, eigenvalue_rank)`, with rank zero denoting the
smallest eigenvalue. Eigenvalue rank, rather than eigenvector sign, gives a
deterministic identity. A selective transform changes only chosen modes:

```text
T(alpha, M) = V diag(gamma) V^T
gamma[i] = lambda[i]^(-alpha/2) if i is in M, else 1
```

The existing per-head mean and standard-deviation restoration and all DCF
kernel invariants remain mandatory.

## Exact causal screen

Every one of the 256 layer-mode pairs (8 layers × 32 modes) is intervened on
alone at `alpha=0.42`. The screen uses a frozen, category-balanced subset of
tuning folds: the first four prompt IDs in each of five categories after
lexicographic sorting, for 20 prompts total. Narrative is excluded because it
is neither a target nor a competence sentinel. Each screen follows the first
12 tokens of the frozen teacher-forced continuation.

The target group is `analogical` plus `open_association`. The protection group
is `constrained`, `factual`, plus `code`. For every mode record median
token-level output KL and top-1 agreement separately for both groups.

This is called causal because the score comes from an exact forward pass with
only that mode altered, not from correlation with observed activations.

## Selection without threshold fishing

For each mode define:

```text
selectivity = log10((target_KL + 1e-7) / (protection_KL + 1e-7))
```

Modes are eligible for ranking only if target KL is greater than protection KL.
Rank eligible modes by decreasing selectivity, then decreasing target KL, then
layer and eigenvalue rank. Candidate masks are the first 8, 16, 32, and 64
eligible modes. If fewer than eight modes are eligible, stop with
`NO_SELECTIVE_MODES`.

This screen discovers differential causal sensitivity. It does not establish
that a mode improves association, which requires later behavioral evaluation.

## Tuning calibration

Evaluate every candidate mask at alphas `[0.20, 0.30, 0.40, 0.50, 0.60]` on
all 48 tuning prompts and all 24 frozen tokens. A candidate is eligible only if:

- median absolute correlation falls by at least 15%;
- effective rank rises by at least 10%;
- overall median output KL is between `0.005` and `0.08` nats;
- overall top-1 agreement is at least 80%;
- protection-group top-1 agreement is at least 82%;
- target-group median KL is greater than protection-group median KL;
- every numerical, mask, cache, and moment-preservation invariant passes.

Choose lexicographically: highest protection top-1 agreement, then highest
target/protection KL ratio, then fewer modes, then lower alpha. Thresholds are
never relaxed. No candidate means `NO_ELIGIBLE_SELECTIVE_DOSE`.

## Untouched confirmation

Evaluate the single frozen mask and dose once on folds 1 and 3. It passes only
if:

- the DCF mechanism point and bootstrap gates still pass: correlation reduction
  at least 15% (upper 95% bound at most -10%) and effective-rank increase at
  least 10% (lower 95% bound above 5%);
- median output KL is no more than 0.08 nats;
- overall top-1 agreement is at least 80%;
- protection-group top-1 agreement is at least 82%;
- target-group median KL remains greater than protection-group median KL;
- all invariants pass.

Failure stops the experiment. Passing unlocks a separately frozen behavioral
experiment with the DCF random-basis, KL-matched attention-noise, and
entropy-matched temperature controls.

## Required artifacts and stop rules

The run must preserve the protocol and input hashes, complete per-mode screen,
deterministic ranking and masks, every calibration candidate, the untouched
result if reached, source/model hashes, and file checksums. The runner is
resumable after each screened mode and each calibration candidate. Any source,
model, data, mapping, baseline, or checksum mismatch fails closed.

Later behavioral stages remain forbidden unless untouched confirmation passes.
