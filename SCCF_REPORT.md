# I Found the LLM's Eight Dominant Attention Pathways

**Selective Causal Connectivity Flattening (SCCF)**

**Registered status: `MECHANISM_PASS`**

Global DCF could desynchronize Qwen3's attention, but it hit the damage line as
soon as the output moved enough to count. SCCF asked whether the problem was
flattening everything at once.

The answer is a qualified **yes**. A frozen 16-mode intervention passed every
registered tuning and untouched-fold mechanism gate. This is the first network
experiment in the project to reach an active, reproducible held-out regime.
It is not yet evidence of greater creativity, and the category breakdown shows
that “competence-safe” needs a stricter definition before behavioral testing.

## What SCCF did

The experiment decomposed the stable DCF correlation matrix in layers 15–22
into 32 eigenmodes per layer. It intervened on all 256 layer-mode pairs one at a
time, using only tuning folds, and measured which modes changed analogical/open
association predictions more than constrained/factual/code predictions.

The strongest result was unusually orderly: the top-ranked mode was the largest
eigenvalue mode in every one of the eight layers. In plain language, each layer's
dominant “heads moving together” direction mattered across the whole window.

The preregistered ranking produced masks of 8, 16, 32, and 64 modes. Twenty
mask/dose combinations were then tested without changing any threshold.

## The one eligible tuning candidate

Only the 16-mode mask at `alpha=0.50` passed all registered calibration gates.

| Tuning measure | Result | Gate |
| --- | ---: | ---: |
| Correlation change | -47.0% | at most -15% |
| Effective-rank change | +174.7% | at least +10% |
| Median output KL | 0.00824 | 0.005–0.08 |
| Overall top-1 agreement | 80.56% | at least 80% |
| Protection-group top-1 | 84.38% | at least 82% |
| Target median KL | 0.00679 | above protection |
| Protection median KL | 0.000000814 | below target |

No neighboring candidate passed. The weaker doses did not reach the active KL
floor; stronger or broader interventions crossed a top-1 gate.

## It reproduced on untouched prompts

The mask and dose were frozen before folds 1 and 3 were evaluated. Every
registered held-out gate passed:

| Untouched measure | Result |
| --- | ---: |
| Correlation change | -46.9% |
| Correlation-change 95% interval | -47.6% to -45.9% |
| Effective-rank change | +174.5% |
| Effective-rank 95% interval | +171.4% to +177.6% |
| Median output KL | 0.01505 |
| Overall top-1 agreement | 81.25% |
| Protection-group top-1 | 84.38% |
| Target-group top-1 | 82.29% |
| Target median KL | 0.00800 |
| Protection median KL | 0.00000138 |

There were no non-finite values, zero-variance failures, or moment-preservation
violations. Maximum mean and scale errors remained within `1e-5`.

## The important catch

The broad registered groups hide a more complicated category pattern:

| Held-out category | Mean top-1 agreement | Median prompt KL |
| --- | ---: | ---: |
| Analogical | 91.15% | 0.00000774 |
| Code | 90.63% | 0.000000081 |
| Factual | 89.58% | 0.000000301 |
| Constrained formatting | 72.92% | 0.1284 |
| Narrative | 69.79% | 0.1934 |
| Open association | 73.44% | 0.1343 |

So SCCF did not isolate “association” as a single faculty. It found modes that
strongly affect open-ended language and format-sensitive generation while
largely sparing analogy, factual recall, and code. The registered protection
average passed because code and factual tasks were extremely stable even though
constrained formatting was not.

That does not invalidate the preregistered mechanism pass. It does invalidate a
casual summary such as “we changed creativity without damage.” Before any such
claim, the next protocol must require a floor in every protected category—not
only the pooled protection average—and must compare generated behavior with
random-basis, KL-matched attention-noise, and entropy-matched temperature
controls.

## Decision

SCCF reached `MECHANISM_PASS`. Behavioral testing is technically unlocked by
the frozen protocol, but it should proceed under a new preregistration with:

- per-category competence gates;
- separate analogy and open-association outcomes;
- constrained formatting retained as a fragility sentinel;
- paired baseline, sham, SCCF, random-basis, attention-noise, and temperature
  controls;
- fresh pilot and untouched confirmation prompts.

The scientifically defensible result is narrower and more interesting than a
generic success: **Qwen3 has a small, reproducible set of attention-correlation
modes whose transient flattening selectively changes some kinds of generation
far more than factual and code prediction.** What those changes do to complete
answers remains untested.

## Integrity

The frozen source commit is `5332e36`. The local run is
`runs/sccf-qwen3-5332e36`. It contains all 256 mode screens, deterministic mode
ranking, all 20 calibration candidates, the frozen selection, untouched-fold
telemetry, provenance lock, and verified SHA-256 checksums.

See [SELECTIVE_CAUSAL_CONNECTIVITY_SPEC.md](SELECTIVE_CAUSAL_CONNECTIVITY_SPEC.md)
for the protocol.
