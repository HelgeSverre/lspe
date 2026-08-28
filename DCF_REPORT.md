# I Desynchronized an LLM's Attention—and Found the Damage Line

**Dynamic Connectivity Flattening (DCF)**

**Final status: `NO_ELIGIBLE_DOSE`**

The earlier experiments tried to identify static “networks” of attention heads.
Those maps fell apart on new prompts. DCF removed the clustering step and went
after the relationship itself: the moment-to-moment correlation between all 32
attention heads in each Transformer layer.

This time the mechanism was real.

## What changed

DCF measured standardized pre-softmax attention rows over 24-token trajectories
from 96 fresh prompts. It then fitted a reversible fractional-whitening
transform that flattened ordinary head-to-head correlations during generation
while preserving every head's attention-logit mean and variance.

The model was Qwen3 4B at the same pinned revision as FNDE. The run covered all
36 layers, 2,304 generated-token steps, and 39,048 head/key observations per
fold and layer. The wrapped zero-dose path was exactly identical to baseline:
maximum logit error `0.0` and identical top-1 output.

## The map finally passed

Unlike the static communities, the dynamic correlation matrices reproduced
almost perfectly across independent prompts.

| Mapping measure | Result | Gate |
| --- | ---: | ---: |
| Stable tuning layers | 36 / 36 | at least 24 |
| Stable untouched layers | 36 / 36 | at least 24 |
| Selected window | layers 15–22 | eight contiguous layers |
| Window tuning similarity | 0.9998 | at least 0.75 |
| Window untouched similarity | 0.9998 | at least 0.75 |
| Window tuning synchrony | 0.731 | above 0.02 |
| Sham maximum logit error | 0.0 | at most 0.00001 |

The candidate was selected using folds 0 and 2 only. Folds 1 and 3 confirmed
the relationship map but were not used to select the window.

## The intervention worked

Even small doses moved the intended internal statistic. For example,
`alpha=0.20` reduced median absolute head correlation by 22.5% and increased
correlation-matrix effective rank by 87.4%, while retaining 90.7% top-1
agreement.

The transform was numerically clean at every tested dose: no non-finite values,
no zero-variance rows, and maximum mean/scale preservation errors below the
registered `1e-5` tolerance.

So this experiment achieved something the previous attempts did not: it
created a measurable, reversible desynchronized computational regime.

## Where it failed

The registered calibration required a dose to produce meaningful output
divergence—median KL between `0.005` and `0.08` nats—while preserving at least
80% top-1 agreement.

The original coarse grid found no dose satisfying both. A separately frozen
denser-grid amendment then examined the unresolved bracket without changing
any threshold:

| Alpha | Correlation change | Effective-rank change | Output KL | Top-1 agreement | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.38 | -47.4% | +249.8% | 0.00202 | 81.86% | KL too low |
| 0.40 | -50.4% | +273.8% | 0.00216 | 80.30% | KL too low |
| 0.42 | -53.2% | +297.7% | 0.00505 | 79.51% | competence too low |
| 0.44 | -56.0% | +323.0% | 0.00584 | 77.95% | competence too low |
| 0.46 | -59.0% | +348.6% | 0.00894 | 76.30% | competence too low |
| 0.48 | -61.7% | +373.9% | 0.00816 | 75.35% | competence too low |

The crossover is unusually clear. At `0.40`, the model remains just competent
enough but the aggregate output movement is below the active-dose floor. At
`0.42`, the output movement qualifies but top-1 retention falls just below the
80% gate.

## Decision

The terminal status is `NO_ELIGIBLE_DOSE`.

The untouched intervention confirmation was not run, nor were behavioral
generation, creativity comparisons, or noise controls. Those stages require a
candidate that passes calibration first.

This does not collapse back to “we only damaged the model.” DCF demonstrably
changed the intended relationship statistic at mild doses while leaving most
next-token decisions intact. What failed was the stronger claim that this
particular global whitening transform has a safe, behaviorally active window.

The next technical refinement, if pursued, should be selective rather than
stronger: flatten only the correlation modes or layer-token contexts that exert
causal influence on associative tasks, instead of whitening every mode across
an eight-layer window.

## Integrity

The mapping run is `runs/dcf-map-qwen3-6bdfd20`. The v1 and v2 calibration runs
are `runs/dcf-calibration-qwen3-12b8e1b` and
`runs/dcf-calibration-v2-qwen3-040bdd8`. Data, source, model revision, mapping,
protocol, numerical invariants, and artifact checksums are recorded. Large
logit caches remain local and ignored by Git; the machine-readable closeout and
this report are tracked.
