# I Tried Harder to Desegregate an LLM's Brain

**Functional Network Desegregation Experiment (FNDE), Phase 2**

**Final status: `MECHANISM_NOT_ACHIEVED`**

The second experiment asked a sharper version of the mushroom question.

Instead of shoving a random vector into one hidden layer, could we first map the
model's functional networks, then temporarily blur the boundaries between them?
That would be a much closer computational analogy to the idea that psilocybin
changes communication among brain networks.

The answer arrived before the drug was administered: **even after a much larger,
stricter second mapping attempt, the network map was not stable or balanced
enough to intervene on honestly.**

## The stronger second attempt

The first mapping result left open a fair objection: perhaps 200 prompts and
one greedy plus one sampled position per prompt were simply too weak. FNDE v2
therefore started over with a frozen amendment and fresh data:

- 240 new prompts, 40 in each of six task families;
- 60 paraphrase pairs and 30 unrelated-prompt controls;
- two greedy and two sampled generated positions per prompt;
- 960 observations across all 1,152 attention heads;
- one graph per task family, combined with equal weight;
- four folds, with folds 0/2 selecting the graph and folds 1/3 untouched;
- 11 graph densities and three through eight communities tested without using
  held-out outcomes to choose among them.

The run produced roughly 22 GB of telemetry. Its selected candidate used 7.5%
graph density and three communities.

| V2 measure | Result | Gate |
| --- | ---: | ---: |
| Tuning-fold ARI | 0.808 | selection only |
| Held-out evaluable-head coverage | 94.3% | at least 80% |
| Held-out ARI | **0.548** | **at least 0.700** |
| Final community sizes | **803 / 18 / 10** | each at least 5% |
| Smallest community | **1.2%** | **at least 5%** |

This is a clearer failure than v1. The tuning score proves the machinery can
find an apparently repeatable partition. The untouched prompts show that the
chosen partition does not generalize, while the 803/18/10 split shows that it
is mostly one giant cluster with two tiny leftovers—not three plausible
functional systems.

## The first attempt

The subject was the pinned 4-bit Qwen3 4B model used for the v1 architecture
replication. The mapping stage covered all 36 decoder layers and all 1,152
attention heads.

For 200 frozen prompts, the harness collected both greedy and sampled generated
positions. For every head and matched position it stored:

- the head's full 2,560-dimensional contribution to the residual stream;
- a 16-bin relative attention pattern;
- output RMS and normalization statistics;
- linear centered-kernel alignment (CKA), the primary graph edge measure;
- cosine, RMS-correlation, attention-pattern, same-layer, and shared-KV
  sensitivity fields.

Observation was tested on the real model and left logits bit-for-bit unchanged.
The 440 prompts across all Phase 2 splits were hashed before telemetry, with no
normalized cross-split duplicates and a maximum character similarity of 0.800
against the 0.920 leakage stop threshold.

## V1 primary map

At the frozen 2% graph density:

| Measure | Result | Gate |
| --- | ---: | ---: |
| Total heads observed | 1,152 | — |
| Stable non-isolated heads | 760 | — |
| Selected communities | 7 | at least 3 |
| Graph modularity | 0.191 | above null 95th percentile |
| Null 95th percentile | -0.004 | — |
| Paraphrase similarity | 0.802 | greater than unrelated |
| Unrelated-pair similarity | 0.615 | — |
| Median bootstrap assignment probability | 1.000 | at least 0.800 |
| Layer/community ARI | 0.007 | not layer-only |
| Shared-KV/community ARI | -0.0003 | not shared-KV-only |
| Split-half community ARI | **0.350** | **at least 0.700** |

Seven of eight mapping gates passed. The one that failed was the one that makes
the network boundaries meaningful: independent prompt halves did not recover
the same communities.

## V1 anti-cherry-picking check

A mapping-only sensitivity sweep tested graph densities from 0.5% through 20%
and community counts from three through eight. This produced some tempting
numbers. For example, 7.5% density reached a non-nested split-half ARI of 0.888.
Choosing that number after seeing it would have been exactly the kind of
researcher freedom the protocol was designed to prevent.

So the sweep was nested:

1. folds 0 and 2 selected density and community count;
2. folds 1 and 3 remained untouched until one candidate was fixed;
3. the selected candidate had density 10% and five communities;
4. its tuning ARI was 0.858;
5. its held-out ARI fell to **0.660**, below the frozen 0.700 gate.

Some candidates that looked better on the held-out folds had performed poorly
on the tuning folds. They were not eligible to replace the selected candidate
afterward.

## Decision

Both registered attempts end at `STOP_MAPPING_UNSTABLE`, yielding the experiment
status `MECHANISM_NOT_ACHIEVED`.

The result does not say Qwen has no functional structure. The graph was
non-random, paraphrase-sensitive, mixed across layers, and not reducible to
shared-KV families. It says this particular activity definition, corpus size,
continuation design, graph construction, and model did not produce boundaries
stable enough to support the next causal claim.

Accordingly, the following stages were deliberately not run:

- attribution and exact activation patching;
- cross-community attention diffusion (CCAD);
- within-community, attention-noise, randomized-graph, and temperature controls;
- dose calibration;
- pilot generations;
- confirmation and architecture replication.

Running them anyway would turn an unstable clustering into a story about
networks. The stop is the result.

## Integrity

The v1 local run is `runs/fnde-network-map-qwen3-61e90a4`. Its exact mapping data,
400 continuation rows, component and attention array shapes, finite telemetry,
primary failure, nested held-out failure, stop decision, and file checksums all
pass verification. Large telemetry remains local and ignored by Git; the
machine-readable closeout and this report are tracked.

The stronger run is `runs/fnde-v2-consensus-map-qwen3-d8e2e6b`. Its fresh-data
hash, 960-row collection, array shapes, finite telemetry, tuning-only selection,
corrected held-out evaluation, stop decision, and artifact checksums all pass
verification. Its machine-readable closeout is `fnde-v2-closeout.json`.

The original protocol remains in
[NETWORK_DESEGREGATION_SPEC.md](NETWORK_DESEGREGATION_SPEC.md). A future attempt
should change the representation itself—likely modelling information exchange
between components over longer trajectories—not merely add prompts or relax the
0.700 gate after the fact.
