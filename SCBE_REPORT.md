# The LLM Changed—But Not in a Way We Could Trust

**Selective Connectivity Behavioral Experiment (SCBE)**

**Final status: `PILOT_GATE_FAILED` / degenerative behavioral result**

SCCF had already shown that sixteen selected attention-correlation modes could
be flattened reproducibly. SCBE was the real “did the mushrooms do anything
interesting?” test: generate complete answers, compare them with matched
randomness controls, and refuse to call the result creative if competence
slipped.

It stopped at the pilot.

## The controls finally matched

This was not another mechanism or calibration failure. Two coarse control grids
first missed the SCCF dose, before any behavioral response was generated. A
recorded interpolation amendment then produced unusually close matches:

| Condition | Calibration target/result |
| --- | ---: |
| SCCF | KL `0.05730` |
| Random basis, alpha 1.25 | KL `0.05627` |
| Attention noise, sigma 0.46 | KL `0.05664` |
| Temperature, 0.994 | entropy mismatch `0.00064` nats |
| Sham | maximum logit error `0.0` |

Every numerical invariant passed. The comparison was ready and fair enough to
reach generation.

## What was generated

The pilot used 24 fresh prompts across open association, analogy, narrative,
constrained writing, factual exact-answer, and code tasks. Every prompt received
three paired generations under baseline, sham, SCCF, random basis, attention
noise, and matched temperature: 432 complete generations in total.

Baseline and sham text matched exactly in every pair. There were no runtime,
cache, moment-preservation, or non-finite failures.

## Why it stopped

The frozen pilot required each protected task family to retain baseline validity
within 15 percentage points. SCCF failed the factual exact-answer gate:

| Pilot family | Baseline validity | SCCF validity | Change |
| --- | ---: | ---: | ---: |
| Constrained writing | 91.7% | 100.0% | +8.3 pp |
| Factual exact answer | 100.0% | 75.0% | **-25.0 pp** |
| Code | 0.0% | 0.0% | unusable sentinel |

The factual misses were all the same prompt. Asked for only the number of sides
in a dodecagon, baseline returned `12`; SCCF returned `12 sides`. The knowledge
was correct, but exact instruction following was not. This is therefore better
described as a format/obedience failure than a hallucinated fact. It still
violates the registered task and competence gate.

SCCF also exceeded the very strict repeated-4-gram degeneration margin in
analogy, narrative, and code. Some of these flags came from structured output
rather than obvious gibberish, so they are supporting warnings rather than the
cleanest reason to stop. The exact-answer gate alone was sufficient.

## Other pilot behavior

| Family | Baseline validity | SCCF validity | Baseline entropy | SCCF entropy |
| --- | ---: | ---: | ---: | ---: |
| Open association | 100.0% | 83.3% | 0.674 | 0.703 |
| Analogy | 100.0% | 91.7% | 0.306 | 0.479 |
| Narrative | 100.0% | 100.0% | 0.805 | 1.067 |
| Constrained | 91.7% | 100.0% | 0.635 | 0.853 |
| Factual | 100.0% | 75.0% | 0.013 | 0.043 |

The intervention clearly changed full-answer behavior and generally raised
token-level uncertainty. It did not simply collapse: most target responses
remained valid, narrative validity stayed perfect, and constrained validity
improved. But “interesting in some places, less obedient in others” is exactly
the ambiguous pattern the competence gates were designed to reject.

## Measurement limitation

The code sentinel was bad. Qwen commonly returned Markdown-fenced code despite
being asked for code only; the strict parser therefore scored baseline at 0%.
That makes code retention uninformative. A future protocol should safely strip
one recognized outer code fence before parsing, and require a minimum baseline
validity before a category can count as a competence gate.

The repeated-4-gram rule was also too sensitive for JSON and code. Future work
should distinguish structural repetition from runaway repetition, without
retroactively changing this run.

## Decision

SCBE ends at `PILOT_GATE_FAILED`. The untouched 48-prompt confirmation corpus,
blinded judge, embeddings, and confirmatory novelty tests were not run because
the preregistered pilot gate forbade them.

The defensible conclusion is:

> Selective connectivity flattening produces real and structured behavioral
> changes, but this dose did not preserve exact instruction-following competence
> well enough to test a creativity claim on confirmation data.

So: the internal “drug” worked; the behavioral benefit did not.

## Integrity

The frozen base source is commit `01ca7ea`; control amendments are `39f8eb3` and
`f0694f3`. The terminal run is `runs/scbe-v3-qwen3-f0694f3`. Its 432 generation
records include token telemetry, validators, degeneration metrics, controller
invariants, frozen calibration, provenance, and verified SHA-256 checksums.

See [SCCF_BEHAVIORAL_SPEC.md](SCCF_BEHAVIORAL_SPEC.md),
[SCBE_CONTROL_MATCH_AMENDMENT.md](SCBE_CONTROL_MATCH_AMENDMENT.md), and
[SCBE_NOISE_INTERPOLATION_AMENDMENT.md](SCBE_NOISE_INTERPOLATION_AMENDMENT.md).

The exact verification/resume command is:

```bash
uv run lspe run-connectivity-behavior \
  --config configs/fallback.qwen3-4b.yaml \
  --map-run runs/dcf-map-qwen3-6bdfd20 \
  --sccf-run runs/sccf-qwen3-5332e36 \
  --data-root data/phase4 \
  --output runs/scbe-v3-qwen3-f0694f3 \
  --offline
```

It validates the source/data/parent run lock, reuses all 432 content-addressed
generation identities, rebuilds the terminal summaries, and fails if any frozen
input differs.
