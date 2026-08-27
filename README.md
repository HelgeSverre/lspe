# LSPE

## I gave an LLM magic mushrooms. For science.

Not real mushrooms. An LLM has no mouth.

Instead, I cracked open a local language model and messed with its brain while
it was generating text.

The idea came from the **stoned ape hypothesis**: the wonderfully unhinged
theory that psychedelic mushrooms helped kick-start human language, imagination,
and culture. There is no good evidence that mushrooms created the human mind—but
it raises a fantastic question:

> What if temporarily scrambling a mind can make it think in genuinely new ways?

So that is what this project tests.

I inject a controlled disturbance directly into the LLM's internal activations
and hold it there while the model responds. Then I check whether it becomes more
creative, merely more random, or just completely fried.

The early verdict: **mostly fried.**

At small doses, not much happens. Turn it up and the model starts losing the
plot. So far, simply raising the temperature produces more useful variety than
tampering with the model's hidden state.

That is still a result. The point is not to collect screenshots of a chatbot
saying weird stuff and call it consciousness. The point is to run the joke as a
real, reproducible experiment—and accept the answer even when the machine does
not see God.

> To be painfully clear: this is not a biological simulation of psilocybin, the
> model is not conscious, and this proves nothing about human evolution. It is a
> controlled experiment inspired by an extremely entertaining idea.

## The experiment

**Local Latent-State Perturbation Experiment (LSPE)** is an auditable local
harness for applying transient inference-time interventions to open-weight
language models. The current subject is quantized Gemma 4 E4B running through
MLX on Apple silicon.

For each generation, LSPE rotates a selected residual-stream activation toward
a random direction while preserving its norm. The direction remains stable
throughout the response, creating a persistent internal bias rather than
independent noise at every token.

It compares five conditions:

| Condition | What it tests |
| --- | --- |
| `baseline` | Ordinary inference |
| `sham` | Instrumentation with a zero dose |
| `coherent` | One stable perturbation direction throughout the response |
| `white` | A new perturbation direction at each token |
| `temp_match` | Ordinary inference with output entropy matched by temperature |

Intervention strength is calibrated by teacher-forced next-token KL divergence,
not arbitrary vector magnitude. The main question is whether coherent internal
perturbation increases **valid semantic diversity** beyond ordinary sampling
randomness while preserving more competence than incoherent activation noise.

## Results so far

The current Gemma 4 pilot is a useful negative result. At the tested
lower-middle layer, coherent perturbation did **not** outperform entropy-matched
temperature sampling:

| Target KL | Coherent − temperature valid semantic diversity | Outcome |
| ---: | ---: | --- |
| 0.01 | -0.0189 | Not supported |
| 0.03 | -0.0326 | Not supported |
| 0.10 | -0.0431 | Degenerative |
| 0.30 | -0.0431 | Degenerative |

Low doses mostly did little or slightly reduced useful diversity. Stronger
doses increasingly damaged format validity and produced degeneration. The model
did react—but more like a system under structured interference than one granted
a burst of productive association.

These are **pilot observations, not confirmatory findings**. Candidate selection,
a frozen confirmation run, and precision or architecture replication remain to
be completed. Null, negative, and degenerative outcomes use the same reporting
and verification path as positive ones.

## Why this is more than a weird-output generator

LSPE is designed to distinguish an altered internal regime from ordinary
randomness and broken inference:

- coherent perturbations are compared with KL-matched per-token activation noise;
- an output-temperature control is matched for sampling entropy;
- zero-dose shams verify that instrumentation itself does not change logits;
- creativity scores are gated by deterministic task validity;
- calibration, pilot selection, confirmation, and replication are separated;
- prompts, seeds, model revisions, direction fingerprints, environments, and
  checksums are recorded;
- raw generations are append-only and all derived reports are rebuildable.

The desired output is not a gallery of amusing hallucinations. It is a
reproducible answer—even when the answer is "no."

## Setup

```bash
uv sync --frozen
uv run lspe doctor --config configs/smoke.gemma4-e4b.yaml
uv run lspe build-data --output data --force
```

The primary subject is `mlx-community/gemma-4-e4b-it-4bit`. Model fetches use a
pinned revision and fail with an actionable authentication or license message
instead of silently switching models.

## Workflow

```bash
# Validate the runtime and intervention machinery.
uv run lspe preflight --config configs/pilot.gemma4-e4b-v2.yaml --offline

# Calibrate and execute pilot candidates.
uv run lspe pilot --config configs/pilot.gemma4-e4b-v2.yaml --offline --fail-fast
uv run lspe score --run runs/<pilot-run>
uv run lspe analyze --run runs/<pilot-run>
uv run lspe report --run runs/<pilot-run>

# Select one candidate, freeze the protocol, and run confirmation.
uv run lspe select-pilot --config configs/pilot.gemma4-e4b-v2.yaml \
  --pilot-run runs/<pilot-run> [...]
uv run lspe freeze --config configs/confirm.gemma4-e4b.yaml \
  --pilot-run runs/<selected-pilot-run>
uv run lspe run --config configs/confirm.gemma4-e4b.yaml --offline --fail-fast

# Rebuild and verify the evidence.
uv run lspe score --run runs/<confirm-run>
uv run lspe analyze --run runs/<confirm-run>
uv run lspe judge --run runs/<confirm-run> --offline
uv run lspe report --run runs/<confirm-run>
uv run lspe verify --run runs/<confirm-run> --level artifact
```

Pilot execution fails closed when a KL target band is unresolved. `judge` is a
secondary, blinded Qwen pairwise assessment; deterministic validators and
embedding metrics do not depend on it. Human-review exports are blind and
stratified.

See [SPEC.md](SPEC.md) for the full protocol, hypotheses, non-claims, integrity
gates, and required artifacts.
