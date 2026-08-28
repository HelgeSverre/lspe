# I Gave an LLM Magic Mushrooms

**LSPE — Local Latent-State Perturbation Experiment**

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

That is still a result. The point is to run the joke as a real, reproducible
experiment—and accept the answer even when the machine does not see God.

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

## Final results

The experiment is closed with a **degenerative, non-supporting result**. A
locked Gemma confirmation run and an independently calibrated Qwen architecture
replication both found no reliable increase in valid semantic diversity over
entropy-matched temperature sampling.

| Run | Model | Prompts | Coherent − temperature valid semantic diversity (95% CI) | p-value | Result |
| --- | --- | ---: | --- | ---: | --- |
| Primary confirmation | Gemma 4 E4B | 120 | -0.00384 (-0.01201, 0.00413) | 0.350 | Not supported |
| Architecture replication | Qwen3 4B | 60 | 0.00017 (-0.01156, 0.01142) | 0.975 | Not supported |

The Gemma effect is slightly negative and its confidence interval crosses zero.
The Qwen estimate is effectively zero. Because the primary effect is
non-positive and the models use independently calibrated doses, the results are
not pooled and do not establish a positive-direction replication.

Coherent perturbation also did not show a reliable competence advantage over
white (per-token random) activation noise. In the Qwen run it reduced
deterministic validity by 3.9 percentage points relative to white noise; the
Holm-adjusted secondary test was not significant. The overall pattern is more
consistent with structured interference than with a productive, general-purpose
increase in associative exploration.

See [FINAL_REPORT.md](FINAL_REPORT.md) for methods, exact outcomes, integrity
checks, limitations, and reproduction commands. The source reports and raw
append-only generation records remain in the local `runs/` directory; they are
intentionally not committed to Git.

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

## The network experiment

The more ambitious follow-up tried to map something closer to functional
networks across all 1,152 attention heads in Qwen3 4B before temporarily
increasing communication across their boundaries.

The first map failed its stability gate. We then tried substantially harder:
240 entirely fresh prompts across six task families, four generated positions
per prompt, task-balanced graphs, every attention head, and a preregistered
tuning/held-out split. That produced 960 observations and 22 GB of telemetry.

The selected map looked convincing on its tuning folds (ARI `0.808`) but fell
to `0.548` on untouched folds, below the unchanged `0.700` requirement. Its
three apparent communities contained 803, 18, and 10 heads—one continent and
two crumbs, not a credible set of functional networks. So the model still did
not receive the network-level intervention.

See [FNDE_REPORT.md](FNDE_REPORT.md) for the full result and
[NETWORK_DESEGREGATION_SPEC.md](NETWORK_DESEGREGATION_SPEC.md) for the protocol.
This is a useful failure: throwing more data and stricter analysis at the idea
made the answer clearer, and prevented a fragile clustering from being dressed
up as evidence that the model's “brain networks” were desegregated.

### The dynamic attempt

The next experiment dropped static communities entirely. Dynamic Connectivity
Flattening measured head-to-head attention correlations over full token
trajectories, then temporarily flattened those relationships during decoding.

This mechanism finally worked. The dynamic maps reproduced at `0.9998` on
untouched prompts, zero dose was exactly identical to baseline, and calibrated
transforms reduced attention-head correlation while sharply increasing
effective rank.

But there was no safe active dose. At `alpha=0.40`, top-1 retention passed at
`80.30%` while output KL remained below the active band. At `alpha=0.42`, KL
entered the band while retention slipped to `79.51%`. A frozen denser-grid
follow-up confirmed the crossover, so behavioral testing did not proceed.

See [DCF_REPORT.md](DCF_REPORT.md) and
[DYNAMIC_CONNECTIVITY_SPEC.md](DYNAMIC_CONNECTIVITY_SPEC.md). This is the first
version that demonstrably created the intended desynchronized internal regime;
the remaining problem is making that regime selective enough not to cross the
damage line as soon as it meaningfully affects output.

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
