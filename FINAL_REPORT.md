# LSPE v1 Final Report

## Result

The Local Latent-State Perturbation Experiment is complete. Its confirmatory
result is **degenerative and does not support the primary hypothesis**.

A coherent, norm-preserving residual-stream rotation did not increase valid
semantic diversity beyond entropy-matched temperature sampling in the locked
Gemma 4 confirmation. An independently calibrated Qwen3 replication produced
an estimate effectively equal to zero. Both models showed modest evidence of
signal damage rather than a useful altered computational regime.

## Confirmatory outcomes

| Role | Model | Prompt clusters | Coherent − temperature VSD | 95% CI | p-value | Status |
| --- | --- | ---: | ---: | --- | ---: | --- |
| Primary | Gemma 4 E4B 4-bit | 120 | -0.00384 | [-0.01201, 0.00413] | 0.350 | Degenerative |
| Architecture replication | Qwen3 4B 4-bit | 60 | 0.00017 | [-0.01156, 0.01142] | 0.975 | Degenerative |

The results are shown separately rather than pooled because the subjects use
different architectures and independently calibrated raw intervention doses.

## Primary confirmation

The frozen Gemma candidate intervened at decoder layer 20 with:

- coherent raw dose `0.042`;
- achieved teacher-forced median KL `0.01049` nats;
- independently matched white-noise dose `0.045`;
- white-noise median KL `0.00983` nats;
- entropy-matched temperature `1.112`;
- 120 prompts, five generations, and five paired conditions;
- 3,000 planned and 3,000 observed generations.

Relative to baseline, coherent validity decreased by 1.33 percentage points and
degeneration increased by 2.33 percentage points. Coherent competence was 0.5
percentage points higher than white noise, but the contrast was not significant.
Coherent valid semantic diversity was 0.00742 higher than white noise, but its
Holm-adjusted test was also not significant.

## Architecture replication

The Qwen replication used its independently calibrated middle-depth layer and
target KL `0.01`. It completed 720 planned generations and 720 blinded judge
rows.

Its primary effect was essentially zero. Coherent validity was 1.67 percentage
points below baseline and 3.89 points below white noise. Degeneration was 2.78
points above baseline. These outcomes do not establish a useful or
positive-direction replication.

## Interpretation

LSPE v1 answers a narrow question: a persistent random residual rotation is not
equivalent to merely raising output temperature, but the distinction did not
produce a general creativity benefit. As dose increased in the pilot, validity
usually fell and degeneration increased. The smallest eligible dose survived
the pilot gates but failed confirmation.

The result does not show that every internal intervention is harmful. It shows
that this particular intervention family—random, coherent, norm-preserving
rotation at one decoder layer—did not create the hypothesized useful regime in
the tested models.

The preregistered follow-up in
[NETWORK_DESEGREGATION_SPEC.md](NETWORK_DESEGREGATION_SPEC.md) tests a different
mechanism: temporary changes to empirically mapped functional communication
rather than displacement of one residual stream.

## Integrity

- Pilot selection considered all 12 registered Gemma candidates.
- Exactly one candidate passed the frozen eligibility rules.
- Invalid and superseded runs remain labelled and excluded.
- Raw generation matrices are complete.
- Blinded judge matrices are complete.
- Model revisions, prompts, sampling plans, directions, runtime evidence, and
  token telemetry are preserved locally.
- SHA-256 manifests were rebuilt after final judging and reporting.
- Artifact verification passes for both source runs.

Verified artifact roots:

```text
Gemma: 98cfd333434008f69f2953b23f4af034c59c12d0fe6d8b075e7ea7eb0bd318ab
Qwen:  57afdf54a10e2e92d015f09fde8b05abe07376523fe293adc0b6907e0877b18a
```

## Local source artifacts

```text
runs/gemma4-e4b-coherent-residual-confirm-v2-475b9088d297/
runs/qwen3-4b-architecture-replication-v3-configured-t0.01-50d427756c6b/
runs/combined-gemma4-e4b-qwen3-v1/
```

Verification commands:

```bash
uv run lspe verify \
  --run runs/gemma4-e4b-coherent-residual-confirm-v2-475b9088d297 \
  --level artifact

uv run lspe verify \
  --run runs/qwen3-4b-architecture-replication-v3-configured-t0.01-50d427756c6b \
  --level artifact
```
