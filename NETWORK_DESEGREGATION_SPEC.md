# Functional Network Desegregation Experiment

**Project:** LSPE Phase 2

**Short name:** FNDE

**Status:** Future implementation protocol; do not execute until the LSPE v1 closeout gate passes

**Specification version:** 0.1

**Primary subject:** To be frozen after LSPE v1 closeout

**Purpose:** Test whether temporarily reducing functional segregation among causally relevant Transformer components produces useful associative behavior that cannot be explained by output randomness, attention noise, or generic signal damage.

---

## 1. Executive decision

LSPE Phase 2 MUST model a psychedelic-inspired change in **functional
communication**, not a drug molecule and not a permanent change to model
weights.

The primary intervention MUST:

1. map reproducible functional communities of attention heads from baseline
   activity;
2. confirm that the mapped components causally affect the registered task
   family;
3. temporarily diffuse attention patterns across community boundaries during
   inference;
4. leave checkpoint weights byte-identical before and after every generation;
5. demonstrate that the intervention actually reduces the model-side network
   modularity metric it claims to manipulate;
6. compare the effect with a sham, ordinary temperature, independent attention
   noise, within-community diffusion, and degree-preserving randomized-network
   controls;
7. match controls by teacher-forced output KL and report attention-space
   divergence separately;
8. separate discovery, calibration, pilot selection, frozen confirmation, and
   architecture replication;
9. reject any candidate that achieves novelty primarily through invalidity,
   degeneration, copying, verbosity, or evaluator exploitation;
10. treat later fine-tuning as a separate **integration** experiment, never as
    the acute intervention.

The main question is:

> Does temporarily weakening a model's normal functional boundaries create
> valid cross-domain associations while preserving competence better than
> matched randomness and signal damage?

This is a computational analogy motivated by network-level psychedelic
findings. It is not a biological simulation of psilocybin.

---

## 2. Why this follow-up exists

LSPE v1 applies a coherent, norm-preserving rotation to one residual stream.
That is a clean test of persistent latent displacement, but it does not directly
change relationships among identified functional subsystems.

Human imaging research instead reports acute changes in functional
organization under psilocybin: reduced within-network integration, reduced
between-network segregation, widespread desynchronization, and strong effects
in association networks. Task engagement can reduce those changes, indicating
context dependence. See [Siegel et al., Nature
2024](https://www.nature.com/articles/s41586-024-07624-5).

The analogy must be used carefully:

| Neuroscience term | Meaning in that literature | FNDE operational analogue |
| --- | --- | --- |
| Brain region | Anatomical population | No direct equivalent |
| Functional network | Regions with correlated dynamics | Empirically clustered component activity |
| Functional connectivity | Statistical dependence over time | Dependence between component-output traces |
| Network segregation | Stronger within- than between-network coupling | Weighted graph modularity |
| Desynchronization | Reduced coordination of activity | Reduced within-community dependence |
| Global integration | Less separation across networks | Increased cross-community pattern sharing |
| Drug dose | Pharmacokinetic exposure | Calibrated intervention strength; not biologically equivalent |
| Neuroplasticity | Lasting cellular or synaptic change | A separate optional training experiment |

A Transformer layer MUST NOT be called a brain network. Layers are successive
computational stages. Functional communities must be measured rather than
declared from architectural boundaries.

Fine-tuning MUST NOT represent the acute drug condition. Fine-tuning changes
weights persistently. FNDE's acute intervention changes inference-time routing
and is removed after each generation. Research reporting persistent dendritic
spine changes after psilocybin motivates a later integration study, but does
not justify mixing training into the acute experiment. See [Shao et al., Neuron
2021](https://pubmed.ncbi.nlm.nih.gov/34228959/).

---

## 3. Claims and non-claims

### 3.1 Permitted claim levels

Reports MUST use the strongest claim supported by completed gates and no
stronger:

1. **Mechanism achieved:** the intervention reproducibly reduced the registered
   model-side modularity statistic.
2. **Behavior changed:** the intervention changed registered behavioral
   outcomes relative to baseline.
3. **Distinct from noise:** it outperformed every KL-matched damage and
   randomness control.
4. **Useful regime:** it increased validated association while meeting all
   competence and degeneration non-inferiority gates.
5. **Replicated computational effect:** the direction reproduced in a frozen
   second architecture or precision condition.

### 3.2 Forbidden claims

FNDE MUST NOT claim that:

- attention heads are brain regions;
- detected Transformer communities are homologous to the default mode network;
- graph modularity in a Transformer is the same measurement as fMRI functional
  connectivity;
- the intervention implements 5-HT2A receptor agonism;
- the model is conscious, intoxicated, hallucinating, or having an experience;
- successful behavior validates the stoned ape hypothesis;
- fine-tuning is equivalent to taking a psychedelic;
- novelty alone is creativity;
- a result in one prompt set or model generalizes beyond it.

The phrase **psilocybin-inspired** is permitted. **Psilocybin simulation** is
not.

---

## 4. Research questions and hypotheses

### 4.1 Primary question

At matched teacher-forced output divergence, does cross-community attention
diffusion increase validated cross-domain association more than ordinary output
temperature, independent attention-logit noise, and randomized-network
diffusion while retaining deterministic competence?

### 4.2 Confirmatory hypotheses

**H1 — Mechanism manipulation**

Let `delta_Q = Q_ccad - Q_sham`, where lower modularity means less segregation.
The selected acute dose must satisfy `delta_Q <= -delta_min` on held-out mapping
prompts, with the upper bound of its 95% prompt-clustered confidence interval
also at or below `-delta_min`.

**H2 — Useful association**

The intervention produces greater valid semantic and relational diversity than
the entropy-matched temperature control.

**H3 — Structured routing versus damage**

At matched teacher-forced output KL, the intervention retains greater
deterministic validity and competence than independent attention-logit noise.

**H4 — Network specificity**

Cross-community diffusion outperforms both within-community diffusion and the
degree-preserving randomized-community control.

**H5 — Architecture robustness**

The direction of the preregistered primary effect replicates in a frozen second
model architecture or precision condition.

H2–H5 MUST NOT be tested confirmatorily unless H1 passes. A behavioral change
without an achieved modularity change is an intervention failure, not evidence
for the theory.

### 4.3 Exploratory questions

- Does externally grounded task engagement suppress the intervention effect?
- Is there an inverted-U dose response?
- Which functional communities and layer bands are most sensitive?
- Does a pulse leave context-mediated effects after the intervention ends?
- Are effects stronger for analogy and conceptual blending than lexical
  diversity?
- Does an optional integration stage retain useful discoveries?

All exploratory results MUST be labelled exploratory permanently.

---

## 5. Overall experimental sequence

```text
LSPE v1 closeout
      |
      v
baseline component mapping
      |
      v
stability + causal validation
      |
      v
kernel calibration and falsification
      |
      v
pilot: layer window x dose x seed
      |
      v
single candidate selected and frozen
      |
      v
confirmatory generation + scoring
      |
      v
architecture or precision replication
      |
      v
optional integration study (separate protocol)
```

No stage may inspect a later split. Every transition MUST produce an immutable
lock artifact and a machine-readable gate decision.

---

## 6. Entry gate: LSPE v1 must be closed first

Phase 2 work MUST NOT begin until all of the following are true:

- every planned LSPE v1 candidate is complete or explicitly abandoned with a
  reason;
- one frozen LSPE v1 confirmation has run, or the closeout states why no
  candidate was eligible;
- invalid historical runs are excluded by machine-readable status, not only a
  Markdown warning;
- the v1 report, artifact verification, and checksums are complete;
- the exact source commit for v1 is recorded;
- lessons from v1 are frozen in `lspe-v1-closeout.json`;
- no Phase 2 mapping prompt or confirmatory prompt has been used to tune v1;
- Phase 2 data splits have immutable hashes before functional mapping begins.

Failure of any entry item is a hard stop.

---

## 7. Subject and runtime requirements

### 7.1 Model requirements

The primary subject MUST:

- be open-weight and locally executable;
- expose per-head attention logits or pre-softmax scores;
- expose per-head value/output contributions before head summation;
- permit deterministic hooks without changing checkpoint weights;
- support exact disabling and removal of hooks;
- fit with mapping telemetry under the memory hard limit;
- pass baseline, cache, and zero-intervention equivalence tests;
- have a pinned immutable revision and hashed model files.

Gemma 4 E4B MAY remain the primary subject only if MLX exposes the required
per-head tensors without a silent architecture patch. Otherwise, a conventional
dense Qwen model SHOULD be used for Phase 2 and labelled accordingly.

### 7.2 Weight immutability

Before and after every execution phase:

- hash every model file;
- hash or checksum every in-memory parameter tensor in a stratified sample;
- assert that no optimizer exists in the acute runtime;
- assert that autograd/training mode is disabled;
- fail if any parameter value or model file changes.

### 7.3 Runtime determinism

Record model revision, runtime commit, package lock, hardware, prompt rendering,
cache implementation, kernels, seeds, intervention graph, and compilation flags.
Sham generation MUST be identical to baseline under greedy decoding and within
the registered logit tolerance under cached and uncached execution.

---

## 8. Defining functional nodes

### 8.1 Primary node definition

One node is one attention head's contribution to the residual stream after its
value/output projection and before heads are summed:

```text
z[l,h,p] in R^d_model
```

where `l` is layer, `h` is head, and `p` is token position.

Raw neurons, residual dimensions, and layers MUST NOT be used as primary nodes.
Head outputs are selected because they are separable communication components
and attention value vectors are an architectural information bottleneck. This
is consistent with the Transformer-circuits account of attention heads, while
remaining an engineering analogy rather than a biological mapping. See [A
Mathematical Framework for Transformer
Circuits](https://transformer-circuits.pub/2021/framework/index.html).

### 8.2 Secondary node definitions

MLP outputs and validated sparse features MAY be examined exploratorily.
Cross-layer sparse coders MUST NOT be required for the first confirmation:
cross-layer features are promising but remain an additional learned model with
reconstruction and interpretability failure modes. See [Sparse Crosscoders for
Cross-Layer Features](https://transformer-circuits.pub/2024/crosscoders/index.html).

### 8.3 Grouped-query attention

Query heads sharing a key/value head MUST remain distinct primary nodes, but
shared-KV ancestry MUST be recorded. Sensitivity analyses MUST repeat community
mapping after collapsing shared-KV families. A result that exists only because
the graph rediscovers architectural KV sharing MUST be labelled artifactual.

---

## 9. Baseline functional mapping

### 9.1 Mapping corpus

Create a dedicated `network_map` split containing at least:

- 200 prompts;
- balanced constrained, factual, narrative, analogical, code, and control
  tasks;
- fixed greedy continuations and fixed sampled continuations;
- no prompt from pilot, confirmation, replication, or judge training;
- at least 50 paraphrase pairs for stability analysis;
- at least 25 deliberately unrelated prompt pairs as negative controls.

The split and hashes MUST be frozen before collecting component activity.

### 9.2 Activity matrices

For each head, collect residual contribution vectors over eligible generated
positions. Exclude padding, prompt-template control tokens, EOS, and positions
after a failure state.

Construct a matrix per head:

```text
Z_i = rows(samples x token positions), columns(d_model)
```

Normalize per head using statistics fit only on the mapping split. Persist
means, scales, inclusion masks, and hashes.

### 9.3 Functional dependence

The primary edge weight between heads `i` and `j` MUST be linear centered kernel
alignment (linear CKA) between `Z_i` and `Z_j`, computed on matched positions.
CKA is basis-insensitive and avoids pretending that individual residual
dimensions are aligned biological units.

The implementation MUST also compute:

- mean cosine similarity of residual contributions;
- correlation of head-output RMS time series;
- attention-pattern Jensen–Shannon similarity;
- shared-KV and same-layer indicators.

Only linear CKA defines the primary graph. The others are sensitivity measures.

### 9.4 Graph construction

Construct a weighted undirected graph after:

1. removing self-edges;
2. retaining positive finite weights;
3. applying a preregistered density threshold selected from mapping data only;
4. requiring every retained node to meet minimum activity and variance;
5. recording connected components and isolated nodes.

Run deterministic spectral clustering over a preregistered range of community
counts. Select the count by nested mapping-only stability, not by behavioral
outcome. Louvain or Leiden MAY be a sensitivity analysis but MUST NOT silently
replace the primary deterministic method.

### 9.5 Stability gate

Community mapping passes only if all conditions hold:

- split-half adjusted Rand index is at least `0.70`;
- bootstrap median node-assignment probability is at least `0.80`;
- paraphrase mapping is more stable than unrelated-prompt mapping;
- communities are not explained solely by layer number or shared-KV family;
- at least three stable communities exist;
- at least two selected layers contain heads from more than one community;
- graph modularity exceeds the degree-preserving null's 95th percentile.

Thresholds MAY be revised during a mapping-only feasibility study, but the
final values MUST be frozen before pilot generations. Failure means the model
does not supply a stable functional-network substrate for this protocol.

---

## 10. Causal validation

Correlation is not mechanism. Community nodes MUST be causally screened before
intervention selection.

### 10.1 Screening method

Use attribution patching to rank heads on the mapping corpus, followed by exact
activation patching or ablation for every head admitted to an intervention
window. Approximate methods can have false negatives; exact validation is
mandatory for the final set. See [AtP*: Kramár et al.
2024](https://arxiv.org/abs/2403.00745).

### 10.2 Causal eligibility

A head is eligible only if:

- exact intervention changes at least one registered task metric beyond the
  sham tolerance;
- the effect replicates over prompt bootstrap samples;
- its activation is not dominated by template or positional artifacts;
- ablation does not cause immediate catastrophic degeneration;
- its causal effect is not explained solely by output-token frequency.

At least two communities must contain eligible heads. Otherwise Phase 2 stops.

### 10.3 No interpretability storytelling

Communities MAY receive neutral identifiers such as `C01`. Semantic names such
as “self network,” “creative network,” or “visual network” are forbidden unless
validated by preregistered counterfactual tasks and causal interventions.

---

## 11. Primary acute intervention

### 11.1 Name

**Cross-Community Attention Diffusion (CCAD)**

### 11.2 Intervention site

CCAD operates on pre-softmax attention scores for eligible heads within the same
layer and token step. It does not modify weights, cached values, MLP outputs, or
the residual stream directly.

Same-layer operation is required for v1 of FNDE because attention-score rows
share a well-defined key-position space. Cross-layer score mixing is forbidden
until an explicit causal alignment method is separately validated.

### 11.3 Transformation

For layer `l`, head `h`, and current query position, let `s_h` be the finite,
mask-applied attention-logit row over available key positions. Let `C(h)` be its
frozen community. Define eligible donors as causally validated heads in the same
layer whose community differs from `C(h)`.

Standardize each donor row over unmasked positions:

```text
u_h = (s_h - mean(s_h)) / max(std(s_h), epsilon)
```

Construct a cross-community donor using frozen, normalized graph weights:

```text
d_h = sum_j w[h,j] * u_j,  where C(j) != C(h)
```

Apply diffusion strength `alpha`:

```text
u'_h = (1 - alpha) * u_h + alpha * d_h
s'_h = mean(s_h) + std(s_h) * u'_h / max(std(u'_h), epsilon)
```

Masked positions remain negative infinity. The transformed row then enters the
model's ordinary softmax.

Properties that MUST be tested:

- `alpha = 0` is numerically identical to sham;
- no masked key becomes reachable;
- rows remain finite before mask restoration;
- pre-softmax mean and standard deviation are preserved within tolerance;
- intervention never reads a future token;
- no donor comes from the same frozen community;
- all donor weights and community assignments are immutable during a run.

This kernel intentionally changes **who attends similarly to whom** while
constraining gross logit scale. It does not claim to reproduce receptor
pharmacology.

CCAD does not mathematically guarantee lower measured modularity: donor mixing
can have unexpected downstream effects. That is why H1 is an empirical gate.
A candidate that fails to lower held-out modularity is rejected even if its
outputs look interesting.

### 11.4 Temporal envelope

The confirmatory intervention MUST use a frozen onset–plateau–decay envelope:

```text
alpha(t) = alpha_peak * envelope(t)
```

The envelope is indexed by generated token count, never wall-clock time. Pilot
selection may compare continuous and pulsed envelopes. The confirmation uses
exactly one selected envelope.

### 11.5 Online mechanism telemetry

For every intervened token and layer, record:

- pre/post attention entropy by head;
- pre/post attention-logit mean and variance;
- pre/post head-pattern similarity matrix;
- online modularity estimate;
- donor communities and frozen edge-weight hash;
- attention KL and JS divergence;
- output-distribution KL and JS divergence;
- top-1 agreement and top-k overlap;
- non-finite, masked-position, and zero-variance counts.

Full matrices MAY be stored in Parquet for a stratified subset; sufficient
summaries to reproduce every gate are mandatory for all rows.

---

## 12. Required conditions

Every confirmatory prompt and sampling seed MUST run under all conditions in a
randomized order:

| ID | Intervention | Purpose |
| --- | --- | --- |
| `baseline` | None | Ordinary inference |
| `sham` | CCAD wrapper, `alpha=0` | Instrumentation control |
| `ccad` | Cross-community diffusion | Primary altered-state analogue |
| `within_diffusion` | Same kernel using same-community donors | Tests boundary specificity |
| `random_graph` | Cross-community labels from degree-preserving graph randomization | Tests discovered-network specificity |
| `attn_white` | Independent, zero-mean attention-logit noise per head/token | Damage control |
| `temp_match` | No internal intervention; entropy-matched output temperature | Output randomness control |

Optional exploratory conditions:

- `ccad_pulse`;
- `ccad_ramp`;
- `residual_rotation_v1` as a bridge to LSPE v1;
- `creative_prompt`;
- `head_dropout_match`;
- `logit_noise_match`.

No required control may be dropped because it performs poorly or is expensive.

---

## 13. Calibration

### 13.1 Two-axis calibration

Calibration MUST report both:

1. **Mechanism dose:** change in functional modularity from sham.
2. **Behavioral dose:** teacher-forced next-token KL from baseline.

Raw `alpha` alone is never a dose claim.

### 13.2 Calibration corpus

Use a dedicated calibration split disjoint from mapping, pilot, confirmation,
and replication. Use teacher-forced continuations so every condition evaluates
the same token history.

### 13.3 Candidate bands

The mapping-only feasibility phase MUST define:

- at least four monotonic target bands for modularity reduction;
- output-KL safety ceilings;
- attention-entropy safety bounds;
- candidate-specific matching tolerances.

The calibration curve MUST densely bracket every selected target. Interpolation
across an unobserved discontinuity is forbidden.

### 13.4 Control matching

`attn_white`, `within_diffusion`, and `random_graph` MUST be independently
matched to CCAD's median teacher-forced output KL within the frozen tolerance.
`temp_match` MUST match CCAD's mean post-filter sampling entropy.

Matching failures invalidate the candidate before generation. Reusing CCAD's
raw strength for a control is forbidden.

### 13.5 Calibration acceptance gate

A candidate passes only if:

- its modularity reduction is inside the registered target band;
- its output KL is below the safety ceiling;
- every mandatory control is matched;
- sham equivalence passes;
- no numerical or mask violation occurs;
- the dose curve is locally monotonic or the non-monotonic region is excluded;
- at least 95% of eligible tokens have valid donors.

---

## 14. Prompt corpus and outcome design

### 14.1 Split isolation

Use immutable, non-overlapping splits:

```text
network_map
causal_map
calibration
pilot
confirm
replication
controls
```

Near-duplicate detection MUST run across splits using normalized text hashes,
character similarity, and embedding similarity. Suspected leakage is a hard
stop.

### 14.2 Primary behavioral tasks

The primary outcome SHOULD emphasize relational novelty rather than word-list
variation:

- cross-domain analogies with an explicit source, target, and mechanism;
- conceptual blending under factual constraints;
- alternative uses with feasibility mechanisms;
- hypothesis generation from supplied evidence;
- story or design continuation with registered constraints;
- remote-association tasks with verifiable links.

Every task requires deterministic schema validation plus content-grounded
checks. Fluency or unusual wording alone receives no novelty credit.

### 14.3 Competence controls

Include arithmetic, exact retrieval from supplied context, JSON transformation,
short code functions, logical consistency, and instruction adherence. Controls
must be sufficiently easy that baseline avoids floor effects and sufficiently
varied that one formatting failure does not dominate competence.

### 14.4 Primary outcome

The primary behavioral outcome is **Validated Relational Novelty (VRN)**:

```text
VRN = validity * grounding * relation_novelty * noncopying
```

Each factor is in `[0,1]`. Deterministic failures set VRN to zero. Relation
novelty MUST compare extracted source–relation–target triples, not only whole
sentence embeddings.

The exact extraction, embedding model, thresholds, and aggregation MUST be
frozen before confirmation. A blinded human audit of at least 10% of rows MUST
estimate extraction error.

### 14.5 Secondary outcomes

- valid semantic diversity for continuity with LSPE v1;
- deterministic validity and competence;
- factual grounding;
- pairwise blinded usefulness and surprise;
- lexical and structural degeneration;
- self-BLEU and copying from prompts;
- response length and verbosity;
- refusal and template leakage;
- output entropy and calibration error.

No single model judge may define the primary outcome.

---

## 15. Pilot selection

The pilot may explore only the preregistered matrix of:

- eligible layer windows;
- modularity-reduction bands;
- temporal envelopes;
- intervention seeds.

Candidate selection MUST be algorithmic and frozen before pilot scoring. Use
the following lexicographic rule:

1. reject incomplete or integrity-invalid candidates;
2. reject candidates failing the H1 mechanism manipulation;
3. reject candidates crossing competence or degeneration safety margins;
4. reject candidates with unmatched controls;
5. among survivors, maximize the minimum standardized VRN contrast against
   `temp_match`, `attn_white`, and `random_graph`;
6. break ties by lower output KL, then lower raw `alpha`, then candidate ID.

If no candidate survives, report `NO_ELIGIBLE_CANDIDATE`. Do not loosen gates
after inspecting behavior.

---

## 16. Frozen confirmation

Before confirmation, write an immutable experiment lock containing:

- source commit and dirty-tree status;
- model and runtime revisions;
- every data hash;
- functional graph and community hashes;
- node eligibility and causal-validation evidence;
- selected layer window, donor graph, dose, and envelope;
- every control's independently calibrated parameter;
- prompt and condition order-generation algorithm;
- sample-size calculation and stopping rule;
- all primary and secondary metrics;
- exclusion rules;
- confidence intervals, multiplicity correction, and decision thresholds;
- report schema and expected artifact count.

Confirmation MUST fail if the working tree, model, graph, configuration, or data
does not match the lock.

Sequential peeking is forbidden. Operational monitoring may inspect completion,
memory, timing, hashes, and numerical errors—but not condition-labelled
behavioral outcomes.

---

## 17. Statistics and decision rules

### 17.1 Unit of analysis

The prompt is the clustering unit. Multiple generations and conditions are
paired within prompt. Intervention seeds are crossed with prompts rather than
confounded with conditions.

### 17.2 Primary contrasts

Preregister:

```text
H1: ccad - sham modularity change
H2: ccad - temp_match VRN
H3: ccad - attn_white competence and VRN
H4a: ccad - within_diffusion VRN
H4b: ccad - random_graph VRN
```

H2 support requires a positive prompt-clustered 95% confidence interval and
all non-inferiority gates. H3 and H4 require Holm correction across their
registered family.

### 17.3 Non-inferiority gates

Freeze margins from baseline variance and domain relevance before the pilot.
At minimum include:

- deterministic competence;
- factual grounding;
- schema validity;
- degeneration;
- refusal rate;
- response length.

A statistically positive novelty effect that fails any safety margin is
classified `DEGENERATIVE`, not supported.

### 17.4 Status vocabulary

Use exactly one primary status:

- `MECHANISM_NOT_ACHIEVED`
- `NO_ELIGIBLE_CANDIDATE`
- `NOT_SUPPORTED`
- `SUPPORTED_UNREPLICATED`
- `SUPPORTED_REPLICATED`
- `DEGENERATIVE`
- `INTEGRITY_INVALID`

---

## 18. Falsification and negative controls

The implementation MUST actively try to disprove its preferred interpretation.

Required falsification tests:

- shuffle community labels while preserving layer and community sizes;
- use degree-preserving randomized donor graphs;
- repeat mapping on template tokens only to detect formatting circuits;
- regress out layer, head index, shared-KV family, output RMS, and token position;
- verify results under alternative graph density thresholds;
- verify the modularity result with at least one secondary dependence metric;
- test whether response length alone explains novelty;
- test whether invalid JSON converted to free text reverses the effect;
- test whether a direct “be more associative” prompt matches the effect;
- test intervention removal for exact recovery on a fresh context;
- rerun a stratified sample from the immutable lock.

If a simpler explanation survives, the report MUST prefer it.

---

## 19. Artifact and audit requirements

Each run MUST contain:

```text
manifest.json
experiment.lock.yaml
resolved-config.yaml
architecture.json
model-files.json
packages.lock.json
environment.txt
source-status.json
prompts.snapshot.jsonl
prompt-renders.jsonl
generation-plan.jsonl
generations.jsonl
component-map.parquet
functional-graph.parquet
communities.json
causal-validation.parquet
intervention-telemetry.parquet
token-metrics.parquet
scores.parquet
prompt-effects.parquet
analysis.json
report.json
report.md
report.html
checksums.sha256
```

Raw generations and telemetry are append-only. Derived artifacts MUST be
rebuildable. Invalid runs remain preserved with a machine-readable invalidity
code and MUST be rejected by scoring, selection, analysis, and reporting.

Verification levels:

- `artifact`: schema, counts, hashes, lock, exclusions, graph provenance;
- `mechanism`: rebuild mapping and modularity gates from component telemetry;
- `replay`: reproduce a deterministic stratified sample;
- `full`: artifact + mechanism + replay.

---

## 20. Software architecture

Phase 2 SHOULD extend LSPE without changing v1 behavior:

```text
src/lspe/
  networks/
    nodes.py
    activity.py
    dependence.py
    graph.py
    communities.py
    stability.py
    causal.py
  interventions/
    attention_base.py
    ccad.py
    attention_white.py
    graph_randomized.py
    envelope.py
  calibration/
    network_dose.py
    attention_match.py
  metrics/
    modularity.py
    relational_novelty.py
  analysis/
    network_analysis.py
```

Required CLI additions:

```bash
uv run lspe network-map --config <config>
uv run lspe network-validate --map <map-run>
uv run lspe network-calibrate --config <config> --map <validated-map>
uv run lspe network-pilot --config <config> --calibration <calibration-run>
uv run lspe network-freeze --config <config> --pilot-run <run>
uv run lspe network-run --lock <experiment.lock.yaml>
uv run lspe network-score --run <run>
uv run lspe network-analyze --run <run>
uv run lspe network-report --run <run>
uv run lspe network-verify --run <run> --level full
```

Every command MUST support dry-run, offline operation, resumption by
content-addressed row ID, structured events, and non-zero failure status.

---

## 21. Mandatory test gates

### 21.1 Unit tests

- exact sham identity;
- mask preservation;
- mean/variance preservation of CCAD logits;
- donor exclusion by community;
- deterministic graph and clustering output;
- stable seed derivation;
- modularity calculation against known graphs;
- degree-preserving null invariants;
- envelope boundary values;
- control-dose independence;
- VRN gating and aggregation;
- invalid-run rejection.

### 21.2 Integration tests

- hooks affect only selected heads and decode positions;
- cached and uncached paths agree within frozen tolerances;
- weights remain unchanged after intervention;
- wrapping then unwrapping restores exact baseline output;
- graph hashes propagate into every generation row;
- telemetry reproduces the online mechanism gate;
- resume does not duplicate or reorder rows;
- deliberate donor, mask, and calibration corruption fails closed.

### 21.3 Scientific smoke tests

- sham equals baseline;
- stronger calibrated doses usually produce greater mechanism change;
- attention white noise is distinguishable from CCAD in mechanism telemetry;
- random labels remove network specificity;
- output temperature does not change internal modularity;
- no condition leaks into prompts, filenames presented to judges, or review UI.

No pilot may begin until all mandatory tests pass on the exact runtime profile.

---

## 22. Optional integration study

Integration is a separate experiment performed only after the acute
confirmation is reported and locked.

The integration study MAY:

1. collect CCAD outputs that passed deterministic validity and blinded human
   review;
2. create matched baseline discoveries with identical selection effort;
3. fine-tune separate checkpoint copies on each selected corpus;
4. evaluate both checkpoints on untouched transfer tasks;
5. compare learning efficiency, generalization, and catastrophic forgetting.

It MUST include:

- a baseline-output fine-tuning control;
- a temperature-output fine-tuning control;
- identical example counts and optimization budgets;
- contamination checks;
- held-out evaluation created before training;
- independent checkpoint names and hashes.

The acute-state model is never overwritten. A benefit after selective
fine-tuning supports only the claim that the altered condition generated useful
training material under external selection.

---

## 23. Stop conditions

Stop immediately and preserve artifacts if:

- functional communities fail stability or null-graph gates;
- fewer than two communities contain causally eligible components;
- required tensors cannot be exposed without an unreviewed model patch;
- sham differs from baseline beyond tolerance;
- any weight changes;
- attention masks are violated;
- any mandatory control cannot be dose-matched;
- candidate modularity does not move in the registered direction;
- output KL crosses the safety ceiling;
- non-finite tensors occur;
- prompt leakage or condition unblinding occurs;
- source, model, data, graph, or lock hashes mismatch;
- the execution matrix is incomplete at analysis time.

Stopping is a valid experimental result, not permission to weaken the protocol.

---

## 24. Implementation order

1. Close and lock LSPE v1.
2. Add machine-readable invalid-run exclusion to v1 if absent.
3. Implement per-head observation with no intervention.
4. Prove baseline and wrapped-sham equivalence.
5. Build mapping corpus and leakage tests.
6. Implement functional dependence and deterministic community mapping.
7. Pass stability and null-graph gates.
8. Implement approximate then exact causal validation.
9. Implement CCAD and mandatory controls.
10. Pass mask, weight, cache, and telemetry tests.
11. Calibrate mechanism and output divergence independently.
12. Execute the frozen pilot matrix.
13. Select exactly one candidate or report none eligible.
14. Freeze and execute confirmation without peeking.
15. Verify, report, and only then run replication.
16. Consider the separate integration study only after acute closeout.

---

## 25. Definition of done

FNDE is complete only when:

- the functional graph is stable, causally screened, and independently
  rebuildable;
- the selected intervention demonstrably changes registered modularity;
- every mandatory control is independently calibrated and complete;
- confirmation was run from an immutable lock without outcome peeking;
- competence and degeneration gates were applied regardless of effect direction;
- all raw and derived artifacts pass full verification;
- one replication is complete or explicitly reported as unavailable;
- claims remain within the hierarchy in Section 3;
- null, negative, degenerate, and mechanism-failure outcomes are reported as
  fully as positive outcomes.

The success criterion is not that the model becomes more creative. The success
criterion is that the experiment can tell, reproducibly and honestly, whether
it did.

---

## 26. Evidence base and confidence

### High confidence

- Human psilocybin imaging supports acute desynchronization, reduced network
  segregation, strong association-network effects, and modulation by task
  engagement: [Siegel et al., Nature
  2024](https://www.nature.com/articles/s41586-024-07624-5).
- Psilocybin can produce persistent dendritic-spine changes in mouse frontal
  cortex: [Shao et al., Neuron
  2021](https://pubmed.ncbi.nlm.nih.gov/34228959/).
- Activation patching supplies causal component tests, and approximate methods
  require exact follow-up because of false negatives: [AtP*,
  2024](https://arxiv.org/abs/2403.00745).

### Medium confidence

- Attention heads are a useful first component granularity for a model-side
  functional graph. They are architectural information-routing units, but no
  accepted standard says they are the correct analogue of functional brain
  regions.
- Linear CKA plus stability and causal screening is a defensible operational
  definition of functional dependence. It is a protocol choice, not established
  psychedelic-computation methodology.

### Low confidence / explicitly experimental

- CCAD may reproduce behaviorally useful consequences of reduced network
  segregation.
- Transformer modularity reduction may have any meaningful relationship to
  psychedelic cognition.
- Outputs from an acute intervention may provide better material for later
  learning.

These low-confidence propositions are precisely what FNDE is designed to test;
they MUST NOT appear as assumptions in its conclusions.
