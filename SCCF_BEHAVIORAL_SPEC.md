# Selective Connectivity Behavioral Experiment

**Short name:** SCBE

**Protocol version:** 0.1

**Status:** Frozen before behavioral telemetry

**Subject:** `mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`

**Required parent:** SCCF run `runs/sccf-qwen3-5332e36`, status
`MECHANISM_PASS`, 16-mode mask, `alpha=0.50`

## Question

SCCF found a reproducible internal regime. SCBE asks the question that the
mechanism experiment deliberately could not answer:

> Does selective connectivity flattening make complete answers more
> associatively diverse or novel than ordinary sampling randomness, without
> sacrificing competence in any protected task family?

## Fresh data and split discipline

`data/phase4/` contains 12 control-calibration prompts, 24 pilot prompts, and 48
confirmation prompts. Each behavioral split is balanced across:

- `open_association` and `analogical` (primary target families);
- `narrative` (exploratory open-ended family);
- `constrained`, `factual`, and `code` (separate protection families).

All prompts are new to LSPE, FNDE, DCF, and SCCF. Template structure may repeat,
but objects, domain pairs, required words, questions, and code cases do not.
Exact normalized text and token 5-gram overlap are audited before execution:
no exact prior prompt is allowed and maximum prior-prompt 5-gram Jaccard
similarity must remain below 0.80.
Pilot outcomes cannot alter the frozen SCCF mask, dose, conditions, metrics, or
confirmation gates.

## Paired conditions

Every prompt and generation index uses the same token-sampling seed under:

| Condition | Definition |
| --- | --- |
| `baseline` | Ordinary inference at temperature 0.8 |
| `sham` | Wrapped selective kernel with identity transforms |
| `sccf` | Frozen 16-mode mask at alpha 0.50 |
| `random_basis` | Same per-layer transform eigenvalues in frozen random orthogonal bases |
| `attn_noise` | Independent score noise with per-head moments restored |
| `temp_match` | Ordinary inference at entropy-matched output temperature |

All interventions are decode-only. Model weights and KV-cache history are never
modified in place. Each behavioral prompt receives three generations per
condition, capped at 128 new tokens.

## Control calibration

On the 12 calibration prompts, use frozen greedy 16-token teacher-forced
continuations. Measure SCCF output KL and sampling entropy at temperature 0.8.

Choose the random-basis alpha from
`[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]` and attention-noise sigma from
`[0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.32]`. Select the value minimizing
absolute log-KL error. Each control must achieve median KL within 25% of SCCF or
SCBE stops with `CONTROL_MATCH_FAILED`. Temperature is selected on a frozen
0.05–3.0 grid to match mean SCCF post-filter sampling entropy within 0.02 nats.

Sham must have maximum logit error at most `1e-5`, identical greedy top-1, and
identical generated text under paired seeds. All intervention moment,
non-finite, cache, and model-integrity invariants from SCCF remain mandatory.

## Deterministic validity and degeneration

Validators run without a judge model:

- open association: exactly ten unique common-noun strings in JSON;
- analogy: exact three-field cross-domain bridge JSON;
- narrative: 60–180 words of non-empty prose;
- constrained: exactly four lines, 6–14 words each, including both required
  words;
- factual: exact answer only;
- code: one import-free Python function passing frozen cases in isolation.

Degeneration means any repeated 4-gram or an identical-token run of eight.

## Semantic outcomes

The frozen `sentence-transformers/all-MiniLM-L6-v2` model embeds complete
responses after the subject is unloaded. For every prompt-condition cell,
valid semantic diversity (VSD) is mean pairwise cosine distance, with invalid
pairs contributing zero.

Primary contrasts, tested separately:

1. `sccf - temp_match` VSD on `open_association` confirmation prompts;
2. `sccf - temp_match` VSD on `analogical` confirmation prompts.

The open-association contrast is confirmatory primary; analogy is co-primary
and Holm corrected with it. A positive behavioral claim requires the relevant
prompt-clustered 95% bootstrap interval above zero and Holm-adjusted sign-flip
`p < 0.05`.

Secondary contrasts compare SCCF with `random_basis` and `attn_noise`, and
measure novelty, usefulness, coherence, plausibility, and constraint adherence
with a blinded local judge. Judge parse failure above 5% invalidates judge
outcomes but not deterministic or embedding outcomes.

## Pilot gate

Pilot proceeds to confirmation only if:

- every condition/prompt cell has three successful generations;
- sham text is identical to baseline for every pair;
- each control remains within 35% of SCCF median realized output KL on paired
  first-step telemetry or its calibration match remains within 25%;
- SCCF validity in each protected category is no more than 15 percentage points
  below baseline;
- SCCF degeneration is no more than 5 points above baseline in every category;
- no numerical, cache, weight, or artifact invariant fails.

Pilot novelty does not gate confirmation. Failure stops and preserves artifacts.

## Untouched confirmation gates

Regardless of novelty, a positive interpretation requires all of:

- SCCF validity no more than 10 percentage points below baseline separately in
  `constrained`, `factual`, and `code`;
- SCCF degeneration no more than 2 points above baseline in every category;
- SCCF coherence and plausibility each no more than 0.25 judge points below
  baseline, if judge outcomes are valid;
- complete paired data and all integrity checks.

If competence fails, the status is `DEGENERATIVE` even if novelty rises. If
competence passes but neither primary novelty contrast is positive, status is
`MECHANISM_ONLY`. `BEHAVIORAL_SUPPORT` requires competence plus at least one
corrected positive primary outcome, and the report must name which family.

## Required artifacts

- tracked fresh datasets and leakage audit;
- source, protocol, parent, data, model, and environment hashes;
- control calibration curves and frozen selections;
- append-only generations and token telemetry;
- deterministic validation, degeneration, embeddings, VSD, and paired effects;
- blinded judge responses and unblinding key;
- prompt-clustered bootstrap and corrected tests;
- machine-readable closeout, human report, exact resume command, and SHA-256
  manifests.

Any mismatch or missing required artifact fails closed.
