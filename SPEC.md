# Local Latent-State Perturbation Experiment (LSPE)

**Status:** Implementation specification, version 1.0  
**Primary target:** Apple M2 MacBook with 32 GB unified memory  
**Primary subject:** Quantized Gemma 4 E4B Instruct on MLX  
**Purpose:** Test whether transient, reproducible perturbations of internal language-model activations produce structured changes in associative behaviour that cannot be explained by ordinary output sampling noise alone.

---

## 1. Executive decision

Implement a local experimental harness named **LSPE**. The harness MUST:

1. Load a local open-weight language model through a backend that exposes its decoder layers.
2. Apply transient inference-time interventions to selected residual-stream activations without modifying stored weights.
3. Compare a temporally coherent perturbation against unmodified inference, a zero-dose sham, per-token white activation noise, and an output-temperature control matched for sampling entropy.
4. Calibrate intervention strength by its effect on the model's next-token distribution rather than by raw vector magnitude alone.
5. Evaluate both divergent behaviour and convergent competence using deterministic validators, independent embeddings, blinded model judging, and optional human review.
6. Separate calibration, pilot selection, and confirmatory evaluation.
7. Produce append-only raw data, reproducible manifests, checksums, statistical analyses, Markdown and HTML reports, and a verification command.
8. report null, negative, and degenerative results with the same completeness as positive results.

The first implementation MUST use **activation perturbation**, not permanent random mutation of model weights. Weight perturbation, attention-head mixing, and cross-layer rewiring belong to later experiments because they introduce additional confounds.

This is an altered-state-*inspired* computational experiment. It MUST NOT claim that a language model is conscious, intoxicated, psychedelic, or a biological simulation.

---

## 2. Model and runtime matrix

### 2.1 Primary subject

Use:

```text
mlx-community/gemma-4-e4b-it-4bit
```

Preferred runtime:

```text
MLX + MLX-VLM on Apple silicon
```

The fetch command MUST detect whether the Gemma license has been accepted and whether a Hugging Face token is required. It MUST fail with an actionable authentication message rather than falling back to an unintended model revision.

Rationale:

- The quantized checkpoint is small enough to leave substantial memory headroom on a 32 GB M2 MacBook.
- Gemma 4 E4B is capable enough for structured creativity and control tasks.
- Its hybrid local/global attention architecture provides meaningful layer categories for later analysis.
- Its per-layer embeddings make it an interesting subject, while the first intervention can remain confined to the ordinary post-layer residual stream.

Use **text-only input** in version 1. Do not initialize or exercise vision/audio processing during the experiment unless the runtime requires their modules to exist. Disable speculative decoding, draft models, KV-cache quantization, tool use, network access, and reasoning/thinking mode.

### 2.2 Precision control

Use one of the following, in priority order:

```text
google/gemma-4-E2B-it converted or loaded in BF16 through MLX-VLM
mlx-community/gemma-4-e2b-it-8bit
mlx-community/gemma-4-e2b-it-4bit
```

The preferred control is E2B BF16 because it tests whether the main result survives outside the E4B 4-bit quantization regime while remaining practical on 32 GB unified memory.

### 2.3 Architecture fallback and replication control

Use:

```text
mlx-community/Qwen3-4B-Instruct-2507-4bit
```

This model serves two purposes:

- automatic fallback when the Gemma adapter fails mandatory integrity checks;
- independent architecture replication after the Gemma experiment.

A fallback run MUST be labelled as Qwen and MUST NOT be reported as a Gemma result.

### 2.4 Runtime compatibility policy

Gemma 4 support in MLX tooling has been version-sensitive. The agent MUST NOT assume that the newest package combination is correct merely because it imports or loads weights.

The repository MUST contain a bootstrap resolver that:

1. Creates an isolated Python 3.12 environment.
2. Tries the current stable MLX/MLX-VLM combination.
3. Runs the complete model-integrity preflight in Section 8.
4. If it fails, tries an explicitly configured fallback dependency profile.
5. Writes the exact selected versions, package hashes, source commit identifiers, and model revisions into a lock manifest.
6. Refuses to run pilot or confirmatory experiments when no profile passes.

Maintain explicit runtime candidates rather than an open-ended dependency search:

1. `mlx-vlm-stable`: the current stable MLX and MLX-VLM releases resolved at implementation time;
2. `mlx-vlm-pinned`: a reviewed MLX-VLM source commit containing the relevant Gemma checkpoint-loading fixes;
3. `mlx-lm-text`: an optional text-backbone adapter using a reviewed MLX-LM source revision;
4. `qwen-fallback`: the Qwen adapter on a stable MLX-LM release.

Historical issue reports found that one Gemma checkpoint loaded under an older MLX/MLX-LM combination and failed under its immediate successor. Treat that only as evidence that preflight and exact locking are necessary, not as a recommendation to install old versions blindly.

Do not monkey-patch `load_weights(strict=False)` silently. If a compatibility patch is necessary, implement it in a named adapter module, test it, record its source hash, and list every ignored checkpoint key. Ignored keys MUST match a reviewed allow-list; unexpected ignored keys are fatal.

---

## 3. Scientific question and non-claims

### 3.1 Main question

At matched output uncertainty, can a transient, internally applied, temporally coherent activation perturbation increase **valid semantic diversity** more than ordinary temperature sampling while preserving more competence than temporally incoherent activation noise?

### 3.2 Interpretation

A positive result would establish only that a particular intervention creates a distinct and potentially useful computational regime in the tested models. It would not establish:

- subjective experience;
- an analogue of human intoxication in any biological sense;
- a mechanism of human evolution;
- generalized creativity across all tasks or models;
- useful novelty without external selection and verification.

### 3.3 Primary hypotheses

**H1 — Internal intervention versus output randomness**  
At a preregistered moderate dose, coherent residual perturbation will produce higher valid semantic diversity than an unmodified model whose decoding temperature has been calibrated to match the perturbed model's mean post-filter sampling entropy.

**H2 — Structured state versus signal damage**  
At matched teacher-forced output divergence, coherent perturbation will retain a higher deterministic validity and competence score than independently resampled per-token activation noise.

**H3 — Quantization or architecture robustness**  
The direction of the H1 effect will replicate in at least one precision or architecture control, even if its magnitude differs.

### 3.4 Exploratory questions

- Which layer regions are most sensitive to perturbation?
- Do local-attention and global-attention layers respond differently?
- Is there an inverted-U dose-response curve?
- Does temporary perturbation create effects that persist after the intervention is disabled because altered tokens remain in the context and KV cache?
- Does external task grounding suppress perturbation effects?
- Does context or “set and setting” shape the altered trajectory rather than being erased by noise?

Exploratory findings MUST be labelled as such and MUST NOT be promoted to confirmatory findings after inspecting results.

---

## 4. Experimental conditions

Every confirmatory prompt and decoding seed MUST be run under all core conditions in randomized order.

| ID | Condition | Internal intervention | Decoder calibration |
|---|---|---|---|
| `baseline` | Ordinary inference | None | Fixed base sampler |
| `sham` | Instrumented but zero dose | Wrapper active, dose = 0 | Identical to baseline |
| `coherent` | Coherent altered state | Fixed random direction per selected layer and generation | Fixed base sampler |
| `white` | Temporally incoherent control | New random direction for each selected layer and token | Dose matched by teacher-forced KL |
| `temp_match` | Output-randomness control | None | Temperature matched to coherent condition's sampling entropy |

Optional exploratory conditions:

| ID | Description |
|---|---|
| `logit_noise_match` | Gaussian logit noise matched to coherent output entropy |
| `creative_prompt` | Baseline model explicitly prompted to be unusually associative |
| `pulse` | Intervention active only during a fixed token window |
| `ramp` | Fixed onset, peak, and decay envelope |
| `global_only` | Perturb only full/global-attention layers |
| `local_only` | Perturb a matched count of sliding/local-attention layers |

The `sham` condition is mandatory during software validation. It may be omitted from the large confirmatory run only after it has demonstrated exact or tolerance-bounded equivalence to `baseline` across all preflight prompts and at least 100 generation steps.

---

## 5. Repository layout

The coding agent MUST produce at least the following structure:

```text
lspe/
├── SPEC.md
├── README.md
├── pyproject.toml
├── uv.lock
├── LICENSE
├── configs/
│   ├── smoke.gemma4-e4b.yaml
│   ├── pilot.gemma4-e4b.yaml
│   ├── confirm.gemma4-e4b.yaml
│   ├── replicate.gemma4-e2b.yaml
│   └── fallback.qwen3-4b.yaml
├── data/
│   ├── source/
│   ├── calibration.jsonl
│   ├── pilot.jsonl
│   ├── confirm.jsonl
│   ├── controls.jsonl
│   └── vocabulary.txt
├── schemas/
│   ├── config.schema.json
│   ├── prompt.schema.json
│   ├── generation.schema.json
│   ├── manifest.schema.json
│   └── report.schema.json
├── src/lspe/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── doctor.py
│   ├── locking.py
│   ├── hashing.py
│   ├── rng.py
│   ├── run_store.py
│   ├── models/
│   │   ├── base.py
│   │   ├── mlx_gemma4.py
│   │   └── mlx_qwen3.py
│   ├── interventions/
│   │   ├── base.py
│   │   ├── spherical.py
│   │   ├── additive.py
│   │   └── controller.py
│   ├── generation/
│   │   ├── sampler.py
│   │   ├── loop.py
│   │   └── telemetry.py
│   ├── calibration/
│   │   ├── dose.py
│   │   ├── entropy.py
│   │   └── layer_scan.py
│   ├── tasks/
│   │   ├── loader.py
│   │   ├── divergent_words.py
│   │   ├── alternative_uses.py
│   │   ├── cross_domain.py
│   │   ├── constrained.py
│   │   ├── arithmetic.py
│   │   └── code_tasks.py
│   ├── metrics/
│   │   ├── deterministic.py
│   │   ├── embeddings.py
│   │   ├── degeneration.py
│   │   ├── judge.py
│   │   └── internal.py
│   ├── analysis/
│   │   ├── bootstrap.py
│   │   ├── tests.py
│   │   ├── effects.py
│   │   └── status.py
│   └── reporting/
│       ├── build.py
│       ├── templates/
│       └── plots.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── runs/
    └── .gitkeep
```

Use `uv` for dependency management and lock the environment. Prefer Pydantic for schemas, Typer for the CLI, PyArrow/Parquet for metric tables, NumPy/SciPy for analysis, and Matplotlib plus Jinja for self-contained reports. Do not require a server, database, Docker, or cloud service.

---

## 6. Command-line contract

The completed implementation MUST support:

```bash
uv sync --frozen

uv run lspe doctor --config configs/smoke.gemma4-e4b.yaml
uv run lspe fetch --config configs/smoke.gemma4-e4b.yaml
uv run lspe preflight --config configs/smoke.gemma4-e4b.yaml
uv run lspe calibrate --config configs/pilot.gemma4-e4b.yaml
uv run lspe pilot --config configs/pilot.gemma4-e4b.yaml
uv run lspe freeze --config configs/confirm.gemma4-e4b.yaml \
  --pilot-run runs/<pilot-run-id>
uv run lspe run --lock experiment.lock.yaml
uv run lspe score --run runs/<run-id>
uv run lspe analyze --run runs/<run-id>
uv run lspe report --run runs/<run-id>
uv run lspe verify --run runs/<run-id> --level artifact
uv run lspe verify --run runs/<run-id> --level replay --sample 20
```

Additional required behaviour:

- `--dry-run` prints the complete execution matrix and expected output count without loading a model.
- `--resume` skips already completed content-addressed generation IDs.
- `--fail-fast` stops on the first invalid model state or non-finite tensor.
- `--offline` prohibits network access and succeeds only with cached dependencies and models.
- Every command emits human-readable logs and structured JSON events.
- Commands return non-zero exit status on validation failure.

---

## 7. Configuration contract

A complete experiment is defined by one YAML file plus immutable referenced data files. Configuration validation MUST reject unknown keys.

Example:

```yaml
schema_version: 1
experiment:
  name: gemma4-e4b-coherent-residual
  phase: pilot
  master_seed: 721984
  output_root: runs

hardware:
  expected_platform: darwin-arm64
  minimum_memory_gb: 24
  memory_soft_limit_fraction: 0.80
  memory_hard_limit_fraction: 0.92

model:
  adapter: mlx_gemma4
  repo_id: mlx-community/gemma-4-e4b-it-4bit
  revision: null          # fetch resolves this to an immutable commit
  local_path: null
  trust_remote_code: false
  text_only: true
  thinking: false
  speculative_decoding: false
  kv_cache_quantization: false

prompting:
  system: >-
    Follow the requested output format exactly. Do not discuss this experiment,
    your internal state, or the generation process.
  use_model_chat_template: true
  max_prompt_tokens: 2048

sampling:
  temperature: 0.80
  top_k: 64
  top_p: 1.0
  repetition_penalty: 1.0
  max_new_tokens: 192
  stop_on_eos: true
  store_top_logprobs: 64

intervention:
  site: post_decoder_layer
  kernel: spherical_rotation
  timing: decode_only
  direction_mode: coherent_per_layer
  selected_layers: auto
  target_kl_nats: 0.10
  raw_dose_grid: [0.0, 0.003, 0.01, 0.03, 0.10, 0.30]
  preserve_norm: true
  group_scale: inverse_sqrt_count

conditions:
  - baseline
  - sham
  - coherent
  - white
  - temp_match

data:
  calibration: data/calibration.jsonl
  pilot: data/pilot.jsonl
  confirm: data/confirm.jsonl
  controls: data/controls.jsonl

execution:
  generations_per_prompt: 3
  batch_size: 1
  randomized_condition_order: true
  save_every: 1
  flush_every: 1

scoring:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  embedding_revision: null
  local_judge_model: mlx-community/Qwen3-4B-Instruct-2507-4bit
  judge_enabled: true
  human_review_export: true

statistics:
  bootstrap_samples: 10000
  confidence_level: 0.95
  familywise_method: holm
  validity_noninferiority_margin_pp: 5.0
  degeneration_margin_pp: 2.0
```

The `freeze` command MUST replace all nullable revisions with immutable commits, calculate hashes for all data/config files, select exactly one primary layer group and dose from the pilot according to Section 18, and write `experiment.lock.yaml`. The locked file MUST be sufficient to reproduce the confirmatory run.

---

## 8. Mandatory model-integrity preflight

No pilot or confirmatory run may begin until all applicable checks pass.

### 8.1 Environment checks

Record:

- macOS version and build;
- hardware model, chip, core count, and physical/unified memory;
- Python implementation and version;
- all installed package versions and hashes;
- MLX default device;
- Git commit and dirty-tree state;
- available disk space;
- current memory pressure;
- model repository, immutable revision, local path, and every weight-file SHA-256.

### 8.2 Architecture introspection

The adapter MUST discover rather than hardcode:

- decoder layer path;
- number of layers;
- hidden width;
- layer type for every layer;
- vocabulary size;
- whether per-layer inputs/embeddings exist;
- cache type and cache count;
- final normalization and output-head path.

For Gemma 4 through current MLX-VLM, the expected layer path is structurally equivalent to:

```python
model.language_model.model.layers
```

The implementation MUST still verify this at runtime and fail with an actionable error if upstream structure changes.

Write the discovered architecture to:

```text
runs/<run-id>/architecture.json
```

### 8.3 Baseline sanity prompts

Run deterministic greedy generation for fixed prompts covering:

- a trivial factual answer;
- integer arithmetic;
- a JSON-only response;
- a short constrained sentence;
- a 100-token free-form response.

The preflight MUST reject models that:

- emit obvious control-token leakage;
- repeat one phrase beyond the configured degeneration threshold;
- fail all trivial answers;
- produce non-finite logits;
- ignore the chat template;
- begin reasoning-channel output while thinking mode is disabled;
- exceed the memory hard limit.

These prompts are integrity checks, not claims about general benchmark quality.

### 8.4 Zero-dose identity

Wrap the target layers with the intervention mechanism at exactly zero dose.

For at least:

- five prompts;
- a prompt-prefill pass;
- 100 decode steps in total;
- cache and no-cache paths where supported;

verify:

```text
baseline logits ≈ sham logits
baseline greedy token IDs == sham greedy token IDs
```

The default maximum absolute logit tolerance is `1e-6`, configurable only with a documented reason. Exact equality is preferred. A zero-dose mismatch is fatal.

### 8.5 Cache equivalence

For a fixed short token sequence, compare logits produced by:

1. one full forward pass;
2. prefill plus token-by-token cached decoding.

Compare the positions whose semantics are equivalent. Record maximum and mean absolute error. Failure beyond tolerance is fatal because the experiment relies on decode-time intervention.

### 8.6 Intervention liveness

At a small non-zero dose, verify that:

- the selected hidden state changes;
- output logits change;
- all values remain finite;
- norm preservation holds within tolerance;
- removing the wrapper restores the original output;
- repeated execution with identical seeds is reproducible;
- changing only the intervention seed changes the intervention direction.

### 8.7 Known compatibility failures

The preflight MUST detect and provide explicit diagnostics for:

- extra or missing Gemma KV-sharing parameters;
- incorrect strict checkpoint loading;
- absent or incorrectly applied chat templates;
- malformed reasoning-channel tokens;
- degenerate repetitive output after quantized loading;
- architecture paths changed by an MLX-VLM update.

Do not continue with `strict=False` merely because generation appears to work. Validate every ignored key against an architecture-specific allow-list and include the list in the manifest.

---

## 9. Model adapter interface

All runtime-specific code MUST live behind an interface conceptually equivalent to:

```python
class ModelAdapter(Protocol):
    def load(self, spec: ModelSpec) -> None: ...
    def unload(self) -> None: ...
    def format_prompt(self, messages: list[Message]) -> list[int]: ...
    def architecture(self) -> ArchitectureInfo: ...
    def wrap_layers(self, controller: InterventionController) -> None: ...
    def unwrap_layers(self) -> None: ...
    def make_cache(self) -> Any: ...
    def forward(self, token_ids, cache=None) -> ForwardResult: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
```

`ForwardResult` MUST expose logits and permit capture of selected hidden-state summaries. The core experiment, calibration, scoring, storage, and statistics modules MUST not import MLX-VLM implementation classes directly.

### 9.1 Gemma layer wrapper

The adapter SHOULD wrap each selected decoder layer rather than fork MLX-VLM. Conceptual shape:

```python
class InstrumentedLayer(nn.Module):
    def __init__(self, base, index, controller):
        super().__init__()
        self.base = base
        self.index = index
        self.controller = controller
        self.layer_type = base.layer_type

    def __call__(
        self,
        x,
        mask=None,
        cache=None,
        per_layer_input=None,
        shared_kv=None,
        offset=None,
    ):
        h, kv, next_offset = self.base(
            x,
            mask,
            cache,
            per_layer_input=per_layer_input,
            shared_kv=shared_kv,
            offset=offset,
        )
        h = self.controller.apply_post_layer(self.index, h)
        return h, kv, next_offset
```

The real implementation MUST preserve all attributes upstream code reads and MUST be covered by zero-dose and cache-equivalence tests. Do not edit the model's quantized weights.

### 9.2 Text-only loading

For Gemma 4, use the official chat template through MLX-VLM prompt helpers and explicitly set thinking disabled. Raw user strings MUST NOT be passed directly to an instruction-tuned checkpoint.

The exact rendered prompt text and token IDs MUST be stored for each unique prompt hash.

---

## 10. Intervention kernel

### 10.1 Primary kernel: norm-preserving spherical rotation

For hidden vector `h` at one token and a fixed random direction `r`:

1. Normalize `h` to unit length: `u = h / ||h||`.
2. Orthogonalize `r` against `u`: `v_raw = r - dot(r, u) * u`.
3. Normalize the orthogonal component: `v = v_raw / ||v_raw||`.
4. Rotate by angular dose `theta`:

```text
h' = ||h|| * (cos(theta) * u + sin(theta) * v)
```

Properties:

- exact L2-norm preservation apart from floating-point error;
- no mutation of weights or persistent model state;
- dose has a geometrically interpretable meaning;
- a fixed `r` creates a coherent bias throughout one generation;
- it changes activation direction without globally increasing activation magnitude.

Perform dot products, norms, trigonometry, and normalization in float32, then cast the result back to the incoming activation dtype. Use an epsilon guard for near-zero `h` or `v_raw`. Such events MUST be counted and reported.

### 10.2 Secondary kernel: RMS-scaled additive perturbation

Implement for comparison and prior-art compatibility:

```text
raw = h + alpha * RMS(h) * unit_rms(r)
h'  = raw * RMS(h) / max(RMS(raw), epsilon)
```

This kernel is exploratory unless explicitly selected before the confirmatory lock.

### 10.3 Direction modes

Implement:

- `coherent_per_layer`: one independent random direction per selected layer and generation, fixed for all generated tokens;
- `coherent_shared`: one direction shared across compatible selected layers;
- `white_per_token`: a new direction for each selected layer and generated token;
- `zero`: no mathematical change while the wrapper and telemetry remain active.

The primary coherent condition uses `coherent_per_layer`. The primary white-noise control uses `white_per_token`.

### 10.4 RNG separation

Derive independent seeds from the master seed using a cryptographic hash over explicit domains:

```text
prompt-order
condition-order
intervention-direction
sampling-token
bootstrap
judge-order
human-review-order
```

A change in one domain MUST NOT alter another. Never rely on incidental global RNG state.

A direction seed SHOULD be derived from:

```text
hash(master_seed, run_id, prompt_id, generation_index,
     condition_id, layer_index, "direction")
```

Sampling keys SHOULD be derived from:

```text
hash(master_seed, prompt_id, generation_index,
     token_index, "sampling")
```

Use the same sampling-key schedule across paired conditions. Different token histories will still diverge, but paired random keys reduce an avoidable source of variance.

### 10.5 Timing

The primary experiment uses `decode_only`:

1. Process all prompt tokens except the final token with the intervention disabled.
2. Enable the intervention.
3. Process the final prompt token so the first generated-token distribution is altered.
4. Keep the intervention active for subsequent decode tokens.

This avoids rewriting the prompt's cached representation while still affecting every generated token.

Implement but do not make confirmatory by default:

- `prefill_and_decode`;
- `token_window(start, end)`;
- `ramp`;
- `onset_peak_decay`;
- `intervene_then_recover`.

### 10.6 Multi-layer scaling

If `k` layers are perturbed together, initialize each raw layer dose as:

```text
theta_layer = theta_group / sqrt(k)
```

Then recalibrate the complete group against target output KL. Do not assume independent layer effects.

---

## 11. Layer selection

### 11.1 Discovery

Build a layer table from runtime introspection:

```text
index | normalized depth | upstream layer type | local/global | KV-sharing role
```

Never assume a fixed global-attention pattern. Record what the loaded model actually declares.

### 11.2 Sentinel pilot

The initial layer scan SHOULD include:

- one early layer near 15% depth;
- one early-middle layer near 35%;
- one middle layer near 50%;
- one late-middle layer near 70%;
- one late layer near 90%;
- every global/full-attention layer if there are few enough;
- matched local/sliding layers near global candidates.

### 11.3 Candidate groups

Construct these groups from discovered metadata:

- `early_third`;
- `middle_third`;
- `late_third`;
- `global_attention`;
- `local_attention_matched`;
- `single_best_pilot_layer`.

The pilot selection rule may choose only from groups enumerated before pilot execution.

---

## 12. Dose calibration

Raw angles or additive coefficients are not comparable across layers, groups, models, or quantization levels. Calibrate them by their effect on output distributions.

### 12.1 Teacher-forced calibration corpus

Use a calibration split not reused for pilot selection or confirmation. For every calibration prompt:

1. Generate a deterministic baseline continuation or use a fixed reference continuation.
2. Teacher-force the same prompt and continuation through every candidate condition.
3. At each continuation position, compare the baseline and altered next-token distributions under the identical prefix.
4. Repeat every non-zero calibration point across at least three intervention-direction seeds and aggregate across prompts, positions, and directions.

This avoids comparing distributions generated from already-diverged histories and prevents one unusually effective random direction from defining a dose.

### 12.2 Divergence metric

Compute from the full, untruncated softmax distribution before top-k/top-p sampling filters:

```text
KL(P_altered || P_baseline)
JS(P_altered, P_baseline)
Top-1 agreement
Top-k overlap
```

KL is the primary calibration metric. Store enough logit fingerprints to audit the computation; full logits need not be retained for every position if a full replay can reproduce them.

### 12.3 Target bands

Default target median KL bands:

```text
D0: 0.00 nats
D1: 0.01 nats
D2: 0.03 nats
D3: 0.10 nats
D4: 0.30 nats
```

The pilot SHOULD emphasize D2 and D3. D4 exists mainly to map degeneration and should not automatically be considered a candidate confirmatory dose.

Use monotonic interpolation or bounded search over the raw dose grid. If the response is non-monotonic, retain the empirical curve and choose the smallest raw dose reaching the target band. Report all failures to reach a band.

### 12.4 White-noise matching

Calibrate `white_per_token` independently so its median teacher-forced KL matches the coherent condition's chosen target. Matching raw angle is insufficient.

### 12.5 Entropy-matched temperature control

On the calibration split, find one unperturbed decoder temperature for each coherent candidate that minimizes:

```text
abs(mean post-filter token entropy_temp -
    mean post-filter token entropy_coherent)
```

Use the same top-k/top-p rules in both conditions. Entropy MUST be computed after temperature scaling and truncation, on the actual distribution sampled from.

Use bounded search and report:

- selected temperature;
- target and achieved mean entropy;
- absolute and relative mismatch;
- prompt-level mismatch distribution.

The confirmatory temperature is frozen after pilot selection.

---

## 13. Generation loop and telemetry

Do not call a high-level black-box `generate()` function for experimental runs unless it exposes every required control. Implement a small adapter-owned generation loop based on the pinned runtime's reference logic.

For every generated token, record:

- token index and token ID;
- decoded token fragment;
- selected-token log probability;
- entropy of the final sampling distribution;
- top-1 probability and top-1/top-2 margin;
- top-N token IDs and log probabilities;
- intervention phase and active dose;
- selected layer set;
- hidden-state norm before and after intervention;
- finite-value check;
- generation stop reason.

Do not store full activations by default. Use online summaries and fixed random projections.

### 13.1 Internal-state sketches

For selected audit runs, project hidden states into a fixed 128-dimensional sketch using a seeded random orthogonal or Gaussian projection. Accumulate:

- mean and variance;
- covariance/eigenvalue spectrum in sketch space;
- effective rank/participation ratio;
- cosine distance from teacher-forced baseline;
- downstream recovery by layer depth.

Store the projection seed and matrix hash. Full hidden tensors may be stored only for a small, explicitly configured audit subset.

### 13.2 Degeneration guards

Stop and mark an output when any threshold is crossed:

- repeated 4-gram ratio;
- identical-token run length;
- repeated phrase loop;
- non-finite logits or activations;
- invalid decoder state;
- maximum token limit;
- memory hard limit.

Do not silently discard these outputs. They are scientifically meaningful failures and MUST remain in the dataset with a failure code.

---

## 14. Prompt and task suite

All prompt files use JSON Lines. Each row MUST contain:

```json
{
  "schema_version": 1,
  "prompt_id": "aut-brick-001",
  "split": "confirm",
  "task_type": "alternative_uses",
  "system_variant": "neutral",
  "prompt": "...",
  "response_schema": "alternative_uses.v1",
  "validator": "alternative_uses",
  "expected": null,
  "tags": ["creativity", "structured"]
}
```

Prompt datasets MUST be immutable during a run and content-hashed in the lock file.

### 14.1 Divergent word association

Ask for exactly ten common nouns that are mutually as unrelated as possible, in a strict JSON array.

Deterministic checks:

- JSON parses;
- exactly ten items;
- each item is a short string;
- no duplicates after normalization;
- no explanation or extra fields.

Scores:

- average pairwise embedding cosine distance within each response;
- cross-seed response diversity;
- vocabulary validity rate;
- lexical duplication.

### 14.2 Alternative Uses Task

Use a balanced set of ordinary objects, such as brick, paperclip, towel, bottle, cardboard box, spoon, rubber band, and newspaper. Ask for a fixed number of unusual but physically plausible uses in JSON.

Each idea MUST contain:

```json
{
  "idea": "...",
  "mechanism": "...",
  "feasibility": "..."
}
```

Deterministic checks:

- schema validity;
- exact count;
- duplicate and near-duplicate rate;
- forbidden ordinary-use matches;
- response length bounds.

Secondary judge scores:

- novelty;
- usefulness;
- physical plausibility;
- diversity across ideas.

### 14.3 Cross-domain bridge tasks

Ask the model to connect two distant concepts under a strict schema:

```json
{
  "bridge": "one-sentence connection",
  "mechanism": ["step 1", "step 2", "step 3"],
  "test": "how the proposed connection could be checked",
  "failure_mode": "why the analogy might be misleading"
}
```

Pair concepts from preregistered domains such as distributed systems, ecology, music, logistics, linguistics, materials science, economics, and game design.

Deterministic validity plus blinded novelty/usefulness judging makes this task especially relevant to useful conceptual recombination.

### 14.4 Constrained creative tasks

Create prompts with machine-checkable constraints, for example:

- exactly four lines;
- exact word-count range;
- specified required words;
- specified forbidden words;
- acrostic or end-word pattern;
- valid JSON with bounded field lengths;
- three solutions with mutually exclusive mechanisms.

These measure whether increased novelty survives constraint pressure.

### 14.5 Convergent controls

Include local, fixed-answer tasks:

- generated integer arithmetic;
- symbolic transformations;
- simple logic;
- factual items stored with answers in the repository;
- JSON-schema following;
- small pure-function code tasks with executable tests.

Generated code MUST run in a restricted subprocess with:

- no network;
- temporary working directory;
- CPU and wall-clock limits;
- memory limit where supported;
- import allow-list;
- no shell execution;
- captured stdout/stderr.

Do not execute code from prompts involving system access, persistence, credentials, exploitation, or destructive operations.

### 14.6 Optional set-and-setting suite

For a small exploratory subset, run identical user tasks under preregistered system contexts:

- analytical/scientific;
- playful/metaphorical;
- pragmatic/engineering.

Measure whether condition-specific outputs remain separable by context. A useful altered-state analogue should not merely erase instruction conditioning.

---

## 15. Dataset sizes and execution profiles

Provide three default profiles.

### 15.1 Smoke

```text
4 creativity prompts
4 control prompts
2 generation seeds
2 sentinel layers
3 raw doses
```

Purpose: verify plumbing only. Never use smoke results as scientific evidence.

### 15.2 Pilot

```text
24 creativity prompts
16 control prompts
3 generation seeds
all preregistered candidate groups
D1-D4 dose bands
all five core conditions where applicable
```

Purpose: characterize dose response and choose one primary layer group and one dose under a fixed selection rule.

### 15.3 Confirmatory

```text
80 creativity prompts:
  20 divergent-word prompts
  20 alternative-uses prompts
  20 cross-domain bridge prompts
  20 constrained-creative prompts
40 control prompts
5 generation seeds per prompt and condition
5 core conditions
one locked primary layer group
one locked moderate dose
```

The confirmatory run therefore contains a complete paired matrix. Missing cells MUST be reported and rerun where possible; they may not be silently removed.

### 15.4 Replication

At minimum:

```text
40 creativity prompts
20 control prompts
3 generation seeds
baseline, coherent, temp_match, white
same target-KL band and selection logic
```

Use E2B higher precision first; Qwen is the architecture replication if resources permit or Gemma precision control cannot run.

---

## 16. Scoring

### 16.1 Independent embedding model

Use a pinned revision of:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Download and hash it during `fetch`. Run embedding scoring only after unloading the subject model. Normalize embeddings before cosine calculations.

The embedding model MUST never see condition labels. Store raw text-to-embedding content hashes so embeddings can be recomputed and deduplicated.

### 16.2 Primary outcome: valid semantic diversity

For prompt `p`, condition `c`, and `K >= 2` generated responses, let:

- `V_i` be 1 when response `i` passes the task's deterministic validity gate, otherwise 0;
- `e_i` be the normalized embedding of a task-specific canonical semantic payload extracted from the response, excluding JSON keys, fixed boilerplate, and formatting tokens.

Define:

```text
VSD(p,c) = 2 / (K * (K - 1)) *
           sum over i<j of [V_i * V_j * (1 - cosine(e_i, e_j))]
```

Invalid pairs contribute zero. This prevents incoherent or schema-breaking text from receiving a creativity reward merely because it is unusual.

Report separately:

- ungated semantic diversity;
- deterministic validity rate;
- VSD;
- mean response length.

The primary confirmatory contrast is:

```text
mean_p(VSD(p, coherent) - VSD(p, temp_match))
```

### 16.3 Competence composite

For convergent controls, calculate a preregistered macro-average of:

- exact-answer accuracy;
- schema pass rate;
- executable-test pass rate;
- constraint satisfaction rate.

Do not allow one large task category to dominate merely because it contains more rows.

### 16.4 Degeneration metrics

At minimum:

- invalid JSON/schema rate;
- repeated 4-gram ratio;
- distinct-1/2/3;
- self-BLEU or equivalent lexical similarity;
- token-loop rate;
- truncation rate;
- control-token leakage;
- refusal rate;
- non-finite or runtime-failure rate.

### 16.5 Internal metrics

For calibrated audit runs:

- teacher-forced output KL and JS divergence;
- top-1 agreement;
- hidden-state cosine drift by downstream layer;
- sketch-space effective rank;
- activation-norm ratio;
- attention-layer category and normalized depth.

These are explanatory measures, not substitutes for behavioural outcomes.

### 16.6 Blinded local judge

Use the pinned Qwen model as a secondary judge only after unloading Gemma. Present outputs without model, condition, dose, layer, or seed labels.

The judge MUST return strict JSON ratings for:

- novelty;
- usefulness;
- coherence;
- constraint adherence;
- plausibility.

Requirements:

- temperature 0 or deterministic decoding;
- shuffled sample order;
- at least two prompt-order variants for pairwise comparisons;
- rubric and judge prompt stored in the run;
- parse failures retained;
- inter-run agreement reported;
- no judge score used as the sole primary outcome.

### 16.7 Human review export

Produce a condition-blinded HTML/JSON bundle containing a stratified sample. Randomize labels using a separate seed and write the unblinding key to a separate file.

Human review is optional for running the experiment but required before making strong claims about “useful creativity.”

---

## 17. Statistical analysis

### 17.1 Unit of pairing

The prompt is the primary independent cluster. Multiple generations per prompt estimate within-prompt variability and MUST NOT be treated as fully independent samples.

### 17.2 Primary estimate

Calculate the prompt-level paired difference:

```text
Delta_p = VSD(p, coherent) - VSD(p, temp_match)
```

Report:

- mean and median `Delta_p`;
- standardized paired effect size;
- 95% cluster-bootstrap confidence interval by resampling prompts;
- paired sign-flip/permutation p-value;
- number of positive, zero, and negative prompt-level effects.

Use at least 10,000 bootstrap resamples in the confirmatory analysis.

### 17.3 H2 analysis

Compare coherent and white conditions at matched target KL:

- competence composite difference;
- validity-rate difference;
- degeneration-rate difference;
- VSD difference.

Use a non-inferiority margin of five percentage points for deterministic validity unless the lock specifies a stricter margin.

### 17.4 Multiplicity

The primary H1 contrast is singular and preregistered. Apply Holm correction to families of secondary comparisons. Exploratory layer/dose heatmaps MUST show uncorrected effect estimates but clearly state that they are exploratory; do not mark them “significant” without correction.

### 17.5 Missing and failed outputs

- Invalid model outputs remain scored as invalid.
- Infrastructure failures may be rerun with the identical content ID and seeds.
- Persistent infrastructure failures remain missing and are reported.
- Never replace a failed scientific output with a fresh seed.
- Report complete expected-versus-observed cell counts.

### 17.6 No selective reporting

The report MUST include every attempted layer, dose, condition, task family, and predeclared metric. The agent may not delete an unsuccessful condition or redefine the primary outcome after viewing confirmatory results.

---

## 18. Pilot selection and experiment freeze

Before pilot execution, enumerate all candidate layer groups, target KL bands, validity thresholds, and the selection formula.

Default candidate eligibility:

1. median teacher-forced KL lies within ±20% of the target band;
2. deterministic validity is no more than five percentage points below baseline;
3. degeneration rate is no more than two percentage points above baseline;
4. no model-integrity failure occurred.

Among eligible candidates, select the layer-group/dose combination maximizing:

```text
pilot utility =
    VSD advantage over temp_match
  + 0.25 * VSD advantage over white
  - 0.50 * competence loss versus baseline
```

All terms MUST be normalized using scales fixed before pilot execution. Ties are resolved by:

1. lower dose;
2. fewer perturbed layers;
3. lower normalized layer depth;
4. lexical candidate ID.

If no candidate is eligible, freeze a **null/degradation confirmatory run** at the lowest non-zero dose that passed integrity checks, or stop with an explicit `NO_ELIGIBLE_INTERVENTION` result. Do not relax thresholds after inspection.

The `freeze` command writes:

- selected model and immutable revision;
- dependency lock hash;
- prompt/data hashes;
- selected layer indices and discovered types;
- intervention kernel and direction mode;
- raw dose and achieved target KL;
- matched temperature;
- all generation and analysis seeds;
- primary hypothesis and metric definition;
- statistical thresholds;
- Git commit;
- pilot run ID and selection calculation.

The lock file is immutable. Any modification creates a new experiment ID.

---

## 19. Result classification

The software MUST produce one of these scientific statuses without embellishment.

### `SUPPORTED`

All are true:

1. H1 mean paired effect is positive and its preregistered 95% confidence interval excludes zero.
2. Coherent validity is within the non-inferiority margin of `temp_match` and baseline.
3. Coherent degeneration is within the preregistered margin.
4. At matched KL, coherent VSD exceeds white-noise VSD, and coherent competence is not worse than white-noise competence beyond the locked non-inferiority margin.
5. The H1 effect direction is positive in the precision or architecture replication control.

### `PROMISING`

The primary estimate is positive and validity is retained, but uncertainty includes zero, replication is absent, or one secondary criterion fails.

### `NOT_SUPPORTED`

The primary estimate is zero/negative, entropy matching explains the effect, or replication reverses it without a clear methodological reason.

### `DEGENERATIVE`

Semantic or lexical diversity rises primarily because validity, competence, coherence, or degeneration worsens beyond the preregistered margin.

### `INVALID_RUN`

Model integrity, environment locking, data completeness, seed reproducibility, or analysis verification failed.

A null result is a successful execution of the experiment when the software and verification criteria pass.

---

## 20. Run storage and provenance

Each run directory MUST contain:

```text
runs/<run-id>/
├── manifest.json
├── architecture.json
├── resolved-config.yaml
├── experiment.lock.yaml        # confirmatory runs
├── environment.txt
├── packages.lock.json
├── model-files.json
├── prompts.snapshot.jsonl
├── prompt-renders.jsonl
├── generation-plan.jsonl
├── generations.jsonl
├── token-metrics.parquet
├── calibration.parquet
├── scores.parquet
├── prompt-effects.parquet
├── analysis.json
├── report.json
├── report.md
├── report.html
├── checksums.sha256
├── logs/
├── plots/
└── human-review/
```

### 20.1 Generation records

One append-only JSONL row per generation, including:

- content-addressed generation ID;
- prompt ID and prompt hash;
- rendered token hash;
- model/revision/quantization;
- condition;
- layer indices/types;
- kernel/dose/direction mode;
- intervention and decoding seeds;
- sampler parameters;
- output token IDs and decoded text;
- stop reason;
- deterministic validator result;
- runtime duration and memory peak;
- error/failure code;
- software commit and config hash.

### 20.2 Atomicity and resume

Write each generation to a temporary file or journal entry, flush and fsync, then atomically commit it. The content ID MUST be derived from all scientific inputs. `--resume` may only reuse a row whose content ID and checksum match.

### 20.3 Checksums

Generate SHA-256 checksums for every immutable artifact. Exclude only ephemeral logs and the checksum file itself. Calculate and store a root digest over sorted `(relative_path, sha256)` pairs.

---

## 21. Reports

Generate both a human-readable and machine-readable report.

### 21.1 Required report sections

1. Title, run ID, date, and scientific status.
2. Plain-language question and non-claims.
3. Hardware, model, quantization, runtime, and exact revisions.
4. Model-integrity preflight results.
5. Architecture and selected intervention sites.
6. Calibration curves and entropy matching.
7. Dataset composition and complete execution counts.
8. Primary H1 result with effect size and confidence interval.
9. H2 competence/degeneration comparison.
10. Replication result.
11. Full secondary and exploratory results.
12. Dose-response and layer-sensitivity analysis.
13. Representative condition-blinded outputs, including failures.
14. Limitations and alternative explanations.
15. Reproduction and verification commands.
16. Complete artifact/checksum inventory.

### 21.2 Required plots

- dose versus teacher-forced KL;
- dose versus VSD;
- dose versus validity/competence;
- layer × dose heatmap;
- semantic diversity versus validity Pareto plot;
- coherent versus entropy-matched prompt-level effect distribution;
- entropy versus VSD scatter;
- coherent versus white competence comparison;
- downstream hidden-state drift/recovery by layer;
- replication effect-size forest plot.

Every plot MUST have an adjacent machine-readable table.

### 21.3 Machine-readable report

`report.json` MUST validate against `schemas/report.schema.json` and include:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "status": "SUPPORTED|PROMISING|NOT_SUPPORTED|DEGENERATIVE|INVALID_RUN",
  "primary": {
    "metric": "valid_semantic_diversity",
    "contrast": "coherent-temp_match",
    "estimate": 0.0,
    "ci95": [0.0, 0.0],
    "p_value": 0.0,
    "n_prompts": 0
  },
  "validity": {},
  "competence": {},
  "degeneration": {},
  "replication": {},
  "integrity": {},
  "artifact_root_hash": "..."
}
```

Do not fill absent values with invented zeros; use `null` and an explanatory status.

---

## 22. Verification levels

### 22.1 Artifact verification

```bash
uv run lspe verify --run runs/<run-id> --level artifact
```

MUST work without loading the subject model and MUST:

- verify all checksums and schemas;
- rebuild deterministic validators from raw generations;
- rebuild embedding and statistical metrics from pinned cached models;
- regenerate `analysis.json` and compare within exact or declared numeric tolerances;
- verify the report is consistent with analysis artifacts;
- verify expected/observed generation counts;
- emit PASS/FAIL with reasons.

### 22.2 Replay verification

```bash
uv run lspe verify --run runs/<run-id> --level replay --sample 20
```

MUST load the locked subject model and replay a deterministic stratified subset, including:

- baseline greedy outputs;
- sham equality;
- intervention direction fingerprints;
- selected calibration KL values;
- fixed-seed sampled generations where backend determinism permits.

If exact sampled replay is not stable across hardware/runtime versions, verify top-token/logit fingerprints under the original locked environment and report the limitation. Never silently loosen tolerances.

### 22.3 Full verification

An optional `--level full` reruns the complete generation matrix from the lock into a new run directory and compares results statistically and, where deterministic, byte-for-byte.

---

## 23. Required tests

At minimum, implement:

```text
test_config_rejects_unknown_keys
test_content_id_changes_for_every_scientific_input
test_rng_domains_are_independent
test_direction_reproducibility
test_direction_changes_with_seed
test_spherical_rotation_preserves_norm
test_additive_zero_dose_identity
test_layer_wrapper_zero_dose_logits
test_wrap_unwrap_restores_model
test_decode_only_does_not_modify_early_prefill
test_cache_equivalence
test_nonfinite_intervention_is_fatal
test_entropy_computation_after_filtering
test_temperature_matcher_converges
test_teacher_forced_kl_uses_identical_prefixes
test_white_noise_uses_new_direction_per_token
test_coherent_mode_reuses_direction
test_generation_plan_is_complete_and_paired
test_resume_is_idempotent
test_failed_scientific_output_is_not_reseeded
test_condition_labels_are_blinded_for_judge
test_invalid_outputs_contribute_zero_to_vsd_pairs
test_cluster_bootstrap_resamples_prompts
test_holm_adjustment
test_report_status_rules
test_checksums_detect_mutation
test_artifact_verification_rebuilds_analysis
```

Also provide one slow integration test for each adapter and one end-to-end smoke test that creates a valid report directory.

Tests requiring downloaded model weights MUST be marked and skipped with a clear reason when the model is absent. CI should run all non-model tests.

---

## 24. Resource and safety constraints

- Batch size is 1 by default.
- Prompt length is capped at 2,048 tokens for version 1.
- Generated length is capped at 192 tokens for creativity tasks and 128 for controls unless locked otherwise.
- Full activations are never retained for the whole run.
- Subject model, embedding model, and judge model are loaded sequentially, not concurrently.
- Call the runtime cache-clearing API and verify memory release before loading the next model.
- The harness watches process memory and system memory pressure.
- At the soft limit, stop scheduling new generations and flush state.
- At the hard limit, terminate the current generation safely, record `MEMORY_LIMIT`, unload the model, and exit non-zero.
- Model generations have no tools, filesystem access, shell access, network access, or external side effects.
- Judge and subject prompts are benign. Safety-alignment degradation testing is outside version 1 and would require a separate protocol.

---

## 25. Implementation sequence for the coding agent

The agent MUST implement in this order and run tests after each stage.

### Stage 1 — Scaffold and contracts

- repository structure;
- Pydantic config and artifact schemas;
- hashing, content IDs, run directory, JSON event logging;
- CLI skeleton and `doctor`;
- deterministic RNG derivation;
- unit tests.

**Exit condition:** configuration, hashing, and resume tests pass without a model.

### Stage 2 — Qwen reference adapter

- implement the simpler Qwen MLX adapter first;
- prompt formatting;
- architecture discovery;
- custom generation loop;
- token telemetry;
- baseline/sham preflight.

**Exit condition:** Qwen smoke generation is reproducible and zero-dose identity passes.

### Stage 3 — Gemma 4 adapter

- MLX-VLM loading;
- official chat-template application with thinking disabled;
- text-only token path;
- hybrid layer discovery;
- KV-sharing-aware wrapper;
- strict ignored-key policy;
- cache-equivalence and integrity diagnostics.

**Exit condition:** Gemma preflight passes or exits with an explicit compatibility diagnosis and Qwen fallback remains usable.

### Stage 4 — Intervention controller

- spherical kernel;
- additive kernel;
- coherent and white modes;
- timing controller;
- per-token telemetry;
- norm and RNG tests.

**Exit condition:** deterministic liveness, norm preservation, and wrap/unwrap tests pass.

### Stage 5 — Calibration

- teacher-forced corpus path;
- exact KL/JS metrics;
- raw-dose sweep;
- layer/group scan;
- white-noise KL matching;
- temperature entropy matching;
- calibration reports.

**Exit condition:** smoke calibration selects doses reproducibly from synthetic fixture logits and a real model.

### Stage 6 — Tasks and deterministic scoring

- JSONL datasets;
- validators;
- embedding scorer;
- VSD;
- competence and degeneration metrics;
- code-task sandbox.

**Exit condition:** fixture outputs produce known expected scores.

### Stage 7 — Pilot, freeze, and confirmatory runner

- randomized paired execution plan;
- append-only store;
- resume;
- pilot eligibility and selection formula;
- immutable lock generation;
- confirmatory run enforcement.

**Exit condition:** the agent cannot run `phase=confirm` without a valid lock.

### Stage 8 — Statistics and reporting

- prompt-cluster bootstrap;
- paired permutation test;
- multiplicity correction;
- status classification;
- plots and adjacent data;
- Markdown, HTML, and JSON reports;
- artifact and replay verification.

**Exit condition:** a complete smoke run verifies from raw artifacts and regenerates the same report.

### Stage 9 — Local experiment execution

Execute in order:

```text
doctor
fetch
preflight
smoke
calibrate
pilot
freeze
confirmatory run
score
analyze
report
artifact verify
replay verify
precision replication
final combined report
```

Do not skip a failed stage. Fix software failures and create a new run ID; preserve failed run directories.

---

## 26. Software acceptance criteria

Implementation is complete when:

1. All non-model unit tests pass.
2. Qwen and Gemma adapters either pass preflight or fail with exact actionable diagnostics.
3. The chosen primary adapter passes zero-dose identity and cache-equivalence checks.
4. A smoke run can be interrupted and resumed without duplicate or changed rows.
5. Calibration produces reproducible target-KL doses and matched temperatures.
6. A pilot produces an immutable confirmatory lock through the declared selection rule.
7. A confirmatory run cannot alter locked scientific parameters.
8. Raw outputs, deterministic scores, statistics, plots, and all report formats are generated.
9. Artifact verification passes from the raw data.
10. Replay verification passes on the required audit sample.
11. A negative synthetic fixture is classified `NOT_SUPPORTED`.
12. A high-diversity but invalid fixture is classified `DEGENERATIVE`.
13. No report calls the model conscious, intoxicated, psychedelic, or biologically equivalent to a human altered state.

Scientific support for the hypothesis is not an implementation acceptance criterion.

---

## 27. Recommended first run

Use this conservative sequence on the 32 GB M2 MacBook:

```text
Primary smoke and pilot:
  mlx-community/gemma-4-e4b-it-4bit
  text only
  thinking off
  batch size 1
  <= 2K prompt tokens
  <= 192 output tokens

Precision replication:
  Gemma 4 E2B BF16 if preflight and memory checks pass
  otherwise Gemma 4 E2B 8-bit

Architecture replication/fallback:
  mlx-community/Qwen3-4B-Instruct-2507-4bit
```

Do not begin with a larger model. Broad, paired sweeps and clean controls matter more than maximum parameter count.

---

## 28. Phase-two extensions

Only after version 1 produces a verified baseline:

1. **Attention-temperature intervention:** flatten or sharpen attention logits internally while keeping output sampling fixed.
2. **Head-output mixing:** apply a near-identity norm-preserving mixing matrix across attention heads.
3. **Per-layer embedding intervention:** perturb Gemma 4 PLE inputs separately from the residual stream.
4. **Cross-layer mixing:** transiently mix compatible residual features between nearby layers.
5. **Low-rank weight perturbation:** apply reversible low-rank deltas during inference, never mutating source checkpoints.
6. **State persistence:** perturb for a token window, turn the intervention off, and measure continuation effects.
7. **Selection and consolidation:** have perturbed runs generate candidates, use sober validators to select them, and train a small LoRA only on validated gains.
8. **Base-versus-instruct comparison:** test whether post-training changes sensitivity to perturbation.
9. **Training-checkpoint study:** use a checkpoint-rich model family to identify when structured sensitivity emerges during training.
10. **Multimodal study:** test whether image/audio grounding suppresses or reshapes the intervention, under a separate protocol.

Each extension requires its own preregistration and controls. Do not fold phase-two experiments into the first confirmatory report.

---

## 29. Reference basis

The implementation and report SHOULD cite these sources by stable title and identifier:

- Google, **Gemma 4 model overview**.
- Google, **Gemma 4 model card**.
- MLX-VLM, **Gemma 4 model implementation and usage documentation**.
- MLX-LM / MLX-VLM issue histories concerning Gemma 4 KV-sharing and checkpoint-loading compatibility.
- Turner et al., **Steering Language Models With Activation Engineering**, arXiv:2308.10248.

Source revisions used to implement adapter-specific behaviour MUST be recorded in the repository and run manifests.

---

## 30. Definition of done

The project is done when another person with the locked model files and a compatible Apple-silicon Mac can:

```bash
uv sync --frozen
uv run lspe verify --run <copied-run> --level artifact
uv run lspe verify --run <copied-run> --level replay --sample 20
```

and independently confirm:

- what model and software were used;
- exactly what intervention was applied and where;
- that zero dose is behaviourally identical to baseline;
- that dose and temperature controls were genuinely matched;
- that all prompts, seeds, failures, and attempted conditions are present;
- that reported metrics and confidence intervals follow from raw outputs;
- that positive, null, or degenerative status follows mechanically from preregistered rules.

The desired deliverable is not a collection of amusing generations. It is a small local experimental platform capable of producing an auditable answer to whether coherent internal perturbation creates a useful behavioural regime beyond ordinary sampling randomness.
