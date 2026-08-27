"""Adapter-owned autoregressive loop; no opaque high-level generation call."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from ..config import SamplingConfig
from ..rng import derive_seed
from .sampler import sample_token
from .telemetry import TokenTelemetry


@dataclass(frozen=True)
class GenerationResult:
    output_token_ids: tuple[int, ...]
    text: str
    stop_reason: str
    token_metrics: tuple[TokenTelemetry, ...]
    duration_seconds: float


class GenerationLoop:
    """Minimal cached decoding loop shared by model adapters and verification."""

    def __init__(self, adapter: Any, sampling: SamplingConfig, master_seed: int) -> None:
        self.adapter = adapter
        self.sampling = sampling
        self.master_seed = master_seed

    def generate(
        self,
        prompt_token_ids: list[int],
        prompt_id: str,
        generation_index: int,
        condition: str,
        selected_layers: tuple[int, ...] = (),
        intervention_active: bool = False,
        intervention_dose: float = 0.0,
    ) -> GenerationResult:
        """Prefill all but final prompt token, then decode one token at a time."""

        if not prompt_token_ids:
            raise ValueError("Cannot generate from an empty formatted prompt")
        started = perf_counter()
        cache = self.adapter.make_cache()
        if len(prompt_token_ids) > 1:
            self.adapter.forward(prompt_token_ids[:-1], cache=cache)
        next_input = [prompt_token_ids[-1]]
        output_ids: list[int] = []
        records: list[TokenTelemetry] = []
        stop_reason = "MAX_TOKENS"
        for token_index in range(self.sampling.max_new_tokens):
            result = self.adapter.forward(next_input, cache=cache)
            logits = np.asarray(result.logits)
            if logits.ndim < 1 or not np.isfinite(logits).all():
                raise FloatingPointError("Model forward returned non-finite logits")
            next_logits = logits.reshape(-1, logits.shape[-1])[-1]
            seed = derive_seed(
                self.master_seed,
                "sampling-token",
                prompt_id,
                generation_index,
                token_index,
            )
            sampled = sample_token(
                next_logits,
                temperature=self.sampling.temperature,
                top_k=self.sampling.top_k,
                top_p=self.sampling.top_p,
                seed=seed,
                store_top_logprobs=self.sampling.store_top_logprobs,
            )
            records.append(
                TokenTelemetry(
                    token_index=token_index,
                    token_id=sampled.token_id,
                    token_fragment=self.adapter.decode([sampled.token_id]),
                    selected_token_log_probability=sampled.log_probability,
                    entropy=sampled.entropy,
                    top1_probability=sampled.top1_probability,
                    top1_top2_margin=sampled.top1_top2_margin,
                    top_token_ids=sampled.top_token_ids,
                    top_log_probabilities=sampled.top_log_probabilities,
                    intervention_active=intervention_active,
                    intervention_dose=intervention_dose if intervention_active else 0.0,
                    selected_layers=selected_layers,
                    finite=True,
                )
            )
            if self.sampling.stop_on_eos and sampled.token_id in self.adapter.eos_token_ids():
                stop_reason = "EOS"
                break
            output_ids.append(sampled.token_id)
            next_input = [sampled.token_id]
        return GenerationResult(
            output_token_ids=tuple(output_ids),
            text=self.adapter.decode(output_ids),
            stop_reason=stop_reason,
            token_metrics=tuple(records),
            duration_seconds=perf_counter() - started,
        )
