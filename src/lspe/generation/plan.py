"""Complete, paired, content-addressed generation plans."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from ..config import LspeConfig
from ..hashing import content_id
from ..rng import derive_seed, deterministic_order
from ..tasks.loader import PromptRecord


@dataclass(frozen=True)
class GenerationPlanItem:
    generation_id: str
    ordinal: int
    prompt_id: str
    prompt_hash: str
    generation_index: int
    condition: str
    sampling_seed: int
    condition_order_seed: int
    scientific_inputs: dict[str, object]

    def record(self) -> dict[str, object]:
        return asdict(self)


def build_generation_plan(
    config: LspeConfig,
    prompts: Iterable[PromptRecord],
    model_revision: str,
    selected_layers: list[int],
    raw_dose: float,
    matched_temperature: float | None = None,
    white_raw_dose: float | None = None,
) -> list[GenerationPlanItem]:
    """Enumerate every prompt/seed/condition cell; no conditions are optional."""

    if not model_revision:
        raise ValueError("Generation plan requires an immutable model revision")
    planned: list[GenerationPlanItem] = []
    for prompt in sorted(prompts, key=lambda item: item.prompt_id):
        for generation_index in range(config.execution.generations_per_prompt):
            order_seed = derive_seed(
                config.experiment.master_seed,
                "condition-order",
                prompt.prompt_id,
                generation_index,
            )
            conditions = list(config.conditions)
            if config.execution.randomized_condition_order:
                conditions = deterministic_order(
                    config.experiment.master_seed,
                    "condition-order",
                    conditions,
                    prompt.prompt_id,
                    generation_index,
                )
            for condition in conditions:
                sampling = config.sampling
                if condition == "temp_match" and matched_temperature is not None:
                    sampling = sampling.model_copy(update={"temperature": matched_temperature})
                sampling_seed = derive_seed(
                    config.experiment.master_seed,
                    "sampling-token",
                    prompt.prompt_id,
                    generation_index,
                    0,
                )
                scientific_inputs: dict[str, object] = {
                    "prompt_id": prompt.prompt_id,
                    "prompt_hash": prompt.content_hash,
                    "model_repo": config.model.repo_id,
                    "model_revision": model_revision,
                    "condition": condition,
                    "generation_index": generation_index,
                    "sampling_seed": sampling_seed,
                    "intervention_seed": derive_seed(
                        config.experiment.master_seed,
                        "intervention-direction",
                        prompt.prompt_id,
                        generation_index,
                        condition,
                    ),
                    "selected_layers": selected_layers,
                    "kernel": config.intervention.kernel,
                    "raw_dose": (
                        raw_dose
                        if condition == "coherent"
                        else (
                            white_raw_dose
                            if condition == "white" and white_raw_dose is not None
                            else raw_dose
                        )
                        if condition == "white"
                        else 0.0
                    ),
                    "sampling": sampling.model_dump(mode="json"),
                }
                planned.append(
                    GenerationPlanItem(
                        generation_id=content_id(scientific_inputs),
                        ordinal=len(planned),
                        prompt_id=prompt.prompt_id,
                        prompt_hash=prompt.content_hash,
                        generation_index=generation_index,
                        condition=condition,
                        sampling_seed=sampling_seed,
                        condition_order_seed=order_seed,
                        scientific_inputs=scientific_inputs,
                    )
                )
    generation_ids = {item.generation_id for item in planned}
    if len(generation_ids) != len(planned):
        raise ValueError("Generation plan contains duplicate scientific content IDs")
    return planned
