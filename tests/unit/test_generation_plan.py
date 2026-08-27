from lspe.config import load_config
from lspe.generation.plan import build_generation_plan
from lspe.tasks.loader import PromptRecord


def test_generation_plan_is_complete_and_paired() -> None:
    config = load_config("configs/smoke.gemma4-e4b.yaml")
    prompts = [
        PromptRecord(
            schema_version=1,
            prompt_id="p-1",
            split="controls",
            task_type="exact",
            system_variant="neutral",
            prompt="one",
            response_schema="text.v1",
            validator="exact_answer",
            expected="one",
            tags=(),
        ),
        PromptRecord(
            schema_version=1,
            prompt_id="p-2",
            split="controls",
            task_type="exact",
            system_variant="neutral",
            prompt="two",
            response_schema="text.v1",
            validator="exact_answer",
            expected="two",
            tags=(),
        ),
    ]
    plan = build_generation_plan(config, prompts, "commit123", [1], 0.1)
    assert len(plan) == len(prompts) * config.execution.generations_per_prompt * len(
        config.conditions
    )
    assert len({item.generation_id for item in plan}) == len(plan)
    assert {
        item.sampling_seed
        for item in plan
        if item.prompt_id == "p-1" and item.generation_index == 0
    } == {
        next(
            item.sampling_seed
            for item in plan
            if item.prompt_id == "p-1" and item.generation_index == 0
        )
    }
    white_matched = build_generation_plan(
        config, prompts, "commit123", [1], 0.1, white_raw_dose=0.03
    )
    assert {
        item.scientific_inputs["raw_dose"] for item in white_matched if item.condition == "coherent"
    } == {0.1}
    assert {
        item.scientific_inputs["raw_dose"] for item in white_matched if item.condition == "white"
    } == {0.03}
