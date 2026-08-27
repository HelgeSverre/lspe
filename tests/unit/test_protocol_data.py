from pathlib import Path

from lspe.config import load_config
from lspe.execution import load_phase_prompts


def test_default_protocol_phase_sizes_and_composition() -> None:
    expected = {
        "smoke.gemma4-e4b.yaml": (8, 4, 4),
        "pilot.gemma4-e4b.yaml": (40, 24, 16),
        "pilot.gemma4-e4b-v2.yaml": (40, 24, 16),
        "confirm.gemma4-e4b.yaml": (120, 80, 40),
        "replicate.gemma4-e2b.yaml": (60, 40, 20),
    }
    for filename, (total, creativity, controls) in expected.items():
        config = load_config(Path("configs") / filename)
        prompts = load_phase_prompts(config, config.experiment.phase)
        assert len(prompts) == total
        assert sum("creativity" in prompt.tags for prompt in prompts) == creativity
        assert sum("control" in prompt.tags for prompt in prompts) == controls


def test_control_profiles_include_preregistered_code_tasks() -> None:
    expected_code_counts = {
        "pilot.gemma4-e4b-v2.yaml": 4,
        "confirm.gemma4-e4b.yaml": 10,
        "replicate.gemma4-e2b.yaml": 6,
    }
    for filename, expected_count in expected_code_counts.items():
        config = load_config(Path("configs") / filename)
        prompts = load_phase_prompts(config, config.experiment.phase)
        assert sum(prompt.validator == "python_function" for prompt in prompts) == expected_count
