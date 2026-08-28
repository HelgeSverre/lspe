from lspe.networks.behavioral_runner import (
    BehavioralProtocol,
    _closest_kl,
    summarize_split,
)


def test_closest_kl_uses_log_distance_and_lower_value_tiebreak() -> None:
    rows = [
        {"value": 0.2, "median_output_kl": 0.005},
        {"value": 0.4, "median_output_kl": 0.02},
    ]
    assert _closest_kl(rows, 0.01)["value"] == 0.2


def test_split_summary_enforces_each_protected_category() -> None:
    protocol = BehavioralProtocol(generations_per_prompt=1)
    prompts = [
        {"prompt_id": f"p-{category}", "category": category}
        for category in (
            "open_association",
            "analogical",
            "narrative",
            "constrained",
            "factual",
            "code",
        )
    ]
    rows = []
    for prompt in prompts:
        for condition in (
            "baseline",
            "sham",
            "sccf",
            "random_basis",
            "attn_noise",
            "temp_match",
        ):
            rows.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "generation_index": 0,
                    "category": prompt["category"],
                    "condition": condition,
                    "output_text": "same",
                    "valid": not (prompt["category"] == "constrained" and condition == "sccf"),
                    "degeneration": {
                        "repeated_4gram_ratio": 0.0,
                        "max_identical_run": 1,
                    },
                    "mean_sampling_entropy": 1.0,
                    "controller_invariants": {"passed": True},
                }
            )
    summary = summarize_split(rows, prompts, protocol, "confirm")
    assert not summary["gates"]["constrained_validity"]
    assert not summary["passed"]
