import numpy as np

from lspe.networks.selective_connectivity import (
    SelectiveConnectivityProtocol,
    build_selective_transforms,
    calibration_eligible,
    candidate_masks,
    rank_selective_modes,
    select_calibration_candidate,
)


def test_mode_ranking_requires_target_selectivity_and_is_deterministic() -> None:
    rows = [
        {
            "layer": 16,
            "eigenvalue_rank": 2,
            "target_median_output_kl": 0.03,
            "protection_median_output_kl": 0.01,
        },
        {
            "layer": 15,
            "eigenvalue_rank": 1,
            "target_median_output_kl": 0.02,
            "protection_median_output_kl": 0.01,
        },
        {
            "layer": 17,
            "eigenvalue_rank": 3,
            "target_median_output_kl": 0.01,
            "protection_median_output_kl": 0.02,
        },
    ]
    ranked = rank_selective_modes(rows)
    assert [(row["layer"], row["eigenvalue_rank"]) for row in ranked] == [(16, 2), (15, 1)]


def test_candidate_masks_require_complete_prefix() -> None:
    ranked = [{"layer": 15, "eigenvalue_rank": rank} for rank in range(10)]
    assert [row["mask_size"] for row in candidate_masks(ranked, (8, 16))] == [8]


def test_selective_transforms_group_modes_by_layer() -> None:
    correlations = np.repeat(np.eye(4)[None, :, :], 20, axis=0)
    transforms = build_selective_transforms(
        correlations,
        [{"layer": 15, "eigenvalue_rank": 0}, {"layer": 15, "eigenvalue_rank": 2}],
        0.4,
    )
    assert set(transforms) == {15}
    assert np.allclose(transforms[15], np.eye(4))


def test_calibration_selection_is_competence_first() -> None:
    rows = [
        {
            "protection_mean_top1_agreement": 0.84,
            "target_median_output_kl": 0.02,
            "protection_median_output_kl": 0.01,
            "mask_size": 16,
            "alpha": 0.4,
        },
        {
            "protection_mean_top1_agreement": 0.86,
            "target_median_output_kl": 0.015,
            "protection_median_output_kl": 0.01,
            "mask_size": 32,
            "alpha": 0.5,
        },
    ]
    assert select_calibration_candidate(rows) is rows[1]


def test_calibration_eligibility_requires_group_protection() -> None:
    protocol = SelectiveConnectivityProtocol()
    summary = {
        "median_correlation_change": -0.2,
        "median_effective_rank_change": 0.2,
        "median_output_kl": 0.02,
        "mean_top1_agreement": 0.84,
        "protection_mean_top1_agreement": 0.83,
        "target_median_output_kl": 0.03,
        "protection_median_output_kl": 0.01,
        "invariants": {
            "maximum_mean_error": 1e-7,
            "maximum_scale_error": 1e-7,
            "nonfinite_count": 0,
            "zero_variance_count": 0,
        },
    }
    assert calibration_eligible(summary, protocol)
    summary["protection_mean_top1_agreement"] = 0.81
    assert not calibration_eligible(summary, protocol)
