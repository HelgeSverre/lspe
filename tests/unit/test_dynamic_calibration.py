import numpy as np

from lspe.networks.dynamic_calibration import (
    DynamicCalibrationProtocol,
    _categorical_kl,
    _eligible,
)


def test_categorical_kl_is_zero_for_identical_logits() -> None:
    logits = np.array([1.0, -2.0, 0.5])
    assert abs(_categorical_kl(logits, logits)) < 1e-12


def test_dose_eligibility_requires_every_frozen_gate() -> None:
    protocol = DynamicCalibrationProtocol()
    summary = {
        "median_correlation_change": -0.2,
        "median_effective_rank_change": 0.15,
        "median_output_kl": 0.02,
        "mean_top1_agreement": 0.9,
        "invariants": {
            "maximum_mean_error": 1e-7,
            "maximum_scale_error": 1e-7,
            "nonfinite_count": 0,
            "zero_variance_count": 0,
        },
    }
    assert _eligible(summary, protocol)
    summary["median_output_kl"] = 0.081
    assert not _eligible(summary, protocol)
