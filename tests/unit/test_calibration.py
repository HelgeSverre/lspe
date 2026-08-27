import numpy as np
import pytest

from lspe.calibration import distribution_divergence, match_temperature, sampling_entropy
from lspe.calibration.runner import select_matching_white_dose, select_target_dose


def test_teacher_forced_kl_uses_identical_prefixes() -> None:
    baseline = np.array([2.0, 0.0, -1.0])
    assert distribution_divergence(baseline, baseline).kl_altered_baseline == 0.0


def test_entropy_computation_after_filtering() -> None:
    logits = np.array([3.0, 2.0, 1.0])
    assert sampling_entropy(logits, 1.0, top_k=1, top_p=1.0) == 0.0


def test_temperature_matcher_converges() -> None:
    logits = [np.array([2.0, 1.0, 0.0]), np.array([1.0, 0.0, -1.0])]
    target = np.mean([sampling_entropy(row, 0.8, 0, 1.0) for row in logits])
    result = match_temperature(logits, target, 0, 1.0, steps=301)
    assert abs(result.temperature - 0.8) < 0.02
    assert result.absolute_mismatch < 0.01


def test_target_dose_rejects_a_grid_that_skips_the_preregistered_band() -> None:
    with pytest.raises(ValueError, match="cannot resolve target KL band"):
        select_target_dose({0.3: 0.0063, 0.6: 0.718}, 0.01)


def test_target_dose_accepts_a_resolved_band() -> None:
    assert select_target_dose({0.3: 0.0063, 0.4: 0.095, 0.6: 0.718}, 0.1) == 0.4


def test_white_dose_rejects_an_unmatched_target_band() -> None:
    with pytest.raises(ValueError, match="cannot match white-noise KL"):
        select_matching_white_dose({0.5: 0.079}, coherent_kl=0.314, target_kl=0.3)
