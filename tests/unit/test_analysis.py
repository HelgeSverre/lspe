import json
from pathlib import Path

import numpy as np

from lspe.analysis.bootstrap import paired_bootstrap
from lspe.analysis.runner import analyze_run
from lspe.analysis.status import StatusInputs, classify_status
from lspe.analysis.tests import holm_adjust


def test_cluster_bootstrap_resamples_prompts() -> None:
    result = paired_bootstrap(np.array([0.2, 0.1, -0.1]), seed=1, samples=1000)
    assert result.estimate == np.mean([0.2, 0.1, -0.1])
    assert result.positive == 2


def test_holm_adjustment() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_report_status_rules() -> None:
    base = dict(
        integrity_ok=True,
        h1_estimate=0.1,
        h1_ci95=(0.01, 0.2),
        validity_retained=True,
        degeneration_retained=True,
        coherent_beats_white_vsd=True,
        coherent_competence_not_worse_than_white=True,
        replication_positive=True,
    )
    assert classify_status(StatusInputs(**base)) == "SUPPORTED"
    assert classify_status(StatusInputs(**{**base, "integrity_ok": False})) == "INVALID_RUN"
    assert (
        classify_status(StatusInputs(**{**base, "diversity_due_to_failure": True}))
        == "DEGENERATIVE"
    )


def test_analysis_reports_h2_competence_and_degeneration(tmp_path: Path) -> None:
    rows = []
    for index, split in enumerate(("controls", "pilot", "pilot"), 1):
        for condition, vsd, validity, degeneration in (
            ("baseline", 0.20, 1.0, 0.00),
            ("coherent", 0.40, 1.0, 0.01),
            ("white", 0.10, 0.5, 0.10),
            ("temp_match", 0.25, 1.0, 0.00),
        ):
            rows.append(
                {
                    "prompt_id": f"p{index}",
                    "split": split,
                    "condition": condition,
                    "vsd": vsd,
                    "validity_rate": validity,
                    "degeneration_rate": degeneration,
                }
            )
    (tmp_path / "prompt-effects.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = analyze_run(tmp_path, master_seed=7, bootstrap_samples=100)
    assert np.isclose(result["secondary"]["h2_coherent_minus_white_vsd"], 0.3)
    assert np.isclose(result["competence"]["coherent_minus_white"], 0.5)
    assert result["degeneration"]["coherent_minus_white"] < 0
