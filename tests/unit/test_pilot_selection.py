import json
from pathlib import Path

from lspe.config import load_config
from lspe.pilot_selection import PilotCandidate, select_pilot_candidate, select_pilot_matrix


def test_selection_retains_explicit_no_eligible_result(tmp_path: Path) -> None:
    run = tmp_path / "pilot"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"selected_layers": [20]}), encoding="utf-8")
    rows = []
    for condition, vsd, validity, degeneration in (
        ("baseline", 0.2, 1.0, 0.0),
        ("coherent", 0.3, 0.8, 0.1),
        ("white", 0.1, 1.0, 0.0),
        ("temp_match", 0.1, 1.0, 0.0),
    ):
        rows.append(
            {
                "prompt_id": "p1",
                "condition": condition,
                "vsd": vsd,
                "validity_rate": validity,
                "degeneration_rate": degeneration,
            }
        )
    (run / "prompt-effects.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    config = load_config("configs/pilot.gemma4-e4b.yaml")
    result = select_pilot_candidate(
        run,
        config,
        {"summary": {"raw_dose": 0.01, "achieved_median_kl": 0.1}},
    )
    assert result.status == "NO_ELIGIBLE_INTERVENTION"
    assert "VALIDITY_NONINFERIORITY_FAILED" in result.selected.eligibility_reasons
    assert (run / "pilot-selection.json").is_file()


def test_matrix_selection_uses_utility_then_lower_dose() -> None:
    base = dict(
        selected_layers=(20,),
        target_kl=0.1,
        achieved_median_kl=0.1,
        white_raw_dose=0.01,
        white_achieved_median_kl=0.1,
        validity_baseline=1.0,
        validity_coherent=1.0,
        degeneration_baseline=0.0,
        degeneration_coherent=0.0,
        vsd_coherent_temp=0.1,
        vsd_coherent_white=0.1,
        utility=0.2,
        eligible=True,
        eligibility_reasons=(),
    )
    result = select_pilot_matrix(
        [
            PilotCandidate(candidate_id="higher", raw_dose=0.03, **base),
            PilotCandidate(candidate_id="lower", raw_dose=0.01, **base),
        ]
    )
    assert result.status == "ELIGIBLE"
    assert result.selected.candidate_id == "lower"
