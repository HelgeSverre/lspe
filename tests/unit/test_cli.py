import json

from lspe.cli import _calibration_curve_matches
from lspe.config import load_config


def test_resume_reuses_only_matching_calibration_curve(tmp_path) -> None:
    loaded = load_config("configs/pilot.gemma4-e4b-v4.yaml")
    config = loaded.model_copy(
        update={"intervention": loaded.intervention.model_copy(update={"selected_layers": [14]})}
    )
    revision = "475b9088d29754a3379866cf5aeb6b41acd313c2"
    doses = {str(dose): 0.0 for dose in config.intervention.raw_dose_grid}
    curve_path = tmp_path / "calibration.json"
    curve_path.write_text(
        json.dumps(
            {
                "model_revision": revision,
                "summary": {"selected_layers": [14]},
                "median_kl_by_raw_dose": {
                    "coherent_per_layer": doses,
                    "white_per_token": doses,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _calibration_curve_matches(config, revision, curve_path)
    assert not _calibration_curve_matches(config, "other-revision", curve_path)
