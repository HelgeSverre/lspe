import json
from pathlib import Path

from lspe.reporting.combined import build_combined_report


def test_combined_report_keeps_model_estimates_separate(tmp_path: Path) -> None:
    def write_report(
        directory: Path,
        run_id: str,
        estimate: float,
        ci95: list[float] | None = None,
        status: str = "DEGENERATIVE",
    ) -> None:
        directory.mkdir()
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": status,
                    "primary": {
                        "contrast": "coherent-temp_match",
                        "estimate": estimate,
                        "ci95": ci95 or [-0.1, 0.1],
                        "p_value": 0.5,
                        "n_prompts": 60,
                    },
                }
            ),
            encoding="utf-8",
        )

    primary = tmp_path / "primary"
    replication = tmp_path / "replication"
    write_report(primary, "gemma", -0.01)
    write_report(replication, "qwen", 0.002)
    payload = build_combined_report(tmp_path / "combined", primary, replication)
    assert payload["primary"]["primary"]["estimate"] == -0.01
    assert payload["architecture_replication"]["primary"]["estimate"] == 0.002
    assert payload["conclusion"]["status"] == "DEGENERATIVE"
    assert (tmp_path / "combined" / "combined.html").is_file()


def test_combined_report_derives_positive_replication_conclusion(tmp_path: Path) -> None:
    def write_report(directory: Path, run_id: str) -> None:
        directory.mkdir()
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "SUPPORTED",
                    "primary": {
                        "contrast": "coherent-temp_match",
                        "estimate": 0.1,
                        "ci95": [0.02, 0.18],
                        "p_value": 0.01,
                        "n_prompts": 60,
                    },
                }
            ),
            encoding="utf-8",
        )

    primary = tmp_path / "primary"
    replication = tmp_path / "replication"
    write_report(primary, "model-a")
    write_report(replication, "model-b")
    payload = build_combined_report(tmp_path / "combined", primary, replication)
    assert payload["conclusion"]["status"] == "SUPPORTED_REPLICATED"
    assert payload["conclusion"]["h1"].startswith("SUPPORTED")
    assert payload["conclusion"]["h3"].startswith("SUPPORTED")
