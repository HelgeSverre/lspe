"""Cross-run report that preserves, rather than pools, model-specific results."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ..hashing import sha256_file


def build_combined_report(
    output_dir: Path, primary_run: Path, replication_run: Path
) -> dict[str, Any]:
    """Write JSON, Markdown, and HTML linking two verified LSPE run reports.

    Models have different architectures and calibrated raw doses, so this
    report deliberately displays their effect estimates side by side rather
    than calculating a statistically invalid pooled estimate.
    """

    primary_path = primary_run / "report.json"
    replication_path = replication_run / "report.json"
    primary = _read_report(primary_path)
    replication = _read_report(replication_path)
    primary_supported = _positive_primary(primary)
    replication_supported = _positive_primary(replication)
    degenerative = any(
        report.get("status") == "DEGENERATIVE" for report in (primary, replication)
    )
    if degenerative:
        status = "DEGENERATIVE"
    elif primary_supported and replication_supported:
        status = "SUPPORTED_REPLICATED"
    elif primary_supported:
        status = "SUPPORTED_UNREPLICATED"
    else:
        status = "NOT_SUPPORTED"
    payload = {
        "schema_version": 1,
        "primary": _summary(primary, primary_path),
        "architecture_replication": _summary(replication, replication_path),
        "conclusion": {
            "status": status,
            "h1": (
                "SUPPORTED: the locked primary estimate and confidence interval are positive."
                if primary_supported
                else "NOT_SUPPORTED: the locked primary confidence interval is not positive."
            ),
            "h3": _replication_conclusion(primary_supported, replication_supported),
            "pooling": (
                "Not performed: the runs use different model architectures and "
                "independently calibrated raw doses."
            ),
        },
        "non_claims": (
            "These activation-perturbation experiments make no claim about "
            "consciousness, intoxication, psychedelics, biological equivalence, "
            "or human evolution."
        ),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    markdown = "\n".join(
        [
            "# LSPE combined experiment report",
            "",
            f"Scientific status: **{payload['conclusion']['status']}**",
            "",
            "## Cross-model conclusion",
            "",
            f"- {payload['conclusion']['h1']}",
            f"- {payload['conclusion']['h3']}",
            f"- {payload['conclusion']['pooling']}",
            "",
            "## Source run summaries",
            "",
            "```json",
            encoded,
            "```",
            "",
            "## Non-claims",
            "",
            payload["non_claims"],
        ]
    )
    page = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            "<title>LSPE combined report</title></head><body>",
            "<h1>LSPE combined experiment report</h1>",
            "<p>Scientific status: "
            f"<strong>{html.escape(payload['conclusion']['status'])}</strong></p>",
            f"<p>{html.escape(payload['conclusion']['h1'])}</p>",
            f"<p>{html.escape(payload['conclusion']['h3'])}</p>",
            f"<pre>{html.escape(encoded)}</pre>",
            "</body></html>",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "combined.json").write_text(encoded + "\n", encoding="utf-8")
    (output_dir / "combined.md").write_text(markdown + "\n", encoding="utf-8")
    (output_dir / "combined.html").write_text(page + "\n", encoding="utf-8")
    return payload


def _read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "primary" not in value or "run_id" not in value:
        raise ValueError(f"Invalid LSPE run report: {path}")
    return value


def _summary(report: dict[str, Any], path: Path) -> dict[str, Any]:
    primary = report["primary"]
    return {
        "run_id": report["run_id"],
        "status": report.get("status"),
        "report_sha256": sha256_file(path),
        "primary": {
            "contrast": primary.get("contrast"),
            "estimate": primary.get("estimate"),
            "ci95": primary.get("ci95"),
            "p_value": primary.get("p_value"),
            "n_prompts": primary.get("n_prompts"),
        },
    }


def _positive_primary(report: dict[str, Any]) -> bool:
    primary = report["primary"]
    estimate = primary.get("estimate")
    ci95 = primary.get("ci95")
    return bool(
        isinstance(estimate, (int, float))
        and isinstance(ci95, list)
        and len(ci95) == 2
        and isinstance(ci95[0], (int, float))
        and estimate > 0
        and ci95[0] > 0
    )


def _replication_conclusion(primary_supported: bool, replication_supported: bool) -> str:
    if not primary_supported:
        return (
            "NOT_SUPPORTED: replication cannot establish a positive-direction result "
            "when the locked primary effect is not supported."
        )
    if replication_supported:
        return "SUPPORTED: the positive primary direction replicated."
    return "NOT_SUPPORTED: the replication did not reproduce the positive primary direction."
