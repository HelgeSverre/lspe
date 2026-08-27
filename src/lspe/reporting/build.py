"""Small dependency-free report renderer for reproducible run summaries."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ..artifact_schema import validate_report
from .plots import build_plots

NON_CLAIMS = (
    "This run tests whether a transient internal activation perturbation changes model behaviour "
    "beyond output sampling randomness. It makes no claim about consciousness, intoxication, "
    "psychedelics, or biological equivalence."
)
STYLE = (
    "<style>body{font-family:system-ui,sans-serif;max-width:72rem;margin:2rem auto;padding:0 1rem}"
    "pre{overflow:auto;background:#f5f5f5;padding:1rem}</style>"
)

REQUIRED_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "status",
        "primary",
        "validity",
        "competence",
        "degeneration",
        "replication",
        "integrity",
        "artifact_root_hash",
    }
)


def build_report(run_dir: Path, report: dict[str, Any]) -> None:
    """Render identical facts into JSON, Markdown, and portable HTML."""

    missing = sorted(REQUIRED_REPORT_KEYS - set(report))
    if missing:
        raise ValueError(f"Report is missing required fields: {', '.join(missing)}")
    report = validate_report(report)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    primary = report["primary"]
    markdown = "\n".join(
        [
            f"# LSPE report: {report['run_id']}",
            "",
            f"Scientific status: **{report['status']}**",
            "",
            "## Question and non-claims",
            "",
            NON_CLAIMS,
            "",
            "## Primary outcome",
            "",
            f"- Metric: `{primary.get('metric')}`",
            f"- Contrast: `{primary.get('contrast')}`",
            f"- Estimate: `{primary.get('estimate')}`",
            f"- 95% CI: `{primary.get('ci95')}`",
            f"- p-value: `{primary.get('p_value')}`",
            f"- Prompt clusters: `{primary.get('n_prompts')}`",
            "",
            "## Competence, degeneration, and secondary outcomes",
            "",
            "```json",
            json.dumps(
                {
                    "competence": report["competence"],
                    "degeneration": report["degeneration"],
                    "secondary": report["secondary"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Integrity and complete execution",
            "",
            "```json",
            json.dumps(
                {"integrity": report["integrity"], "execution": report["execution"]},
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Replication and limitations",
            "",
            "```json",
            json.dumps(
                {"replication": report["replication"], "limitations": report["limitations"]},
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Complete machine-readable record",
            "",
            "```json",
            encoded,
            "```",
            "",
            "## Reproduction",
            "",
            f"`uv run lspe verify --run {run_dir} --level artifact`",
        ]
    )
    page = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8"><title>LSPE report</title>',
            STYLE,
            "</head><body>",
            f"<h1>LSPE report: {html.escape(str(report['run_id']))}</h1>",
            f"<p>Scientific status: <strong>{html.escape(str(report['status']))}</strong></p>",
            "<h2>Question and non-claims</h2>",
            f"<p>{html.escape(NON_CLAIMS)}</p>",
            "<h2>Complete machine-readable record</h2>",
            f"<pre>{html.escape(encoded)}</pre>",
            "<h2>Limitations</h2>",
            f"<pre>{html.escape(json.dumps(report['limitations'], indent=2))}</pre>",
            "</body></html>",
        ]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(encoded + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(markdown + "\n", encoding="utf-8")
    (run_dir / "report.html").write_text(page + "\n", encoding="utf-8")
    build_plots(run_dir)
