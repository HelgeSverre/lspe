"""Frozen embedding and prompt-clustered analysis for SCBE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.bootstrap import paired_bootstrap
from ..analysis.tests import holm_adjust
from ..hashing import sha256_file
from ..metrics.deterministic import valid_semantic_diversity
from .behavioral_runner import CONDITIONS, PROTECTED, BehavioralProtocol


def analyze_behavioral_experiment(
    run_dir: Path,
    data_root: Path,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_revision: str | None = None,
    protocol: BehavioralProtocol | None = None,
) -> dict[str, Any]:
    """Embed complete answers and apply the preregistered paired decisions."""

    protocol = protocol or BehavioralProtocol()
    result = json.loads((run_dir / "result.json").read_text())
    if result["status"] != "GENERATION_COMPLETE":
        raise RuntimeError("SCBE analysis requires completed confirmation generation")
    generations = _read_jsonl(run_dir / "confirm-generations.jsonl")
    prompts = {row["prompt_id"]: row for row in _read_jsonl(data_root / "confirm.jsonl")}
    texts = [str(row["output_text"]) for row in generations]
    embeddings = _embed(texts, embedding_model, embedding_revision)
    individual = []
    grouped: dict[tuple[str, str], list[tuple[bool, np.ndarray]]] = {}
    for row, vector in zip(generations, embeddings, strict=True):
        individual.append(
            {
                "generation_id": row["generation_id"],
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "condition": row["condition"],
                "valid": row["valid"],
                "embedding": vector.tolist(),
            }
        )
        grouped.setdefault((row["prompt_id"], row["condition"]), []).append(
            (bool(row["valid"]), vector)
        )
    cells = []
    for (prompt_id, condition), values in sorted(grouped.items()):
        valid, vectors = zip(*values, strict=True)
        cells.append(
            {
                "prompt_id": prompt_id,
                "category": prompts[prompt_id]["category"],
                "condition": condition,
                "validity": float(np.mean(valid)),
                "vsd": valid_semantic_diversity(list(valid), np.stack(vectors)),
            }
        )
    _write_jsonl(run_dir / "behavioral-scores.jsonl", individual)
    _write_jsonl(run_dir / "behavioral-prompt-effects.jsonl", cells)
    by_prompt: dict[str, dict[str, dict[str, Any]]] = {}
    for cell in cells:
        by_prompt.setdefault(cell["prompt_id"], {})[cell["condition"]] = cell
    primary_names = ("open_association", "analogical")
    primary = {}
    raw_p = []
    for category in primary_names:
        values = np.array(
            [
                conditions["sccf"]["vsd"] - conditions["temp_match"]["vsd"]
                for prompt_id, conditions in by_prompt.items()
                if prompts[prompt_id]["category"] == category
            ]
        )
        outcome = paired_bootstrap(
            values,
            _seed(protocol.master_seed, "primary", category),
            protocol.bootstrap_samples,
        )
        primary[category] = _bootstrap_record(outcome)
        raw_p.append(outcome.sign_flip_p_value)
    for category, adjusted in zip(primary_names, holm_adjust(raw_p), strict=True):
        primary[category]["p_value_holm"] = adjusted
        primary[category]["supported"] = bool(
            primary[category]["ci95"][0] > 0.0 and adjusted < 0.05
        )
    secondary = {}
    for comparison in ("baseline", "random_basis", "attn_noise"):
        for category in primary_names + ("narrative",):
            values = np.array(
                [
                    conditions["sccf"]["vsd"] - conditions[comparison]["vsd"]
                    for prompt_id, conditions in by_prompt.items()
                    if prompts[prompt_id]["category"] == category
                ]
            )
            outcome = paired_bootstrap(
                values,
                _seed(protocol.master_seed, "secondary", comparison, category),
                protocol.bootstrap_samples,
            )
            secondary[f"sccf_minus_{comparison}:{category}"] = _bootstrap_record(outcome)
    confirmation = result["confirm"]
    competence = {category: confirmation["gates"][f"{category}_validity"] for category in PROTECTED}
    competence["all_degeneration"] = all(
        value for key, value in confirmation["gates"].items() if key.endswith("_degeneration")
    )
    deterministic_pass = all(competence.values())
    judge = _judge_summary(run_dir)
    if judge is None:
        raise RuntimeError("SCBE analysis requires the frozen blinded judge artifact")
    judge_pass = judge["gates_passed"]
    supported = [category for category in primary_names if primary[category]["supported"]]
    status = (
        "DEGENERATIVE"
        if not deterministic_pass or not judge_pass
        else "BEHAVIORAL_SUPPORT"
        if supported
        else "MECHANISM_ONLY"
    )
    analysis = {
        "schema_version": 1,
        "status": status,
        "primary": primary,
        "supported_families": supported,
        "secondary": secondary,
        "competence": competence,
        "judge": judge,
        "integrity": {
            "generation_count": len(generations),
            "prompt_cell_count": len(cells),
            "complete": len(generations) == len(prompts) * len(CONDITIONS) * 3,
            "generation_sha256": sha256_file(run_dir / "confirm-generations.jsonl"),
            "embedding_model": embedding_model,
            "embedding_revision": embedding_revision,
        },
    }
    _write_json(run_dir / "behavioral-analysis.json", analysis)
    _refresh_checksums(run_dir)
    return analysis


def _embed(texts: list[str], model: str, revision: str | None) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("SCBE embeddings require the analysis extra") from error
    encoder = SentenceTransformer(model, revision=revision, trust_remote_code=False)
    values = np.asarray(
        encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float64,
    )
    if values.ndim != 2 or not np.isfinite(values).all():
        raise RuntimeError("SCBE embedding output is malformed")
    return values


def _judge_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "behavioral-judge.jsonl"
    if not path.exists():
        return None
    rows = _read_jsonl(path)
    parsed = [row for row in rows if row.get("ratings") is not None]
    parse_rate = len(parsed) / len(rows)
    differences: dict[str, list[float]] = {"coherence": [], "plausibility": []}
    for row in parsed:
        if row["comparison"] != "baseline":
            continue
        for metric in differences:
            differences[metric].append(
                float(row["ratings"]["sccf"][metric]) - float(row["ratings"]["control"][metric])
            )
    means = {
        metric: float(np.mean(values)) if values else None for metric, values in differences.items()
    }
    gates = {
        "parse_rate": parse_rate >= 0.95,
        "coherence": means["coherence"] is not None and means["coherence"] >= -0.25,
        "plausibility": means["plausibility"] is not None and means["plausibility"] >= -0.25,
    }
    return {
        "comparisons": len(rows),
        "parse_rate": parse_rate,
        "baseline_differences": means,
        "gates": gates,
        "gates_passed": all(gates.values()),
    }


def _bootstrap_record(outcome: Any) -> dict[str, Any]:
    return {
        "estimate": outcome.estimate,
        "median": outcome.median,
        "ci95": list(outcome.ci95),
        "p_value": outcome.sign_flip_p_value,
        "positive": outcome.positive,
        "zero": outcome.zero,
        "negative": outcome.negative,
        "standardized_effect": outcome.standardized_effect,
    }


def _seed(master: int, *parts: Any) -> int:
    import hashlib

    payload = json.dumps([master, parts], sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_checksums(run_dir: Path) -> None:
    entries = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    }
    (run_dir / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in entries.items())
    )
