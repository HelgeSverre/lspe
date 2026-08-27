"""Explicit Hugging Face model acquisition with gated-repository diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ModelConfig
from .hashing import sha256_file


@dataclass(frozen=True)
class FetchResult:
    repo_id: str
    revision: str
    local_path: Path
    weight_files: dict[str, str]


def fetch_model(model: ModelConfig, offline: bool = False) -> FetchResult:
    """Fetch exactly the configured revision or explain why authentication is required."""

    if model.local_path is not None:
        local = model.local_path.expanduser().resolve()
        if not local.is_dir():
            raise RuntimeError(
                f"Configured local model path does not exist or is not a directory: {local}"
            )
        if not model.revision:
            raise RuntimeError(
                "A local model path still requires model.revision to record "
                "its immutable source commit."
            )
        return _build_result(model.repo_id, model.revision, local)
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            GatedRepoError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )
    except (
        ImportError
    ) as error:  # pragma: no cover - locked dependency is tested at CLI install time
        raise RuntimeError("huggingface-hub is required; run `uv sync --frozen`.") from error
    source = str(model.local_path) if model.local_path else model.repo_id
    try:
        local = Path(
            snapshot_download(
                repo_id=source,
                revision=model.revision,
                local_files_only=offline,
            )
        )
    except GatedRepoError as error:
        raise RuntimeError(
            f"Access to {model.repo_id} is gated. Accept its license on Hugging Face and "
            "authenticate "
            "with `hf auth login`, then rerun `lspe fetch`. No fallback model was selected."
        ) from error
    except RepositoryNotFoundError as error:
        raise RuntimeError(
            f"Model repository {model.repo_id} is unavailable or authentication is missing. "
            "Verify the repository ID and run `hf auth login`."
        ) from error
    except LocalEntryNotFoundError as error:
        raise RuntimeError(
            "Offline mode is enabled but the requested model revision is not in the local cache. "
            "Run `lspe fetch` once without --offline after accepting any model license."
        ) from error
    revision = _resolved_revision(local, model.revision)
    return _build_result(model.repo_id, revision, local)


def _build_result(repo_id: str, revision: str, local: Path) -> FetchResult:
    weights = {
        file.relative_to(local).as_posix(): sha256_file(file)
        for file in sorted(local.rglob("*"))
        if file.is_file() and file.suffix in {".safetensors", ".npz", ".gguf"}
    }
    if not weights:
        raise RuntimeError(f"No recognized weight files were fetched for {repo_id} at {local}")
    return FetchResult(repo_id, revision, local, weights)


def _resolved_revision(local: Path, requested: str | None) -> str:
    """Prefer the immutable cache snapshot commit; never report a mutable branch name as locked."""

    snapshots = local.parts
    if "snapshots" in snapshots:
        index = snapshots.index("snapshots")
        if index + 1 < len(snapshots):
            return snapshots[index + 1]
    if requested and len(requested) >= 7:
        return requested
    raise RuntimeError("Unable to resolve the fetched model to an immutable Hugging Face commit")
