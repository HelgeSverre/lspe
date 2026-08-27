"""Model-free environment diagnostics."""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import LspeConfig


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    platform: str
    python: str
    machine: str
    available_disk_gb: float
    checks: dict[str, bool]
    warnings: list[str]


def inspect_environment(config: LspeConfig, workspace: Path) -> DoctorReport:
    target = workspace.resolve()
    disk = shutil.disk_usage(target)
    actual_platform = f"{sys.platform}-{platform.machine()}"
    checks = {
        "platform": actual_platform == config.hardware.expected_platform,
        "python": sys.version_info >= (3, 12),
        "output_parent_writable": _is_writable(target),
    }
    warnings: list[str] = []
    if not checks["platform"]:
        warnings.append(
            f"Expected {config.hardware.expected_platform}; detected {actual_platform}. "
            "Model-backed runs should be performed on the locked target platform."
        )
    if not checks["python"]:
        warnings.append("Python 3.12 or later is required.")
    return DoctorReport(
        ok=checks["python"] and checks["output_parent_writable"],
        platform=actual_platform,
        python=platform.python_version(),
        machine=platform.platform(),
        available_disk_gb=round(disk.free / 1024**3, 2),
        checks=checks,
        warnings=warnings,
    )


def report_dict(report: DoctorReport) -> dict[str, object]:
    return asdict(report)


def _is_writable(path: Path) -> bool:
    try:
        probe = path / ".lspe-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True
