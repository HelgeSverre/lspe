"""Conservative process-memory guard for sequential local model execution."""

from __future__ import annotations

import os
import platform
import resource
import subprocess


def physical_memory_bytes() -> int:
    """Read physical memory without adding a runtime dependency."""

    if platform.system() == "Darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError):
        return 0


def peak_process_rss_bytes() -> int:
    """Return the best portable process RSS figure available to the harness."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


class MemoryGuard:
    def __init__(self, soft_fraction: float, hard_fraction: float) -> None:
        self.total_bytes = physical_memory_bytes()
        self.soft_fraction = soft_fraction
        self.hard_fraction = hard_fraction

    def enforce(self) -> None:
        if self.total_bytes <= 0:
            return
        rss = peak_process_rss_bytes()
        if rss >= self.total_bytes * self.hard_fraction:
            raise MemoryError(
                f"MEMORY_LIMIT: process RSS {rss} exceeds hard limit "
                f"{self.total_bytes * self.hard_fraction:.0f}"
            )
        if rss >= self.total_bytes * self.soft_fraction:
            raise MemoryError(
                f"MEMORY_SOFT_LIMIT: process RSS {rss} exceeds soft limit "
                f"{self.total_bytes * self.soft_fraction:.0f}; state is durable and may be resumed"
            )
