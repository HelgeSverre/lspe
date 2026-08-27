from lspe.memory_guard import MemoryGuard


def test_memory_guard_allows_unknown_physical_memory(monkeypatch: object) -> None:
    monkeypatch.setattr("lspe.memory_guard.physical_memory_bytes", lambda: 0)
    MemoryGuard(0.8, 0.9).enforce()


def test_memory_guard_fails_closed_at_hard_limit(monkeypatch: object) -> None:
    monkeypatch.setattr("lspe.memory_guard.physical_memory_bytes", lambda: 100)
    monkeypatch.setattr("lspe.memory_guard.peak_process_rss_bytes", lambda: 95)
    try:
        MemoryGuard(0.8, 0.9).enforce()
    except MemoryError as error:
        assert "MEMORY_LIMIT" in str(error)
    else:
        raise AssertionError("Expected hard memory guard to stop execution")
